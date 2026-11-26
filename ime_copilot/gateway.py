"""Model gateway: the single door every model call goes through.

- Routes each task (classify, extract, questions) to the model set in config,
  so swapping a vendor, or later our own model, is a config change.
- Retries with backoff, meters tokens and cost per case.
- Logs metadata only (token counts, cost, a hash of the prompt), never the
  prompt or output text, because those contain PHI.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Callable, Protocol

from .config import Settings, price_for
from .db import log_model_call


class Provider(Protocol):
    def complete(self, model_id: str, system: str, user: str, max_tokens: int) -> tuple[str, int, int]:
        """Return (text, input_tokens, output_tokens)."""


class AzureOpenAIProvider:
    """Azure OpenAI / Foundry deployments through the v1 API (standard OpenAI client).

    model_id is the *deployment name* in your Azure resource. Auth uses
    Microsoft Entra ID (role: Cognitive Services OpenAI User) unless a key is set.
    Only text is ever sent: never PDFs or page images, whose HIPAA coverage
    under the Microsoft BAA is not established.
    """

    def __init__(self, endpoint: str, api_key: str = "", client=None):
        if client is not None:
            self.client = client
            return
        if not endpoint:
            raise ValueError("Set AZURE_OPENAI_ENDPOINT, e.g. https://<resource>.openai.azure.com")
        from openai import OpenAI  # pip install ime-copilot[azure]
        base = endpoint.rstrip("/")
        if not base.endswith("/openai/v1"):
            base += "/openai/v1"
        if api_key:
            key = api_key
        else:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
            key = get_bearer_token_provider(DefaultAzureCredential(), "https://ai.azure.com/.default")
        self.client = OpenAI(base_url=base + "/", api_key=key)

    def complete(self, model_id: str, system: str, user: str, max_tokens: int) -> tuple[str, int, int]:
        resp = self.client.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            # Reasoning models count thinking tokens here too, so leave headroom.
            max_completion_tokens=max(max_tokens, 1000),
        )
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return text, getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0

    def embed(self, model_id: str, texts: list[str]) -> tuple[list[list[float]], int]:
        resp = self.client.embeddings.create(model=model_id, input=texts)
        return [d.embedding for d in resp.data], getattr(resp.usage, "prompt_tokens", 0) or 0


class LocalProvider:
    """A model running on this machine (Ollama, LM Studio, vLLM...).

    Nothing leaves the machine. Default: Ollama at http://localhost:11434/v1.
    For Ollama, answers use its native /api/chat endpoint so we can set the context
    window (Ollama's default is small and silently cuts long prompts), turn off
    "thinking" and force JSON output. Other servers use the OpenAI-style API.
    """

    def __init__(self, base_url: str = "http://localhost:11434/v1", client=None, timeout: float = 600,
                 num_ctx: int = 16384):
        self.timeout, self.num_ctx = timeout, num_ctx
        self.native_url = None
        if client is not None:
            self.client = client
            return
        from openai import OpenAI  # pip install openai
        self.client = OpenAI(base_url=base_url, api_key="local", timeout=timeout, max_retries=0)
        if ":11434" in base_url:
            self.native_url = base_url.rstrip("/").removesuffix("/v1") + "/api/chat"
        self._think_ok = True

    def complete(self, model_id: str, system: str, user: str, max_tokens: int) -> tuple[str, int, int]:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self.native_url:
            return self._ollama_chat(model_id, messages, max(max_tokens, 1000))
        resp = self.client.chat.completions.create(
            model=model_id, temperature=0, max_tokens=max(max_tokens, 1000), messages=messages,
        )
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return text, getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0

    def _ollama_chat(self, model_id: str, messages: list, max_tokens: int) -> tuple[str, int, int]:
        import urllib.error
        import urllib.request
        body = {"model": model_id, "messages": messages, "stream": False, "format": "json",
                "options": {"temperature": 0, "num_ctx": self.num_ctx, "num_predict": max_tokens}}
        if self._think_ok:
            body["think"] = False
        req = urllib.request.Request(self.native_url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as err:
            detail = err.read().decode(errors="replace")[:200]
            if self._think_ok and "think" in detail.lower():   # model without a thinking switch
                self._think_ok = False
                return self._ollama_chat(model_id, messages, max_tokens)
            raise RuntimeError(f"Ollama error {err.code}: {detail}") from err
        except urllib.error.URLError as err:
            raise RuntimeError(f"Connection error: {err.reason}") from err
        text = (data.get("message") or {}).get("content") or ""
        return text, data.get("prompt_eval_count", 0) or 0, data.get("eval_count", 0) or 0

    def embed(self, model_id: str, texts: list[str]) -> tuple[list[list[float]], int]:
        resp = self.client.embeddings.create(model=model_id, input=texts)
        return [d.embedding for d in resp.data], getattr(getattr(resp, "usage", None), "prompt_tokens", 0) or 0


class BedrockProvider:
    """AWS Bedrock Converse API (HIPAA-eligible under the AWS BAA)."""

    def __init__(self, region: str):
        import boto3  # optional dependency: pip install ime-copilot[aws]
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def complete(self, model_id: str, system: str, user: str, max_tokens: int) -> tuple[str, int, int]:
        resp = self.client.converse(
            modelId=model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0},
        )
        parts = resp["output"]["message"]["content"]
        text = "".join(p.get("text", "") for p in parts)
        usage = resp.get("usage", {})
        return text, usage.get("inputTokens", 0), usage.get("outputTokens", 0)


class FakeProvider:
    """Offline stand-in for tests: a function that maps (task_model, user) -> text."""

    def __init__(self, responder: Callable[[str, str], str]):
        self.responder = responder

    def complete(self, model_id: str, system: str, user: str, max_tokens: int) -> tuple[str, int, int]:
        text = self.responder(model_id, user)
        return text, len(user) // 4, len(text) // 4


class FoundryClaudeProvider:
    """Claude models deployed in Microsoft Foundry (Anthropic Messages API).

    model_id is the deployment name (e.g. claude-sonnet-5-5). Use deployments
    "Hosted on Azure" with region scope "Data Zone" (US) so inference stays in
    Azure and in the US. Anthropic is the processor for prompts and outputs on
    Foundry, so patient data also needs Anthropic's BAA, confirmed in writing.
    """

    def __init__(self, resource: str = "", api_key: str = "", client=None):
        if client is not None:
            self.client = client
            return
        if not resource:
            raise ValueError("Set ANTHROPIC_FOUNDRY_RESOURCE to your Foundry resource name "
                             "(the part before .services.ai.azure.com).")
        from anthropic import AnthropicFoundry  # pip install ime-copilot[azure]
        if api_key:
            self.client = AnthropicFoundry(resource=resource, api_key=api_key)
        else:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
            token = get_bearer_token_provider(DefaultAzureCredential(), "https://ai.azure.com/.default")
            self.client = AnthropicFoundry(resource=resource, azure_ad_token_provider=token)

    def complete(self, model_id: str, system: str, user: str, max_tokens: int) -> tuple[str, int, int]:
        msg = self.client.messages.create(
            model=model_id, system=system, max_tokens=max(max_tokens, 2000),
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text")
        return text, msg.usage.input_tokens, msg.usage.output_tokens


PROVIDER_NAMES = ("local", "azure_openai", "foundry_claude", "bedrock")


def split_model_spec(spec: str, default_provider: str) -> tuple[str, str]:
    """'foundry_claude:claude-sonnet-5-5' -> ('foundry_claude', 'claude-sonnet-5-5');
    a bare deployment name uses IME_PROVIDER."""
    head, sep, tail = spec.partition(":")
    if sep and head in PROVIDER_NAMES:
        return head, tail
    return default_provider, spec


class GatewayError(RuntimeError):
    pass


class Gateway:
    """Routes each task to its provider + model. A task's model can name its own
    provider ('foundry_claude:claude-sonnet-5-5'), so e.g. drafting can use GPT
    while the QA cross-check uses Claude."""

    def __init__(self, settings: Settings, provider: Provider | None = None,
                 engine=None, case_id: str | None = None, retries: int = 3):
        self.settings = settings
        self.engine = engine
        self.case_id = case_id
        self.retries = retries
        self.cost_usd = 0.0
        self._injected = provider                  # tests: one fake provider for everything
        self._providers: dict[str, Provider] = {}

    @property
    def provider(self) -> Provider | None:      # backwards compatible: the default provider
        return self._injected or self._get(self.settings.provider)

    @property
    def enabled(self) -> bool:
        if self._injected is not None:
            return True
        names = {split_model_spec(m, self.settings.provider)[0] for m in self.settings.models.values() if m}
        return any(n in PROVIDER_NAMES for n in names | {self.settings.provider})

    def _get(self, name: str) -> Provider | None:
        if name not in PROVIDER_NAMES:
            return None
        if name not in self._providers:
            s = self.settings
            try:
                if name == "local":
                    self._providers[name] = LocalProvider(s.local_base_url, num_ctx=s.local_num_ctx)
                elif name == "azure_openai":
                    self._providers[name] = AzureOpenAIProvider(s.azure_openai_endpoint, s.azure_openai_api_key)
                elif name == "foundry_claude":
                    self._providers[name] = FoundryClaudeProvider(s.foundry_claude_resource,
                                                                  s.foundry_claude_api_key)
                else:
                    self._providers[name] = BedrockProvider(s.aws_region)
            except Exception as err:  # noqa: BLE001 - missing SDK or settings
                raise GatewayError(f"Cannot set up provider '{name}': {err}") from err
        return self._providers[name]

    def complete_json(self, task: str, system: str, user: str, max_tokens: int = 2000) -> Any:
        spec = self.settings.models.get(task)
        if not spec:
            raise GatewayError(f"No model configured for task '{task}'.")
        provider_name, model_id = split_model_spec(spec, self.settings.provider)
        provider = self._injected or self._get(provider_name)
        if provider is None:
            raise GatewayError("No model provider configured (set IME_PROVIDER=azure_openai or foundry_claude).")

        prompt_hash = hashlib.sha256((system + "\n" + user).encode()).hexdigest()
        last_err: Exception | None = None
        for attempt in range(self.retries):
            start = time.monotonic()
            try:
                text, tin, tout = provider.complete(model_id, system, user, max_tokens)
                cost = self._cost(task, model_id, tin, tout)
                self.cost_usd += cost
                self._log(task, model_id, tin, tout, cost, start, prompt_hash, ok=True)
                return parse_json(text)
            except Exception as err:  # noqa: BLE001 - provider errors vary by SDK
                last_err = err
                self._log(task, model_id, 0, 0, 0.0, start, prompt_hash, ok=False)
                time.sleep(min(2 ** attempt, 8) if attempt < self.retries - 1 else 0)
        raise GatewayError(f"{task} failed after {self.retries} attempts: {last_err}")

    def has_model(self, task: str) -> bool:
        spec = self.settings.models.get(task)
        if not spec:
            return False
        name, _ = split_model_spec(spec, self.settings.provider)
        return self._injected is not None or name in PROVIDER_NAMES

    def embed(self, texts: list[str], task: str = "embed", batch: int = 32) -> list[list[float]]:
        """Embedding vectors for search. Same routing and metering as completions."""
        spec = self.settings.models.get(task)
        if not spec:
            raise GatewayError("No embedding model configured (IME_MODEL_EMBED).")
        provider_name, model_id = split_model_spec(spec, self.settings.provider)
        provider = self._injected or self._get(provider_name)
        if provider is None or not hasattr(provider, "embed"):
            raise GatewayError(f"Provider '{provider_name}' cannot create embeddings.")
        out: list[list[float]] = []
        for i in range(0, len(texts), batch):
            start = time.monotonic()
            chunk = texts[i:i + batch]
            try:
                vectors, tokens = provider.embed(model_id, chunk)
            except Exception as err:  # noqa: BLE001
                self._log(task, model_id, 0, 0, 0.0, start, "", ok=False)
                raise GatewayError(f"Embedding failed: {err}") from err
            cost = self._cost(task, model_id, tokens, 0)
            self.cost_usd += cost
            self._log(task, model_id, tokens, 0, cost, start, "", ok=True)
            out.extend(vectors)
        return out

    def _cost(self, task: str, model_id: str, tin: int, tout: int) -> float:
        pin, pout = self.settings.task_prices.get(task) or price_for(model_id)
        return (tin * pin + tout * pout) / 1_000_000

    def _log(self, task, model_id, tin, tout, cost, start, prompt_hash, ok):
        log_model_call(
            self.engine, case_id=self.case_id, task=task, model_id=model_id,
            input_tokens=tin, output_tokens=tout, cost_usd=cost,
            latency_ms=int((time.monotonic() - start) * 1000),
            prompt_sha256=prompt_hash, ok=ok,
        )


def parse_json(text: str) -> Any:
    """Parse a model reply that should be JSON, tolerating code fences or prose around it."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"Model reply contained no JSON ({len(text)} characters returned"
                         f"{'; empty reply' if not text.strip() else ''}).")
    return json.loads(match.group(1))
