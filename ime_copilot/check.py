"""Connectivity check for Azure services, using no patient data.

  python -m ime_copilot check

Sends one tiny made-up prompt to each configured Azure OpenAI deployment, and
(if IME_OCR=azure_di) one synthetic page to Document Intelligence. Reports
OK / the exact error and what usually fixes it.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from .config import TASKS, Settings
from .gateway import Gateway, GatewayError

_HINTS = {
    "401": "Wrong key, or your identity lacks the role 'Cognitive Services OpenAI User' (Entra ID).",
    "403": "No permission: assign 'Cognitive Services OpenAI User' (Azure OpenAI) or 'Foundry User' / "
           "'Cognitive Services User' (Claude) on the resource, or its network rules block your IP.",
    "ANTHROPIC_FOUNDRY_RESOURCE": "Set ANTHROPIC_FOUNDRY_RESOURCE (and ANTHROPIC_FOUNDRY_API_KEY for a key) for Claude.",
    "No module named": "Install the Azure extras: pip install -e \".[azure]\"",
    "404": "Deployment name not found: IME_MODEL_* must equal the deployment name in Azure AI Foundry, "
           "and AZURE_OPENAI_ENDPOINT must be the resource URL (https://<name>.openai.azure.com).",
    "429": "Quota/rate limit: raise the deployment's tokens-per-minute quota or wait.",
    "DefaultAzureCredential": "No Azure sign-in found: run 'az login', or set the *_KEY variable for local tests.",
    "getaddrinfo": "Endpoint URL is wrong or unreachable (check spelling, private networking/VPN).",
    "Connection refused": "Ollama isn't running: open the Ollama app (llama icon in the menu bar) or run 'ollama serve'.",
    "Connection error": "Ollama isn't running: open the Ollama app (llama icon in the menu bar) or run 'ollama serve'.",
    "not found": "Model not downloaded: run 'ollama pull <model name>' (see 'ollama list').",
}


def _hint(err: Exception) -> str:
    text = str(err)
    for needle, hint in _HINTS.items():
        if needle in text:
            return hint
    return "See the error above."


def run_check(settings: Settings, provider=None, di_client=None) -> bool:
    ok = True
    print(f"Default provider: {settings.provider} | models: {settings.models}")
    print(f"  Local models (Ollama): {settings.local_base_url}")
    print(f"  Azure OpenAI endpoint: {settings.azure_openai_endpoint or '-'} "
          f"({'key' if settings.azure_openai_api_key else 'Entra ID'})")
    print(f"  Foundry (Claude) resource: {settings.foundry_claude_resource or '-'} "
          f"({'key' if settings.foundry_claude_api_key else 'Entra ID'})")
    gw = Gateway(settings, provider=provider)
    if not gw.enabled:
        print("  No model provider set: model checks skipped. Set IME_PROVIDER=azure_openai or foundry_claude.")
    else:
        seen = set()
        for task in TASKS:
            deployment = settings.models.get(task)
            if not deployment or deployment in seen:
                continue
            if task == "embed":
                t0 = time.monotonic()
                try:
                    vec = gw.embed(["lumbar epidural steroid injection"])[0]
                    print(f"  OK   embed      model={deployment:<22} {(time.monotonic() - t0):.1f}s  size={len(vec)}")
                except GatewayError as err:
                    print(f"  FAIL embed      model={deployment:<22} {err}\n       -> {_hint(err)}")
                    ok = False
                seen.add(deployment)
                continue
            seen.add(deployment)
            t0 = time.monotonic()
            try:
                reply = gw.complete_json(task, "Reply with JSON only.",
                                         'Return exactly {"ok": true, "model_check": "IME"}', max_tokens=50)
                good = isinstance(reply, dict) and reply.get("ok") is True
                print(f"  {'OK  ' if good else 'ODD '} {task:<10} deployment={deployment:<20} "
                      f"{(time.monotonic() - t0):.1f}s  reply={reply}")
                ok &= good
            except GatewayError as err:
                print(f"  FAIL {task:<10} deployment={deployment:<20} {err}\n       -> {_hint(err)}")
                ok = False
        print(f"  cost of this check: ${gw.cost_usd:.6f}")

    if settings.ocr == "azure_di" or di_client is not None:
        from .ingest.azure_di import analyze_pdf, make_client
        from .synthetic import build_case
        print(f"Document Intelligence: {settings.azure_di_endpoint or '-'}")
        try:
            client = di_client or make_client(settings.azure_di_endpoint, settings.azure_di_key)
            with tempfile.TemporaryDirectory() as tmp:
                pdf, _ = build_case(tmp, "di_check")
                pages, _ = analyze_pdf(str(pdf), client, page_range=(1, 1))
            words = len(pages[0].text.split()) if pages else 0
            print(f"  {'OK  ' if words > 20 else 'ODD '} read page 1 of a synthetic case: {words} words, "
                  f"confidence {pages[0].ocr_confidence if pages else None}")
            ok &= words > 20
        except Exception as err:  # noqa: BLE001
            print(f"  FAIL {err}\n       -> {_hint(err)}")
            ok = False
    return ok
