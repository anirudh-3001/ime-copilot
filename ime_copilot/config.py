"""Settings, read from environment variables (and an optional .env file).

Target platform: Microsoft Azure (Modus runs on Azure).
  Storage  - Azure Blob Storage (original PDFs, OCR layout, ledger exports)
  OCR      - Azure AI Document Intelligence, prebuilt-layout model
  Models   - Azure OpenAI / Foundry deployments through the v1 API
  Database - Azure Database for PostgreSQL (SQLite for local development)
Authentication defaults to Microsoft Entra ID (managed identity / az login);
keys are supported for local development only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader so we don't need an extra dependency."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.split(" #", 1)[0].split("\t#", 1)[0]       # allow comments after a value
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# List prices in USD per 1M tokens (input, output). Used only for cost metering.
# Azure deployment names are free-form: name deployments after the model
# (e.g. "gpt-6.1-sol") or set IME_PRICE_<TASK>="input,output" explicitly.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "gpt-6-astra": (10.00, 50.00),
    "gpt-6.1-sol": (2.00, 10.00),
    "gpt-6.1-terra": (1.00, 5.00),
    "gpt-6-luna": (0.10, 0.50),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-fable-5-2": (8.00, 40.00),
    "text-embedding-3-large": (0.13, 0.0),
    "text-embedding-3-small": (0.02, 0.0),
}

TASKS = ("classify", "extract", "questions", "answer", "answer_escalate", "embed")


def price_for(model_id: str) -> tuple[float, float]:
    for key, price in MODEL_PRICES.items():
        if key in model_id:
            return price
    return (0.0, 0.0)


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass
class Settings:
    db_url: str = "sqlite:///data/ime_ledger.db"
    ocr: str = "auto"                  # auto | pdftext | tesseract | azure_di | textract
    extractor: str = "rules"           # rules | llm
    provider: str = "none"             # none | local | azure_openai | foundry_claude | bedrock
    local_base_url: str = "http://localhost:11434/v1"   # Ollama (or LM Studio, vLLM) on this machine
    storage: str = "local"             # local | azure_blob
    local_storage_dir: str = "data/blobs"
    max_upload_mb: int = 500
    api_key: str = ""                  # shared secret for the dev API (Entra ID in production)

    # Local OCR (Tesseract) tuning for large scanned files
    ocr_dpi: int = 300                 # 200 is ~2x faster and usually enough for typed records
    ocr_workers: int = 0               # 0 = one per CPU
    ocr_cache_dir: str = ""            # per-page OCR cache; set to resume interrupted runs

    # Azure
    azure_storage_account_url: str = ""
    azure_storage_connection_string: str = ""
    azure_storage_container: str = "ime-cases"
    azure_di_endpoint: str = ""
    azure_di_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    foundry_claude_resource: str = ""        # Foundry resource name for Claude deployments
    foundry_claude_api_key: str = ""

    # AWS (kept as an alternative)
    aws_region: str = "us-east-1"

    models: dict[str, str] = field(default_factory=dict)        # task -> deployment / model id
    task_prices: dict[str, tuple[float, float]] = field(default_factory=dict)

    @classmethod
    def from_env(cls, env_file: str | Path = ".env") -> "Settings":
        _load_dotenv(Path(env_file))
        provider = _env("IME_PROVIDER", "none")
        default_models = {
            "local": {"classify": "gemma4:26b-a4b", "extract": "gemma4:26b-a4b", "questions": "gemma4:26b-a4b",
                      "answer": "gemma4:26b-a4b", "embed": "qwen3-embedding:0.6b"},
            "azure_openai": {"classify": "gpt-6-luna", "extract": "gpt-6-luna", "questions": "gpt-6.1-sol",
                             "answer": "gpt-6.1-sol", "embed": "text-embedding-3-large"},
            "foundry_claude": {"classify": "claude-haiku-5-5", "extract": "claude-haiku-5-5",
                               "questions": "claude-sonnet-5-5", "answer": "claude-sonnet-5-5"},
            "bedrock": {"classify": "us.anthropic.claude-haiku-5-5", "extract": "us.anthropic.claude-haiku-5-5",
                        "questions": "us.anthropic.claude-sonnet-5-5"},
        }.get(provider, {})
        models = {t: _env(f"IME_MODEL_{t.upper()}", default_models.get(t, "")) for t in TASKS}
        prices = {}
        for t in TASKS:
            raw = _env(f"IME_PRICE_{t.upper()}")
            if raw:
                pin, pout = (float(x) for x in raw.split(","))
                prices[t] = (pin, pout)
        return cls(
            db_url=_env("IME_DB_URL", cls.db_url),
            ocr=_env("IME_OCR", cls.ocr),
            extractor=_env("IME_EXTRACTOR", cls.extractor),
            provider=provider,
            local_base_url=_env("IME_LOCAL_BASE_URL", cls.local_base_url),
            storage=_env("IME_STORAGE", cls.storage),
            local_storage_dir=_env("IME_LOCAL_STORAGE_DIR", cls.local_storage_dir),
            max_upload_mb=int(_env("IME_MAX_UPLOAD_MB", str(cls.max_upload_mb))),
            api_key=_env("IME_API_KEY"),
            ocr_dpi=int(_env("IME_OCR_DPI", str(cls.ocr_dpi))),
            ocr_workers=int(_env("IME_OCR_WORKERS", "0")),
            ocr_cache_dir=_env("IME_OCR_CACHE"),
            azure_storage_account_url=_env("AZURE_STORAGE_ACCOUNT_URL"),
            azure_storage_connection_string=_env("AZURE_STORAGE_CONNECTION_STRING"),
            azure_storage_container=_env("AZURE_STORAGE_CONTAINER", cls.azure_storage_container),
            azure_di_endpoint=_env("AZURE_DI_ENDPOINT"),
            azure_di_key=_env("AZURE_DI_KEY"),
            azure_openai_endpoint=_env("AZURE_OPENAI_ENDPOINT"),
            azure_openai_api_key=_env("AZURE_OPENAI_API_KEY"),
            foundry_claude_resource=_env("ANTHROPIC_FOUNDRY_RESOURCE"),
            foundry_claude_api_key=_env("ANTHROPIC_FOUNDRY_API_KEY"),
            aws_region=_env("AWS_REGION", cls.aws_region),
            models=models,
            task_prices=prices,
        )
