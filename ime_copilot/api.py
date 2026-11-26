"""HTTP API for Phase 1: upload a PDF, get its text back page by page.

Run locally:   uvicorn ime_copilot.api:app --port 8000   (docs at /docs)

Security notes for this stage:
- Every request needs the X-API-Key header when IME_API_KEY is set. This is a
  development guard only; production uses Microsoft Entra ID sign-in in front
  of the API (App Service / Container Apps authentication) and role checks.
- Processing runs inside the request for now. For large files it moves to a
  queue (Azure Service Bus or Storage Queue) and a background worker.
- Only synthetic or de-identified files until the HIPAA go-live gate.
"""

from __future__ import annotations

import secrets
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path

from fastapi import Body, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from . import __version__
from .config import Settings
from .db import get_case, get_engine, get_ledger, get_pages, list_cases
from .pipeline import process_case, summary
from .storage import case_key, get_storage, validate_case_id

app = FastAPI(title="IME Copilot", version=__version__,
              description="Stage 1: PDF ingestion, OCR, page classification and treatment ledger.")


@lru_cache
def settings() -> Settings:
    return Settings.from_env()


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = settings().api_key
    if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected)):
        raise HTTPException(status_code=401, detail="Missing or wrong X-API-Key.")


def _engine():
    return get_engine(settings().db_url)


def _case_or_404(case_id: str) -> dict:
    try:
        validate_case_id(case_id)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    case = get_case(_engine(), case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found.")
    return case


@app.get("/health")
def health() -> dict:
    s = settings()
    return {"status": "ok", "version": __version__, "ocr": s.ocr, "storage": s.storage,
            "extractor": s.extractor, "provider": s.provider}


@app.post("/cases", dependencies=[Depends(require_key)])
async def upload_case(file: UploadFile = File(...), case_id: str | None = Form(default=None)) -> dict:
    s = settings()
    case_id = case_id or uuid.uuid4().hex[:12]
    try:
        validate_case_id(case_id)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))

    data = await file.read(s.max_upload_mb * 1024 * 1024 + 1)
    if len(data) > s.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File larger than {s.max_upload_mb} MB.")
    if not data.startswith(b"%PDF"):
        raise HTTPException(status_code=415, detail="Only PDF files are accepted.")

    storage = get_storage(s)
    stored_at = storage.put(case_key(case_id, "original.pdf"), data, "application/pdf")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "original.pdf"
        path.write_bytes(data)
        result = process_case(path, case_id, s, storage=storage)
    out = summary(result)
    out["original_stored_at"] = stored_at if s.storage == "azure_blob" else case_key(case_id, "original.pdf")
    return out


@app.get("/cases", dependencies=[Depends(require_key)])
def cases() -> list[dict]:
    return list_cases(_engine())


@app.get("/cases/{case_id}", dependencies=[Depends(require_key)])
def case(case_id: str) -> dict:
    return _case_or_404(case_id)


@app.get("/cases/{case_id}/pages", dependencies=[Depends(require_key)])
def pages(case_id: str) -> list[dict]:
    _case_or_404(case_id)
    return get_pages(_engine(), case_id)


@app.get("/cases/{case_id}/pages/{page_no}", dependencies=[Depends(require_key)])
def page(case_id: str, page_no: int) -> dict:
    _case_or_404(case_id)
    for p in get_pages(_engine(), case_id, with_text=True):
        if p["page_no"] == page_no:
            return p
    raise HTTPException(status_code=404, detail="Page not found.")


@app.get("/cases/{case_id}/ledger", dependencies=[Depends(require_key)])
def ledger(case_id: str) -> dict:
    _case_or_404(case_id)
    return get_ledger(_engine(), case_id)


@app.post("/cases/{case_id}/ask", dependencies=[Depends(require_key)])
def ask(case_id: str, question: str = Body(..., embed=True)) -> dict:
    """Answer a question about the case: answer text, related CPT codes, source pages."""
    from .rag.service import get_qa
    _case_or_404(case_id)
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question is empty.")
    return get_qa(_engine(), case_id, settings()).ask(question.strip()).model_dump()


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    """A simple page: upload a PDF, pick a case, ask questions."""
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
