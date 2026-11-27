"""HTTP API for Phase 1: upload a PDF, get its text back page by page.

Run locally:   uvicorn ime_copilot.api:app --port 8000   (docs at /docs)

Security notes for this stage:
- Every request needs the X-API-Key header when IME_API_KEY is set. This is a
  development guard only; production uses Microsoft Entra ID sign-in in front
  of the API (App Service / Container Apps authentication) and role checks.
- Uploads from the web page (wait=false) are processed in a background thread with
  progress at /cases/{id}/status. In Azure this moves to a queue and a worker.
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

from . import __version__, jobs
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


@lru_cache
def _engine_for(db_url: str):
    return get_engine(db_url)


def _engine():
    return _engine_for(settings().db_url)


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
            "extractor": s.extractor, "provider": s.provider,
            "answer_model": s.models.get("answer") if s.provider != "none" else None,
            "escalation_model": s.models.get("answer_escalate") or None,
            "embed_model": s.models.get("embed") if s.provider != "none" else None,
            "api_key_required": bool(s.api_key)}


@app.post("/cases", dependencies=[Depends(require_key)])
async def upload_case(file: UploadFile = File(...), case_id: str | None = Form(default=None),
                      wait: bool = Form(default=True)) -> dict:
    """wait=true: process now and return the summary. wait=false (the web page): start in the
    background and return at once; follow progress at GET /cases/{case_id}/status."""
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

    if jobs.is_running(case_id):
        raise HTTPException(status_code=409, detail="This case is already being processed.")
    storage = get_storage(s)
    stored_at = storage.put(case_key(case_id, "original.pdf"), data, "application/pdf")
    if not wait:
        work = Path(s.local_storage_dir).parent / "uploads"
        work.mkdir(parents=True, exist_ok=True)
        path = work / f"{case_id}.pdf"
        path.write_bytes(data)
        jobs.start(path, case_id, s, storage=storage)
        return {"case_id": case_id, "status": "processing"}
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


@app.get("/cases/{case_id}/status", dependencies=[Depends(require_key)])
def status(case_id: str) -> dict:
    """Progress of a background upload: stage, done/total, seconds left in this stage."""
    try:
        validate_case_id(case_id)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    job = jobs.get_job(case_id)
    if job:
        return job
    case = get_case(_engine(), case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found.")
    return {"case_id": case_id, "stage": "ready", "label": jobs.STAGE_LABELS["ready"],
            "pages": case.get("page_count")}


@app.get("/cases/{case_id}/summary", dependencies=[Depends(require_key)])
def case_summary(case_id: str) -> dict:
    """The pinned summary: facts from the file (always) + the written summary (once it's done)."""
    from .case_summary import build_summary, load_overview
    _case_or_404(case_id)
    facts = build_summary(_engine(), case_id)
    overview = load_overview(settings().index_dir, case_id)
    job = jobs.get_job(case_id) or {}
    status = ("writing" if jobs.is_summarizing(case_id) else "done" if overview
              else "unavailable" if settings().provider == "none" else "not_started")
    return {"facts": facts, "overview": overview, "overview_status": status,
            "overview_progress": {"done": job.get("done", 0), "total": job.get("total", 0)}
            if status == "writing" else None}


@app.post("/cases/{case_id}/summary", dependencies=[Depends(require_key)])
def rewrite_summary(case_id: str) -> dict:
    """Write (or rewrite) the report-style summary in the background."""
    _case_or_404(case_id)
    if settings().provider == "none":
        raise HTTPException(status_code=400, detail="No answer model is set, so a written summary isn't available.")
    if jobs.is_running(case_id) or jobs.is_summarizing(case_id):
        raise HTTPException(status_code=409, detail="This file is busy; try again in a moment.")
    jobs.start_summary(case_id, settings())
    return {"case_id": case_id, "status": "writing"}


@app.patch("/cases/{case_id}", dependencies=[Depends(require_key)])
def rename_case(case_id: str, display_name: str = Body("", embed=True)) -> dict:
    """Change the label shown in the case list. The original PDF and its id are not changed."""
    from .db import set_display_name
    _case_or_404(case_id)
    name = " ".join(display_name.split())[:120]
    set_display_name(_engine(), case_id, name)
    return _case_or_404(case_id)


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
def ask(case_id: str, question: str = Body(..., embed=True),
        history: list[dict] = Body(default_factory=list, embed=True)) -> dict:
    """Answer a question about the case: answer text, related CPT codes, source pages.
    history: the last few turns [{"question", "answer"}], so follow-up questions work."""
    from .case_summary import load_overview
    from .rag.service import get_qa
    if jobs.is_running(case_id):
        raise HTTPException(status_code=409, detail="This file is still being processed; try again when it's ready.")
    _case_or_404(case_id)
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question is empty.")
    turns = [{"question": str(h.get("question", ""))[:1000], "answer": str(h.get("answer", ""))[:2000]}
             for h in history[-3:] if isinstance(h, dict)]
    overview = (load_overview(settings().index_dir, case_id) or {}).get("text") or None
    return get_qa(_engine(), case_id, settings()).ask(question.strip(), history=turns, overview=overview).model_dump()


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    """A simple page: upload a PDF, pick a case, ask questions."""
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
