"""Background processing for the web page: read a big PDF, then build its search index,
reporting progress so the page can show a progress bar and time left.

Stages: queued -> reading -> ocr -> analyzing -> indexing -> summarizing -> ready  (or error)
Questions can be asked from 'summarizing' on; the written summary appears when it is done.
One process, in-memory job table: right for a single Mac demo. In Azure this becomes
a queue (Service Bus) and a worker, with the same stages.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from .config import Settings

STAGE_LABELS = {
    "queued": "Waiting to start",
    "reading": "Reading the PDF text",
    "ocr": "Reading scanned pages (OCR)",
    "analyzing": "Finding bills, diagnoses and duplicate pages",
    "indexing": "Building the search index",
    "summarizing": "Writing the case summary (questions already work)",
    "ready": "Ready for questions",
    "error": "Failed",
}

_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


def get_job(case_id: str) -> dict | None:
    with _LOCK:
        job = _JOBS.get(case_id)
        return dict(job) if job else None


def is_running(case_id: str) -> bool:
    """True while the file can't be searched yet (questions are allowed during 'summarizing')."""
    job = get_job(case_id)
    return bool(job) and job["stage"] not in ("ready", "error", "summarizing")


def is_summarizing(case_id: str) -> bool:
    job = get_job(case_id)
    return bool(job) and job["stage"] == "summarizing"


def _update(case_id: str, stage: str, done: int = 0, total: int = 0, **extra) -> None:
    now = time.time()
    with _LOCK:
        job = _JOBS.setdefault(case_id, {"case_id": case_id, "started": now})
        if job.get("stage") != stage:
            job["stage_started"], job["stage_start_done"] = now, done
        job.update(stage=stage, label=STAGE_LABELS.get(stage, stage), done=done, total=total,
                   updated=now, elapsed_s=round(now - job["started"]), **extra)
        # time left in this stage, from the speed so far in this stage
        rate_done = done - job.get("stage_start_done", 0)
        spent = now - job.get("stage_started", now)
        job["stage_eta_s"] = round(spent / rate_done * (total - done)) if (rate_done > 0 and total) else None


def start(pdf_path: Path, case_id: str, settings: Settings, storage=None,
          page_range: tuple[int, int] | None = None) -> None:
    _update(case_id, "queued")
    threading.Thread(target=_run, args=(pdf_path, case_id, settings, storage, page_range),
                     name=f"ime-job-{case_id}", daemon=True).start()


def _run(pdf_path: Path, case_id: str, settings: Settings, storage, page_range) -> None:
    from .db import get_engine
    from .pipeline import process_case, summary
    from .rag.service import forget, get_qa
    try:
        result = process_case(pdf_path, case_id, settings, storage=storage, page_range=page_range,
                              on_progress=lambda stage, d, t: _update(case_id, stage, d, t))
        info = summary(result)
        _update(case_id, "indexing", 0, 0, pages=info["pages"], service_lines=info["service_lines"])
        forget(case_id)
        engine = get_engine(settings.db_url)
        qa = get_qa(engine, case_id, settings, refresh=True,
                    on_progress=lambda d, t: _update(case_id, "indexing", d, t))
        _summarize(qa, case_id, settings)
        _update(case_id, "ready", 1, 1, notes=qa.notes)
    except Exception as err:  # noqa: BLE001 - shown on the page; the server keeps running
        _update(case_id, "error", error=f"{type(err).__name__}: {err}")


def _summarize(qa, case_id: str, settings: Settings) -> None:
    """Write the report-style summary, if an answer model is set (otherwise only the facts show)."""
    from .case_summary import build_summary, case_context, write_overview
    from .db import get_engine
    if qa.gateway is None or not qa.gateway.has_model("answer"):
        return
    try:
        context = case_context(build_summary(get_engine(settings.db_url), case_id))
        write_overview(qa, settings.index_dir, case_id, case_context=context,
                       on_progress=lambda d, t: _update(case_id, "summarizing", d, t))
    except Exception as err:  # noqa: BLE001 - questions still work without the summary
        _update(case_id, "summarizing", 0, 0, summary_error=f"{type(err).__name__}: {err}")


def start_summary(case_id: str, settings: Settings) -> None:
    """Rewrite the summary for an already processed file (the 'Rewrite summary' button)."""
    _update(case_id, "summarizing", 0, 1)

    def run():
        from .db import get_engine
        from .rag.service import get_qa
        try:
            qa = get_qa(get_engine(settings.db_url), case_id, settings)
            _summarize(qa, case_id, settings)
            _update(case_id, "ready", 1, 1)
        except Exception as err:  # noqa: BLE001
            _update(case_id, "error", error=f"{type(err).__name__}: {err}")

    threading.Thread(target=run, name=f"ime-summary-{case_id}", daemon=True).start()
