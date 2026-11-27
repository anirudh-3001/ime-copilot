"""Load a processed case from the ledger database and keep one Q&A engine per case in memory."""

from __future__ import annotations

import json
import threading

from ..db import get_ledger, get_pages
from ..gateway import Gateway
from ..schemas import Citation, Page, PageTable, ServiceRow
from .qa import CaseQA
from .store import DiskVectorCache, index_path

_CACHE: dict[str, CaseQA] = {}


def load_case(engine, case_id: str) -> tuple[list[Page], list[ServiceRow]]:
    pages = []
    for r in get_pages(engine, case_id, with_text=True):
        pages.append(Page(
            page_no=r["page_no"], text=r.get("text") or "", text_source=r.get("text_source") or "pdftext",
            ocr_confidence=r.get("ocr_confidence"), page_type=r.get("page_type") or "other",
            type_confidence=r.get("type_confidence") or 0.0, duplicate_of=r.get("duplicate_of"),
            needs_review=bool(r.get("needs_review")),
            tables=[PageTable(**t) for t in (r.get("tables") or [])],
        ))
    services = [
        ServiceRow(date_of_service=s["date_of_service"], cpt=s["cpt"], modifier=s.get("modifier"),
                   units=s.get("units") or 1, billed_amount=s.get("billed_amount"),
                   dx_pointer=s.get("dx_pointer"), provider=s.get("provider"),
                   citation=Citation(page_no=s["page_no"], quote=s["quote"]),
                   confidence=s.get("confidence") or 1.0, method=s.get("method") or "rules")
        for s in get_ledger(engine, case_id)["services"]
    ]
    return pages, services


_LOCK = threading.Lock()


def get_qa(engine, case_id: str, settings, refresh: bool = False, on_progress=None) -> CaseQA:
    """The Q&A engine for a case. Search vectors are loaded from disk (data/index), and only
    passages not seen before are embedded, so a big file is indexed once."""
    with _LOCK:
        if refresh or case_id not in _CACHE:
            pages, services = load_case(engine, case_id)
            if not pages:
                raise KeyError(f"Case '{case_id}' has not been processed yet.")
            gateway = Gateway(settings, engine=engine, case_id=case_id) if settings.provider != "none" else None
            cache = None
            embed_model = settings.models.get("embed")
            if gateway is not None and embed_model:
                cache = DiskVectorCache(index_path(settings.index_dir, case_id, embed_model))
            from ..db import get_case
            details = (get_case(engine, case_id) or {}).get("details") or {}
            _CACHE[case_id] = CaseQA(pages, services, gateway, vector_cache=cache, on_progress=on_progress,
                                     details=details)
        return _CACHE[case_id]


def forget(case_id: str) -> None:
    with _LOCK:
        _CACHE.pop(case_id, None)


def result_text(r) -> str:
    """Plain-text rendering of an AskResult for the terminal."""
    lines = [f"\nQ: {r.question}", "", r.answer, ""]
    if r.cpt_codes:
        lines.append("CPT / HCPCS codes related to this question:")
        for c in r.cpt_codes:
            billed = (f"  billed {c.billed_lines} line(s), {c.units} unit(s), ${c.billed_total:,.2f}"
                      f" ({c.first_date} to {c.last_date})") if c.billed_lines else ""
            lines.append(f"  {c.code}  {c.category}  pages {', '.join(map(str, c.pages[:12]))}"
                         f"{' ...' if len(c.pages) > 12 else ''}{billed}")
        lines.append("")
    if r.sources:
        lines.append("Sources:")
        lines += [f"  p. {s.page}: “{s.quote[:160]}”" for s in r.sources]
    lines.append(f"\n[{r.method}{' via ' + r.model if r.model else ''} | pages searched: "
                 f"{', '.join(map(str, r.searched_pages))}]")
    lines += [f"  note: {w}" for w in r.warnings]
    return "\n".join(lines)


def result_json(r) -> str:
    return json.dumps(r.model_dump(), indent=2)
