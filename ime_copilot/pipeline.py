"""Stage 1 pipeline: PDF -> pages -> dedupe -> classify -> ledger rows + case questions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import Settings
from .storage import Storage, case_key, validate_case_id
from .db import get_engine, save_case
from .extract.citations import quote_on_page
from .extract.demographics import extract_case_details
from .extract.diagnoses import CLINICAL_TYPES, extract_diagnoses
from .extract.questions import parse_questions
from .extract.services import extract_services
from .gateway import Gateway, Provider
from .ingest.classify import classify_pages
from .ingest.dedupe import mark_duplicates
from .ingest.pdf import read_pages
from .schemas import CaseResult


def process_case(pdf_path: str | Path, case_id: str, settings: Settings,
                 provider: Provider | None = None, save: bool = True,
                 storage: Storage | None = None, di_client: Any = None,
                 page_range: tuple[int, int] | None = None, progress: bool = True,
                 on_progress: Any = None) -> CaseResult:
    validate_case_id(case_id)
    engine = get_engine(settings.db_url) if save else None
    gateway = Gateway(settings, provider=provider, engine=engine, case_id=case_id)
    mode = settings.extractor

    if storage is not None and not storage.exists(case_key(case_id, "original.pdf")):
        storage.put(case_key(case_id, "original.pdf"), Path(pdf_path).read_bytes(), "application/pdf")

    if settings.ocr == "azure_di" and di_client is None:
        from .ingest.azure_di import make_client
        di_client = make_client(settings.azure_di_endpoint, settings.azure_di_key)
    pages, layout = read_pages(
        pdf_path, settings.ocr, settings.aws_region, di_client=di_client, page_range=page_range,
        workers=settings.ocr_workers or None, cache_dir=settings.ocr_cache_dir or None,
        dpi=settings.ocr_dpi, progress=progress, on_progress=on_progress)
    if on_progress:
        on_progress("analyzing", 0, 1)
    if storage is not None and layout is not None:
        storage.put(case_key(case_id, "layout.json"), json.dumps(layout).encode(), "application/json")
    mark_duplicates(pages)
    classify_pages(pages, mode, gateway)

    pages_by_no = {p.page_no: p for p in pages}
    unique = [p for p in pages if p.duplicate_of is None]

    details = extract_case_details(unique)

    services, dropped_s = extract_services(
        [p for p in unique if p.page_type == "bill"], pages_by_no, mode, gateway)

    diagnoses = [d for d in extract_diagnoses([p for p in unique if p.page_type in CLINICAL_TYPES])
                 if quote_on_page(d.citation, pages_by_no)]

    questions, dropped_q = parse_questions(
        [p for p in unique if p.page_type == "referral_letter"], mode, gateway)

    result = CaseResult(
        case_id=case_id, source_file=str(pdf_path), pages=pages, details=details, services=services,
        diagnoses=diagnoses, questions=questions, dropped_uncited=dropped_s + dropped_q,
        model_cost_usd=round(gateway.cost_usd, 6),
    )
    if engine is not None:
        save_case(engine, result)
    if storage is not None:
        storage.put(case_key(case_id, "ledger.json"),
                    result.model_dump_json(exclude={"pages": {"__all__": {"text"}}}).encode(), "application/json")
    return result


def write_outputs(result: CaseResult, out_dir: str | Path) -> dict[str, Path]:
    """Ledger JSON (everything) and a services CSV a physician can open in Excel."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{result.case_id}_ledger.json"
    json_path.write_text(result.model_dump_json(indent=2, exclude={"pages": {"__all__": {"text"}}}),
                         encoding="utf-8")

    csv_path = out / f"{result.case_id}_services.csv"
    lines = ["date_of_service,cpt,modifier,units,billed_amount,provider,page,quote"]
    for s in result.services:
        quote = s.citation.quote.replace('"', '""')
        provider = (s.provider or "").replace('"', '""')
        lines.append(f'{s.date_of_service},{s.cpt},{s.modifier or ""},{s.units},'
                     f'{s.billed_amount if s.billed_amount is not None else ""},"{provider}",'
                     f'{s.citation.page_no},"{quote}"')
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path}


def summary(result: CaseResult) -> dict:
    by_type: dict[str, int] = {}
    for p in result.pages:
        by_type[p.page_type] = by_type.get(p.page_type, 0) + 1
    by_cpt: dict[str, dict] = {}
    for s in result.services:
        entry = by_cpt.setdefault(s.cpt, {"lines": 0, "units": 0, "billed": 0.0})
        entry["lines"] += 1
        entry["units"] += s.units
        entry["billed"] = round(entry["billed"] + (s.billed_amount or 0), 2)
    details = {
        name: f"{fv.value}  (p.{fv.citation.page_no}, on {fv.pages_agreeing} page(s))"
        for name, fv in result.details if fv is not None
    }
    return {
        "case_id": result.case_id,
        "details": details,
        "pages": len(result.pages),
        "page_types": by_type,
        "duplicates": [(p.page_no, p.duplicate_of) for p in result.pages if p.duplicate_of],
        "needs_review": [p.page_no for p in result.pages if p.needs_review],
        "tables_found": sum(len(p.tables) for p in result.pages),
        "service_lines": len(result.services),
        "billed_total": round(sum(s.billed_amount or 0 for s in result.services), 2),
        "by_cpt": by_cpt,
        "diagnoses": [f"{d.code} (p.{d.citation.page_no})" for d in result.diagnoses],
        "questions": [f"{q.number} [{q.category}]" for q in result.questions],
        "dropped_uncited": result.dropped_uncited,
        "model_cost_usd": result.model_cost_usd,
    }


def print_summary(result: CaseResult) -> None:
    print(json.dumps(summary(result), indent=2))
