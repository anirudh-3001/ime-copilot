"""The pinned case summary shown at the top of every file in the web page.

Two parts:
1. Facts (instant, built from the processed file, no AI model): case details, what the
   file contains, billing by CPT code, diagnoses, the IME questions, pages to review.
   Every item carries the page it came from.
2. Overview (on request, written by the answer model): a few standard questions answered
   with the same search + verification as any question, so every sentence is tied to
   pages. Saved next to the search index, so it is written once per file.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .db import get_case, get_ledger, get_pages
from .rag.cpt import category

DETAIL_LABELS = [
    ("patient_name", "Patient"), ("date_of_birth", "Date of birth"), ("claim_number", "Claim number"),
    ("date_of_injury", "Date of injury"), ("employer", "Employer"), ("insurer", "Insurer"),
    ("ssn_last4", "SSN (last 4)"),
]

# The written summary follows the sections of an IME report. Each section is one question,
# answered with the same search + verification as any question, so every sentence cites pages.
OVERVIEW_QUESTIONS = [
    ("hpi", "History of present illness",
     "How did the injury or accident happen, on what date, what care was received right after it, "
     "and how has the patient responded to treatment since?"),
    ("symptoms", "Current symptoms",
     "What are the patient's current symptoms, pain locations and pain levels? Any numbness, tingling "
     "or radiating symptoms?"),
    ("treatment", "Treatment to date",
     "What treatment has the patient received, such as chiropractic care, physical therapy, injections, "
     "surgery or other procedures, from which providers, and did it help?"),
    ("studies", "Diagnostic studies",
     "What imaging or diagnostic tests were done, such as MRI, X-ray, CT or EMG, and what did they show?"),
    ("diagnoses", "Diagnoses", "What diagnoses have been given for the patient's injuries?"),
    ("past", "Past medical and surgical history",
     "What past medical history, prior injuries, prior surgeries or pre-existing conditions are recorded?"),
    ("function", "Function and social history",
     "What activities can the patient no longer do since the injury, what help do they need at home, "
     "and what are their habits and daily activities?"),
    ("work", "Work history and status",
     "What is the patient's job, how has the injury affected their work, and what work restrictions "
     "or work status have been given?"),
]


def build_summary(engine, case_id: str) -> dict | None:
    case = get_case(engine, case_id)
    if case is None:
        return None
    pages = get_pages(engine, case_id)
    ledger = get_ledger(engine, case_id)

    details = []
    for key, label in DETAIL_LABELS:
        fv = (case.get("details") or {}).get(key)
        if fv and fv.get("value"):
            details.append({"field": key, "label": label, "value": fv["value"],
                            "page": (fv.get("citation") or {}).get("page_no"),
                            "pages_agreeing": fv.get("pages_agreeing", 1),
                            "other_values": fv.get("other_values") or []})

    types = Counter((p.get("page_type") or "other") for p in pages if not p.get("duplicate_of"))
    scanned = sum(1 for p in pages if p.get("text_source") not in (None, "pdftext"))

    by_code: dict[str, dict] = {}
    for s in ledger["services"]:
        e = by_code.setdefault(s["cpt"], {"code": s["cpt"], "category": category(s["cpt"])[0], "lines": 0,
                                          "units": 0, "total": 0.0, "dates": set(), "pages": set()})
        e["lines"] += 1
        e["units"] += s.get("units") or 1
        e["total"] += s.get("billed_amount") or 0.0
        if s.get("date_of_service"):
            e["dates"].add(s["date_of_service"])
        e["pages"].add(s["page_no"])
    codes = []
    for e in sorted(by_code.values(), key=lambda x: (-x["total"], x["code"])):
        dates = sorted(e.pop("dates"))
        codes.append({**e, "total": round(e["total"], 2), "first_date": dates[0] if dates else None,
                      "last_date": dates[-1] if dates else None, "pages": sorted(e["pages"])})
    by_category: dict[str, dict] = {}
    for c in codes:
        g = by_category.setdefault(c["category"], {"category": c["category"], "lines": 0, "total": 0.0})
        g["lines"] += c["lines"]
        g["total"] = round(g["total"] + c["total"], 2)
    all_dates = sorted(d for s in ledger["services"] if (d := s.get("date_of_service")))

    providers = Counter()
    for s in ledger["services"]:
        if s.get("provider"):
            providers[s["provider"].strip()] += s.get("billed_amount") or 0.0

    diagnoses, seen = [], set()
    for d in ledger["diagnoses"]:
        if d["code"] in seen:
            continue
        seen.add(d["code"])
        diagnoses.append({"code": d["code"], "description": d.get("description"),
                          "first_seen_date": d.get("first_seen_date"), "page": d["page_no"]})

    return {
        "case_id": case_id,
        "processed_at": str(case.get("processed_at") or ""),
        "details": details,
        "file": {
            "pages": len(pages),
            "unique_pages": sum(1 for p in pages if not p.get("duplicate_of")),
            "duplicates": sum(1 for p in pages if p.get("duplicate_of")),
            "scanned": scanned,
            "needs_review": [p["page_no"] for p in pages if p.get("needs_review")],
            "page_types": [{"type": t, "count": n} for t, n in types.most_common()],
        },
        "billing": {
            "lines": len(ledger["services"]),
            "total": round(sum(c["total"] for c in codes), 2),
            "first_date": all_dates[0] if all_dates else None,
            "last_date": all_dates[-1] if all_dates else None,
            "by_category": sorted(by_category.values(), key=lambda g: -g["total"]),
            "by_code": codes,
            "providers": [{"name": n, "total": round(t, 2)} for n, t in providers.most_common(8)],
        },
        "diagnoses": diagnoses,
        "questions": [{"number": q["number"], "text": q["text"], "category": q.get("category"),
                       "page": q["page_no"]} for q in ledger["questions"]],
    }


def overview_path(index_dir: str | Path, case_id: str) -> Path:
    return Path(index_dir) / case_id / "overview.json"


def load_overview(index_dir: str | Path, case_id: str) -> dict | None:
    path = overview_path(index_dir, case_id)
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, ValueError):
        return None


def write_overview(qa, index_dir: str | Path, case_id: str, processed_at: str = "",
                   on_progress=None) -> dict:
    """Write the report-style summary (one verified answer per section) and save it."""
    sections = []
    for i, (key, title, question) in enumerate(OVERVIEW_QUESTIONS):
        if on_progress:
            on_progress(i, len(OVERVIEW_QUESTIONS))
        r = qa.ask(question)
        sections.append({"key": key, "title": title, "question": question, "answer": r.answer,
                         "method": r.method, "sources": [x.model_dump() for x in r.sources],
                         "cpt_codes": [c.code for c in r.cpt_codes]})
    if on_progress:
        on_progress(len(OVERVIEW_QUESTIONS), len(OVERVIEW_QUESTIONS))
    settings = getattr(qa.gateway, "settings", None)
    out = {"case_id": case_id, "processed_at": processed_at, "sections": sections,
           "model": settings.models.get("answer") if settings else None}
    path = overview_path(index_dir, case_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
