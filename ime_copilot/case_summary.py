"""The pinned case summary shown at the top of every file in the web page.

Two parts:
1. Facts (instant, built from the processed file, no AI model): case details, what the
   file contains, billing by CPT code, diagnoses, the IME questions, pages to review.
   Every item carries the page it came from.
2. Written summary (by the answer model): the model reads a wide sample of the file (the
   whole-file fact sheet plus key passages on the injury, treatment, imaging, diagnoses,
   work, prior history and the IME questions) and writes what it understands about the
   case, in its own structure. Each statement is marked with its source, which the
   system turns into page numbers. Saved next to the search index: written once per file.
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

# Broad searches that together give the model a wide view of the file.
COVERAGE_QUERIES = [
    "how the injury or accident happened, mechanism of injury, date of injury",
    "history of present illness, chief complaint, presenting symptoms",
    "current symptoms, pain level, physical examination findings",
    "diagnoses, assessment and plan",
    "MRI, X-ray, CT, EMG results and impression",
    "treatment: chiropractic, physical therapy, injections, surgery, medications, response to treatment",
    "work status, job duties, restrictions, time off work",
    "past medical history, prior injuries, prior accidents, pre-existing conditions",
    "IME referral questions to be addressed",
    "most recent visit, follow-up plan, discharge, maximum medical improvement",
]
MAX_PASSAGES = 24

_SUMMARY_SYSTEM = """You are an experienced medical-legal analyst helping a physician prepare an independent
medical examination (IME). Below is a sample of one claim file: a fact sheet computed from every
bill in the file, and numbered passages from the file's most relevant pages.

Write your own understanding of this case for the examining physician, in the structure you think
fits this case best. Use short "## " headings, short paragraphs and "- " bullet points. Typically
cover who the claimant is, what happened, the course of treatment over time, the current status,
and what the examiner should pay attention to (for example gaps in treatment, inconsistencies
between records, prior injuries, missing records, or open questions). Lead with the most
important points. Be specific: dates, providers, body parts, findings, treatments and amounts.

Rules:
- Use only what the fact sheet and passages say; if something important is unclear or missing, say so.
- After each statement, put the passage or fact number it came from in square brackets, e.g. [4] or [F2].
  Never write page numbers yourself.
- Your observations are allowed and useful, but phrase them as observations from the records
  (e.g. "The records show no treatment between ... and ..."), not as medical conclusions on
  causation, necessity or impairment: those are the physician's.

Reply with JSON only: {"summary": "<the summary, with [n] markers>"}"""


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


def gather_passages(qa, queries=COVERAGE_QUERIES, per_query: int = 5, limit: int = MAX_PASSAGES):
    """Take the best passages for each broad query in turn (round robin), without repeats."""
    results = [qa.index.search(q, k=per_query) for q in queries]
    hits, seen = [], set()
    for rank in range(per_query):
        for res in results:
            if rank < len(res) and res[rank].passage.pid not in seen and len(hits) < limit:
                seen.add(res[rank].passage.pid)
                hits.append(res[rank])
    return sorted(hits, key=lambda h: (h.passage.page_no, h.passage.pid))


def write_overview(qa, index_dir: str | Path, case_id: str, processed_at: str = "",
                   on_progress=None, case_context: str = "") -> dict:
    """Ask the model to write its own understanding of the case, then link it to pages and save it."""
    from .rag.qa import _resolve_passage_markers, _unsupported_values
    if on_progress:
        on_progress(0, 1)
    hits = gather_passages(qa)
    facts = "\n".join(f"[F{i}] {f['text']}" for i, f in enumerate(qa.facts, 1)) or "none"
    passages = "\n\n".join(f"[{i}] (p. {h.passage.page_no})\n{h.passage.text}" for i, h in enumerate(hits, 1))
    user = (f"{case_context}\n\nFACT SHEET (whole file):\n{facts}\n\nPASSAGES:\n{passages}").strip()

    settings = getattr(qa.gateway, "settings", None)
    out = {"version": 2, "case_id": case_id, "processed_at": processed_at,
           "model": settings.models.get("answer") if settings else None,
           "text": "", "sources": [], "warnings": [], "method": "model",
           "pages_read": sorted({h.passage.page_no for h in hits})}
    try:
        reply = qa.gateway.complete_json("answer", _SUMMARY_SYSTEM, user, max_tokens=3000)
        raw = str((reply or {}).get("summary") or (reply or {}).get("answer") or "").strip() \
            if isinstance(reply, dict) else ""
    except Exception as err:  # noqa: BLE001 - shown on the page with a retry button
        raw = ""
        out["warnings"].append(f"The model could not write the summary: {err}")
    if raw:
        text, sources, bad = _resolve_passage_markers(raw, hits, qa.pages_by_no, qa.facts)
        out["text"] = text
        seen, uniq = set(), []
        for x in sources:
            if (x.page, x.quote) not in seen:
                seen.add((x.page, x.quote))
                uniq.append(x.model_dump())
        out["sources"] = uniq
        if bad:
            out["warnings"].append(f"{bad} source marker(s) didn't match the passages and were removed.")
        if not sources:
            out["method"] = "unlinked"
            out["warnings"].append("The model didn't mark its sources, so this summary isn't linked to pages.")
        missing = _unsupported_values(text, " ".join(h.passage.text for h in hits) + " "
                                      + " ".join(f["text"] for f in qa.facts))
        if missing:
            out["warnings"].append("Check these in the file; they weren't found in the text the model read: "
                                   + ", ".join(missing[:8]))
    elif not out["warnings"]:
        out["warnings"].append("The model returned an empty summary.")
    if not out["text"]:
        out["method"] = "failed"
    if on_progress:
        on_progress(1, 1)
    path = overview_path(index_dir, case_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def case_context(facts: dict | None) -> str:
    """Case details and what the file contains, as plain text for the summary prompt."""
    if not facts:
        return ""
    lines = ["CASE DETAILS (read from the file):"]
    lines += [f"- {d['label']}: {d['value']}" for d in facts["details"] if d["field"] != "ssn_last4"]
    f = facts["file"]
    lines.append(f"- The file has {f['pages']} pages ({f['duplicates']} duplicates); page types: "
                 + ", ".join(f"{t['type'].replace('_', ' ')} {t['count']}" for t in f["page_types"]))
    if facts["questions"]:
        lines.append("IME QUESTIONS:")
        lines += [f"- {q['number']}. {q['text']}" for q in facts["questions"][:15]]
    return "\n".join(lines)
