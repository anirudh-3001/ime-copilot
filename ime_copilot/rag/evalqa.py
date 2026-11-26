"""Question-answering test set on the synthetic case: are the right pages and CPT codes returned?

Run:  python -m ime_copilot eval-qa
It uses whatever model is configured in .env (IME_PROVIDER=local -> Gemma on this Mac),
or extractive search only when no model is set.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from ..config import Settings
from ..gateway import Gateway
from ..pipeline import process_case
from ..synthetic import build_case
from .qa import CaseQA

# Synthetic case layout: 1 referral letter, 2 office note, 3 MRI, 4 office note, 5 faxed copy of 2,
# 6-7 injection procedure notes, 8-9 PT notes (9 scanned), 10-11 PT bills, 12 injection/office/MRI bill.
QUESTIONS = [
    {"q": "What physical therapy was billed?",
     "pages": {8, 9, 10, 11}, "cpt": {"97110", "97140", "97010"}},
    {"q": "How many lumbar epidural steroid injections were done and on which dates?",
     "pages": {6, 7, 12}, "cpt": {"64483"}},
    {"q": "What did the MRI of the lumbar spine show?",
     "pages": {3}, "cpt": {"72148"}},
    {"q": "What office visits were billed?",
     "pages": {2, 4, 12}, "cpt": {"99204", "99214"}},
    {"q": "What are the patient's diagnoses?",
     "pages": {2, 4}, "cpt": set()},
]


def evaluate(qa: CaseQA) -> list[dict]:
    rows = []
    for item in QUESTIONS:
        t0 = time.monotonic()
        r = qa.ask(item["q"])
        got_codes = {c.code for c in r.cpt_codes}
        shown_pages = {s.page for s in r.sources} | set(r.searched_pages)
        want = item["cpt"]
        rows.append({
            "question": item["q"],
            "page_found": bool(item["pages"] & shown_pages),
            "cited_right_page": bool(item["pages"] & {s.page for s in r.sources}),
            "cpt_recall": round(len(want & got_codes) / len(want), 2) if want else 1.0,
            "cpt_extra": sorted(got_codes - want),
            "method": r.method,
            "seconds": round(time.monotonic() - t0, 1),
            "answer": r.answer,
        })
    return rows


def run_eval_qa(settings: Settings, provider=None) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        pdf, _ = build_case(tmp, "qa_eval")
        s = Settings(**{**settings.__dict__, "db_url": f"sqlite:///{Path(tmp) / 'qa.db'}",
                        "ocr": "auto", "extractor": "rules", "storage": "local",
                        "local_storage_dir": str(Path(tmp) / "blobs"), "ocr_cache_dir": ""})
        result = process_case(pdf, "qa_eval", s, progress=False)
        gateway = Gateway(s, provider=provider) if (provider is not None or s.provider != "none") else None
        qa = CaseQA(result.pages, result.services, gateway)
        rows = evaluate(qa)

    print(f"\nModel: {settings.models.get('answer') if gateway else 'none (extractive)'}\n")
    for r in rows:
        print(f"- {r['question']}\n    right page found: {r['page_found']}  | cited: {r['cited_right_page']}"
              f"  | CPT recall: {r['cpt_recall']}  | extra CPTs: {r['cpt_extra'] or '-'}"
              f"  | {r['method']}, {r['seconds']}s")
    found = sum(r["page_found"] for r in rows) / len(rows)
    recall = sum(r["cpt_recall"] for r in rows) / len(rows)
    print(f"\nRight page found: {found:.0%}   CPT recall: {recall:.0%}")
    return 0 if found == 1 and recall == 1 else 1
