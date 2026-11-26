"""Score a pipeline result against a ground-truth file (synthetic now, golden set later).

The same scorer runs on every model, prompt or rule change ("CI for AI"),
and its targets match the architecture doc's launch targets.
"""

from __future__ import annotations

import json
from pathlib import Path

from .extract.citations import normalize, quote_on_page
from .schemas import CaseResult

TARGETS = {
    "page_type_accuracy": 0.95,
    "service_precision": 0.97,
    "service_recall": 0.95,
    "citation_accuracy": 0.99,
    "questions_verbatim": 1.0,
    "questions_order": 1.0,
    "case_details_accuracy": 1.0,
}


def _pr(pred: set, truth: set) -> tuple[float, float]:
    hit = len(pred & truth)
    precision = hit / len(pred) if pred else (1.0 if not truth else 0.0)
    recall = hit / len(truth) if truth else 1.0
    return round(precision, 4), round(recall, 4)


def score(result: CaseResult, truth: dict) -> dict:
    pages_by_no = {p.page_no: p for p in result.pages}

    types = truth["page_types"]
    correct = sum(1 for no, t in types.items() if pages_by_no.get(int(no)) and pages_by_no[int(no)].page_type == t)
    misclassified = {no: {"expected": t, "got": pages_by_no[int(no)].page_type}
                     for no, t in types.items()
                     if pages_by_no.get(int(no)) and pages_by_no[int(no)].page_type != t}

    dup_pred = {(p.page_no, p.duplicate_of) for p in result.pages if p.duplicate_of}
    dup_truth = {(int(k), v) for k, v in truth["duplicates"].items()}

    def skey(s):
        return (s["date_of_service"], s["cpt"], s.get("modifier"), int(s["units"]), round(float(s["billed_amount"]), 2))

    svc_pred = {skey(s.model_dump()) for s in result.services}
    svc_truth = {skey(s) for s in truth["services"]}

    dx_pred = {d.code for d in result.diagnoses}
    dx_truth = set(truth["diagnoses"])

    rows = [s.citation for s in result.services] + [d.citation for d in result.diagnoses]
    cited = sum(1 for c in rows if quote_on_page(c, pages_by_no))

    q_truth = truth["questions"]
    q_pred = [(q.number, normalize(q.text)) for q in result.questions]
    verbatim = sum(1 for t in q_truth if (t["number"], normalize(t["text"])) in q_pred)
    cat_by_number = {q.number: q.category for q in result.questions}
    with_cat = [t for t in q_truth if "category" in t]
    cat_ok = sum(1 for t in with_cat if cat_by_number.get(t["number"]) == t["category"])

    want = truth.get("details", {})
    got = {name: (fv.value if fv else None) for name, fv in result.details}
    details_ok = {k: got.get(k) == v for k, v in want.items()}

    sp, sr = _pr(svc_pred, svc_truth)
    dp, dr = _pr(dx_pred, dx_truth)
    dupp, dupr = _pr(dup_pred, dup_truth)
    metrics = {
        "page_type_accuracy": round(correct / len(types), 4) if types else 1.0,
        "duplicate_precision": dupp, "duplicate_recall": dupr,
        "service_precision": sp, "service_recall": sr,
        "diagnosis_precision": dp, "diagnosis_recall": dr,
        "citation_accuracy": round(cited / len(rows), 4) if rows else 1.0,
        "questions_verbatim": round(verbatim / len(q_truth), 4) if q_truth else 1.0,
        "questions_order": 1.0 if [n for n, _ in q_pred] == [t["number"] for t in q_truth] else 0.0,
        "question_category_accuracy": round(cat_ok / len(with_cat), 4) if with_cat else 1.0,
        "case_details_accuracy": round(sum(details_ok.values()) / len(details_ok), 4) if details_ok else 1.0,
    }
    return {
        "metrics": metrics,
        "targets_met": {k: metrics[k] >= v for k, v in TARGETS.items()},
        "details": {
            "misclassified_pages": misclassified,
            "missed_services": sorted(map(str, svc_truth - svc_pred)),
            "extra_services": sorted(map(str, svc_pred - svc_truth)),
            "wrong_question_categories": {t["number"]: {"expected": t["category"],
                                                        "got": cat_by_number.get(t["number"])}
                                          for t in with_cat
                                          if cat_by_number.get(t["number"]) != t["category"]},
            "wrong_case_details": {k: {"expected": want[k], "got": got.get(k)}
                                   for k, ok in details_ok.items() if not ok},
            "missed_diagnoses": sorted(dx_truth - dx_pred),
            "extra_diagnoses": sorted(dx_pred - dx_truth),
            "pages_needing_review": [p.page_no for p in result.pages if p.needs_review],
        },
    }


def load_truth(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
