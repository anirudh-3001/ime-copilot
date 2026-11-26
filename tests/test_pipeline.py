import json
import re

import pytest
from sqlalchemy import select

from ime_copilot.config import Settings
from ime_copilot.db import get_engine, model_calls, services
from ime_copilot.evaluate import load_truth, score
from ime_copilot.extract.questions import parse_questions_rules
from ime_copilot.extract.services import extract_services_rules
from ime_copilot.gateway import FakeProvider, parse_json
from ime_copilot.ingest.classify import classify_rules
from ime_copilot.ingest.dedupe import mark_duplicates
from ime_copilot.pipeline import process_case
from ime_copilot.schemas import Page
from ime_copilot.synthetic import build_case


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic")
    return build_case(out)


def _settings(tmp_path, **kw):
    s = Settings(db_url=f"sqlite:///{tmp_path / 'ledger.db'}", ocr="auto", extractor="rules")
    s.models = {"classify": "fake-classify", "extract": "fake-extract", "questions": "fake-questions"}
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_rules_pipeline_meets_targets(case, tmp_path):
    pdf, truth = case
    result = process_case(pdf, "t1", _settings(tmp_path))
    report = score(result, load_truth(truth))
    m = report["metrics"]
    assert m["service_precision"] == 1.0 and m["service_recall"] == 1.0, report["details"]
    assert m["questions_verbatim"] == 1.0 and m["questions_order"] == 1.0
    assert m["question_category_accuracy"] == 1.0, report["details"]["wrong_question_categories"]
    assert m["citation_accuracy"] == 1.0
    assert m["duplicate_recall"] == 1.0
    # Text-layer pages must classify perfectly; the scanned page depends on local OCR.
    text_pages = {k: v for k, v in report["details"]["misclassified_pages"].items() if k != "9"}
    assert not text_pages


def test_ledger_rows_saved_with_page_and_quote(case, tmp_path):
    pdf, _ = case
    settings = _settings(tmp_path)
    process_case(pdf, "t2", settings)
    with get_engine(settings.db_url).connect() as conn:
        rows = conn.execute(select(services).where(services.c.case_id == "t2")).fetchall()
    assert len(rows) == 39
    assert all(r.page_no > 0 and r.quote for r in rows)


def _fake_responder(hallucinate: bool):
    """Simulates the model: answers like a good model, optionally adding made-up rows."""
    def respond(model_id: str, user: str) -> str:
        if model_id == "fake-classify":
            ptype, conf = classify_rules(Page(page_no=0, text=user))
            return json.dumps({"page_type": ptype, "confidence": conf})
        if model_id == "fake-extract":
            page_no = int(re.match(r"PAGE (\d+):", user).group(1))
            page = Page(page_no=page_no, text=user.split("\n", 1)[1])
            rows = [{"date_of_service": f"{r.date_of_service[5:7]}/{r.date_of_service[8:]}/{r.date_of_service[:4]}",
                     "cpt": r.cpt, "modifier": r.modifier, "units": r.units,
                     "billed_amount": r.billed_amount, "quote": r.citation.quote}
                    for r in extract_services_rules(page)]
            if hallucinate and rows:
                # 1) a quote that is not on the page, 2) a real quote with a wrong amount
                rows.append({**rows[0], "cpt": "64490", "quote": "09/09/2025 64490 facet injection 999.00 1"})
                rows.append({**rows[0], "billed_amount": 99.0})
            return json.dumps(rows)
        if model_id == "fake-questions":
            pages = [Page(page_no=int(n), text=t) for n, t in re.findall(r"PAGE (\d+):\n(.*?)(?=\n\nPAGE |\Z)", user, re.S)]
            return json.dumps([{"number": q.number, "text": q.text, "category": q.category,
                                "instructions": q.instructions, "page_no": q.citation.page_no}
                               for q in parse_questions_rules(pages)])
        raise AssertionError(model_id)
    return respond


def test_llm_path_drops_uncited_rows_and_meters_cost(case, tmp_path):
    pdf, truth = case
    settings = _settings(tmp_path, extractor="llm", provider="fake")
    result = process_case(pdf, "t3", settings, provider=FakeProvider(_fake_responder(hallucinate=True)))
    report = score(result, load_truth(truth))
    assert report["metrics"]["service_precision"] == 1.0, report["details"]["extra_services"]
    assert result.dropped_uncited >= 2
    assert all(s.method == "llm" for s in result.services)
    with get_engine(settings.db_url).connect() as conn:
        calls = conn.execute(select(model_calls).where(model_calls.c.case_id == "t3")).fetchall()
    assert calls and all(c.prompt_sha256 and len(c.prompt_sha256) == 64 for c in calls)
    assert not hasattr(calls[0], "prompt")   # no prompt text is ever stored


def test_question_numbering_and_subparts():
    page = Page(page_no=1, text=(
        "Please address the following:\n\n"
        "1. What are the diagnoses?\n\n"
        "2. Was the treatment rendered to date reasonable and necessary? Address each:\n"
        "   a) Chiropractic care\n"
        "   b) MRI of the lumbar spine\n\n"
        "3. Provide an impairment rating using the AMA Guides, 6th Edition.\n\nSincerely,"
    ))
    qs = parse_questions_rules([page])
    assert [q.number for q in qs] == ["1", "2", "2a", "2b", "3"]
    assert qs[2].category == "treatment_to_date"
    assert any("AMA Guides" in i for i in qs[4].instructions)


def test_dedupe_ignores_fax_header():
    body = "Office visit note. Patient reports low back pain radiating to the left leg after the accident. " * 3
    pages = [Page(page_no=1, text=body), Page(page_no=2, text="FAX RECEIVED 04/12/2025 Page 1 of 1\n" + body),
             Page(page_no=3, text="Physical therapy daily note. Therapeutic exercise and manual therapy today. " * 3)]
    mark_duplicates(pages)
    assert pages[1].duplicate_of == 1 and pages[2].duplicate_of is None


def test_templated_notes_with_different_dates_are_not_duplicates():
    note = "Physical therapy daily note. Date of service: {d}. Therapeutic exercise 97110 x4, manual therapy " \
           "97140 x2, hot pack. Pain before 7/10, after 6/10. Patient tolerated treatment well today. " * 2
    pages = [Page(page_no=1, text=note.format(d="03/20/2025")), Page(page_no=2, text=note.format(d="03/27/2025"))]
    mark_duplicates(pages)
    assert pages[1].duplicate_of is None


def test_parse_json_tolerates_fences():
    assert parse_json('```json\n[{"a": 1}]\n```') == [{"a": 1}]
    assert parse_json('Here you go: {"page_type": "bill"} done') == {"page_type": "bill"}
