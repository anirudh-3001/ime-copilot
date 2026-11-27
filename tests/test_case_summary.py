"""Pinned case summary: facts from the file, and the report-style written summary."""

import json
import re

import pytest
from fastapi.testclient import TestClient

from ime_copilot import api
from ime_copilot.case_summary import build_summary, case_context, load_overview, write_overview
from ime_copilot.config import Settings
from ime_copilot.db import get_engine
from ime_copilot.gateway import FakeProvider, Gateway
from ime_copilot.pipeline import process_case
from ime_copilot.rag.qa import CaseQA
from ime_copilot.synthetic import build_case


@pytest.fixture(scope="module")
def done(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sum")
    pdf, _ = build_case(tmp)
    s = Settings(db_url=f"sqlite:///{tmp / 's.db'}", ocr="auto", index_dir=str(tmp / "index"))
    r = process_case(pdf, "sum1", s, progress=False)
    return s, r


def test_facts_summary_has_details_billing_questions(done):
    s, _ = done
    f = build_summary(get_engine(s.db_url), "sum1")
    labels = {d["label"]: d for d in f["details"]}
    assert labels["Date of birth"]["value"] == "1981-06-02" and labels["Date of birth"]["page"]
    assert f["billing"]["lines"] == 39 and f["billing"]["total"] > 0
    by_code = {c["code"]: c for c in f["billing"]["by_code"]}
    assert by_code["64483"]["lines"] == 6 and by_code["64483"]["first_date"] == "2025-05-06"
    assert f["file"]["pages"] == 12 and f["file"]["duplicates"] == 1
    assert [q["number"] for q in f["questions"]][:3] == ["1", "2", "3"]
    assert build_summary(get_engine(s.db_url), "nope") is None


def test_written_summary_is_the_models_own_and_linked_to_pages(done, tmp_path):
    s, r = done
    seen = {}

    def write(model, user):
        seen["user"] = user
        n = re.search(r"\[(\d+)\] \(p\. 3\)", user).group(1)        # the MRI page
        return json.dumps({"summary": "## What happened\nRear-end collision on 03/14/2025 [F1].\n\n"
                                      f"## Imaging\n- Lumbar MRI was performed [{n}].\n\n"
                                      "## For the examiner\n- Treatment continued until 10/07/2025 [F2]. "
                                      "A visit on 12/12/2029 is mentioned [F2]."})
    gw = Gateway(Settings(provider="local", models={"answer": "gemma4:26b-a4b"}), provider=FakeProvider(write))
    qa = CaseQA(r.pages, r.services, gateway=gw, use_embeddings=False,
                details=json.loads(r.details.model_dump_json()))
    ctx = case_context(build_summary(get_engine(s.db_url), "sum1"))
    steps = []
    out = write_overview(qa, tmp_path, "sum1", case_context=ctx, on_progress=lambda d, t: steps.append((d, t)))
    assert "CASE DETAILS" in seen["user"] and "FACT SHEET" in seen["user"] and "PASSAGES" in seen["user"]
    assert len(re.findall(r"^\[\d+\] \(p\. \d+\)$", seen["user"], re.M)) >= 8      # a wide sample, not one search
    assert out["method"] == "model" and "## Imaging" in out["text"] and "(p. 3)" in out["text"]
    assert "[F1]" not in out["text"] and out["sources"]
    assert any("12/12/2029" in w for w in out["warnings"])          # made-up date flagged
    assert steps[-1] == (1, 1)
    assert load_overview(tmp_path, "sum1")["version"] == 2


def test_written_summary_failure_is_reported(done, tmp_path):
    s, r = done
    gw = Gateway(Settings(provider="local", models={"answer": "g"}),
                 provider=FakeProvider(lambda m, u: "not json at all"))
    out = write_overview(CaseQA(r.pages, r.services, gateway=gw, use_embeddings=False), tmp_path, "sum1")
    assert out["method"] == "failed" and out["warnings"]


def test_summary_endpoint(done, monkeypatch):
    s, _ = done
    monkeypatch.setattr(api, "settings", lambda: s)
    c = TestClient(api.app)
    d = c.get("/cases/sum1/summary").json()
    assert d["facts"]["billing"]["lines"] == 39
    assert d["overview_status"] == "unavailable"            # no model in this test
    assert c.post("/cases/sum1/summary").status_code == 400
    assert c.get("/cases/nope/summary").status_code == 404


def test_case_label_can_be_changed_and_survives_reprocessing(done, monkeypatch, tmp_path):
    s, r = done
    monkeypatch.setattr(api, "settings", lambda: s)
    c = TestClient(api.app)
    assert c.patch("/cases/sum1", json={"display_name": "  Sample,   Jordan  "}).json()["display_name"] == "Sample, Jordan"
    listed = {x["case_id"]: x for x in c.get("/cases").json()}
    assert listed["sum1"]["display_name"] == "Sample, Jordan" and listed["sum1"]["details"]["patient_name"]
    pdf, _ = build_case(tmp_path)
    process_case(pdf, "sum1", s, progress=False)                     # processed again: label kept
    assert c.get("/cases/sum1").json()["display_name"] == "Sample, Jordan"
    assert c.patch("/cases/sum1", json={"display_name": ""}).json()["display_name"] is None
