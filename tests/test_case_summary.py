"""Pinned case summary: facts from the file, and the report-style written summary."""

import json
import re

import pytest
from fastapi.testclient import TestClient

from ime_copilot import api
from ime_copilot.case_summary import OVERVIEW_QUESTIONS, build_summary, load_overview, write_overview
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


def test_written_summary_one_verified_section_per_heading(done, tmp_path):
    s, r = done

    def answer(model, user):
        m = re.search(r"\[p\. (\d+)\]\n(.+)", user)
        return json.dumps({"answer": f"The records describe this (p. {m.group(1)}).",
                           "citations": [{"page": int(m.group(1)), "quote": m.group(2).strip()[:60]}],
                           "cpt_codes": []})
    gw = Gateway(Settings(provider="local", models={"answer": "gemma4:26b-a4b"}), provider=FakeProvider(answer))
    qa = CaseQA(r.pages, r.services, gateway=gw, use_embeddings=False)
    steps = []
    out = write_overview(qa, tmp_path, "sum1", on_progress=lambda d, t: steps.append(d))
    assert [x["title"] for x in out["sections"]] == [t for _, t, _ in OVERVIEW_QUESTIONS]
    assert all(x["method"] == "model" and x["sources"] for x in out["sections"])
    assert steps[-1] == len(OVERVIEW_QUESTIONS)
    assert load_overview(tmp_path, "sum1")["sections"][0]["title"] == "History of present illness"


def test_summary_endpoint(done, monkeypatch):
    s, _ = done
    monkeypatch.setattr(api, "settings", lambda: s)
    c = TestClient(api.app)
    d = c.get("/cases/sum1/summary").json()
    assert d["facts"]["billing"]["lines"] == 39
    assert d["overview_status"] == "unavailable"            # no model in this test
    assert c.post("/cases/sum1/summary").status_code == 400
    assert c.get("/cases/nope/summary").status_code == 404
