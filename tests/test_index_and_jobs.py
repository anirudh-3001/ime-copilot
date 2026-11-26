"""Saved search index, background uploads with progress, and CPT codes for non-billing questions."""

import time

import pytest
from fastapi.testclient import TestClient

from ime_copilot import api, jobs
from ime_copilot.config import Settings
from ime_copilot.rag.index import CaseIndex, make_passages
from ime_copilot.rag.store import DiskVectorCache, index_path
from ime_copilot.pipeline import process_case
from ime_copilot.schemas import Page
from ime_copilot.synthetic import build_case


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    return build_case(tmp_path_factory.mktemp("jobs"))


@pytest.fixture(scope="module")
def processed(case, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("jobsqa")
    r = process_case(case[0], "jq1", Settings(db_url=f"sqlite:///{tmp / 'q.db'}", ocr="auto"), progress=False)
    return r.pages, r.services


def _fake_embed(calls):
    def embed(texts):
        calls.append(len(texts))
        return [[float(len(t) % 7 + 1), float(t.count("e") + 1), 1.0] for t in texts]
    return embed


def _pages(n=30):
    return [Page(page_no=i, text=f"Page {i} lumbar epidural injection note number {i} " * 30) for i in range(1, n + 1)]


def test_vectors_are_saved_and_reused(tmp_path):
    path = index_path(tmp_path, "case1", "qwen3-embedding:0.6b")
    calls, progress = [], []
    CaseIndex.build(_pages(), embedder=_fake_embed(calls), vector_cache=DiskVectorCache(path),
                    on_progress=lambda d, t: progress.append((d, t)))
    first = sum(calls)
    assert first > 0 and path.exists() and progress[-1][0] == progress[-1][1]

    calls.clear()                                   # a new run (e.g. after a restart) embeds nothing
    idx = CaseIndex.build(_pages(), embedder=_fake_embed(calls), vector_cache=DiskVectorCache(path))
    assert sum(calls) == 0 and len(idx.vectors) == len(make_passages(_pages()))
    assert idx.search("epidural injection", k=3)


def test_interrupted_indexing_resumes(tmp_path):
    path = index_path(tmp_path, "case2", "m")
    pages = _pages(200)
    total = len(make_passages(pages))
    state = {"n": 0}

    def flaky(texts):
        state["n"] += 1
        if state["n"] == 3:
            raise RuntimeError("Ollama stopped")
        return [[1.0, 2.0, 3.0] for _ in texts]

    try:
        CaseIndex.build(pages, embedder=flaky, vector_cache=DiskVectorCache(path))
    except RuntimeError:
        pass
    saved = len(DiskVectorCache(path))
    assert 0 < saved < total                       # two batches were saved before the failure
    calls = []
    CaseIndex.build(pages, embedder=_fake_embed(calls), vector_cache=DiskVectorCache(path))
    assert sum(calls) == total - saved             # only the rest is embedded


def test_background_upload_reports_progress_then_answers(case, tmp_path, monkeypatch):
    pdf, _ = case
    s = Settings(db_url=f"sqlite:///{tmp_path / 'bg.db'}", ocr="pdftext", storage="local",
                 local_storage_dir=str(tmp_path / "blobs"), index_dir=str(tmp_path / "index"))
    monkeypatch.setattr(api, "settings", lambda: s)
    client = TestClient(api.app)
    r = client.post("/cases", data={"case_id": "bg1", "wait": "false"},
                    files={"file": ("case.pdf", pdf.read_bytes(), "application/pdf")})
    assert r.status_code == 200 and r.json()["status"] == "processing"

    for _ in range(200):
        st = client.get("/cases/bg1/status").json()
        if st["stage"] in ("ready", "error"):
            break
        time.sleep(0.05)
    assert st["stage"] == "ready", st
    assert st["pages"] == 12 and st["service_lines"] == 39

    a = client.post("/cases/bg1/ask", json={"question": "How many lumbar epidural injections were billed?"})
    assert a.status_code == 200 and "64483" in [c["code"] for c in a.json()["cpt_codes"]]
    assert client.get("/cases/nope/status").status_code == 404


def test_ask_is_refused_while_processing(monkeypatch, tmp_path):
    s = Settings(db_url=f"sqlite:///{tmp_path / 'x.db'}")
    monkeypatch.setattr(api, "settings", lambda: s)
    jobs._update("busy1", "ocr", 5, 100)
    r = TestClient(api.app).post("/cases/busy1/ask", json={"question": "anything"})
    assert r.status_code == 409


def test_diagnosis_question_gets_no_billing_codes(processed):
    from ime_copilot.rag.qa import CaseQA
    r = CaseQA(*processed, gateway=None, use_embeddings=False).ask("What are the patient's diagnoses?")
    assert r.cpt_codes == []
