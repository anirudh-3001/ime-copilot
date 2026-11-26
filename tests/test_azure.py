"""Azure integration tests with in-memory fakes (no Azure account needed).

The fakes mirror the shapes returned by azure-ai-documentintelligence 1.0
(AnalyzeResult.pages[].lines/words, .tables[].cells/bounding_regions) and the
OpenAI v1 chat completions response used by Azure OpenAI.
"""

import json
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader

from ime_copilot import api
from ime_copilot.config import Settings
from ime_copilot.evaluate import load_truth, score
from ime_copilot.gateway import AzureOpenAIProvider
from ime_copilot.ingest.azure_di import analyze_pdf
from ime_copilot.pipeline import process_case
from ime_copilot.storage import LocalStorage, case_key
from ime_copilot.synthetic import build_case

SCANNED_PAGE = 9
SCANNED_TEXT = ["Riverbend Physical Therapy (fictional)", "PHYSICAL THERAPY DAILY NOTE",
                "Date of service: 04/26/2025 Visit #6", "Treatment: Therapeutic exercise, manual therapy, hot pack."]


class FakeDI:
    """Stands in for DocumentIntelligenceClient; reads the synthetic PDF's text layer."""

    def __init__(self, pdf_path):
        self.texts = [(p.extract_text() or "") for p in PdfReader(str(pdf_path)).pages]
        self.calls = []

    def begin_analyze_document(self, model_id, body, pages=None):
        self.calls.append((model_id, pages))
        start, end = (int(x) for x in pages.split("-"))
        result_pages, tables = [], []
        for n in range(start, end + 1):
            if n == SCANNED_PAGE:                         # poor-quality scan
                lines, conf = SCANNED_TEXT, 0.62
            else:
                lines, conf = [l for l in self.texts[n - 1].splitlines() if l.strip()], 0.99
            words = [NS(content=w, confidence=conf) for l in lines for w in l.split()]
            result_pages.append(NS(page_number=n, width=8.5, height=11, unit="inch", angle=0.0,
                                   lines=[NS(content=l, polygon=[1, 1, 7, 1, 7, 1.2, 1, 1.2]) for l in lines],
                                   words=words))
            if "CMS-1500" in self.texts[n - 1]:
                tables.append(NS(row_count=2, column_count=3, bounding_regions=[NS(page_number=n)],
                                 cells=[NS(row_index=0, column_index=0, content="DOS"),
                                        NS(row_index=0, column_index=1, content="CPT"),
                                        NS(row_index=0, column_index=2, content="CHARGES"),
                                        NS(row_index=1, column_index=0, content="03/20/2025"),
                                        NS(row_index=1, column_index=1, content="97110"),
                                        NS(row_index=1, column_index=2, content="600.00")]))
        return NS(result=lambda: NS(pages=result_pages, tables=tables))


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    return build_case(tmp_path_factory.mktemp("az"))


def _settings(tmp_path, **kw):
    s = Settings(db_url=f"sqlite:///{tmp_path / 'ledger.db'}", ocr="azure_di", extractor="rules")
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_document_intelligence_pipeline(case, tmp_path):
    pdf, truth = case
    storage = LocalStorage(tmp_path / "blobs")
    result = process_case(pdf, "az1", _settings(tmp_path), storage=storage, di_client=FakeDI(pdf))

    report = score(result, load_truth(truth))
    assert report["metrics"]["service_recall"] == 1.0
    assert report["metrics"]["page_type_accuracy"] == 1.0, report["details"]["misclassified_pages"]

    by_no = {p.page_no: p for p in result.pages}
    assert all(p.text_source == "azure_di" for p in result.pages)
    assert by_no[SCANNED_PAGE].needs_review and by_no[SCANNED_PAGE].ocr_confidence == 0.62
    assert sum(len(p.tables) for p in result.pages) == 3          # one per bill page
    assert by_no[10].tables[0].cells[1] == ["03/20/2025", "97110", "600.00"]

    for name in ("original.pdf", "layout.json", "ledger.json"):
        assert storage.exists(case_key("az1", name))
    layout = json.loads(storage.get(case_key("az1", "layout.json")))
    assert len(layout["pages"]) == 12 and layout["pages"][0]["lines"][0]["polygon"]


def test_large_files_are_sent_in_page_chunks(case):
    pdf, _ = case
    fake = FakeDI(pdf)
    pages, _ = analyze_pdf(str(pdf), fake, chunk_size=5)
    assert fake.calls == [("prebuilt-layout", "1-5"), ("prebuilt-layout", "6-10"), ("prebuilt-layout", "11-12")]
    assert [p.page_no for p in pages] == list(range(1, 13))


def test_azure_openai_provider_sends_text_and_reads_usage():
    seen = {}

    class FakeCompletions:
        def create(self, **kw):
            seen.update(kw)
            return NS(choices=[NS(message=NS(content='{"page_type": "bill"}'))],
                      usage=NS(prompt_tokens=120, completion_tokens=8))

    client = NS(chat=NS(completions=FakeCompletions()))
    text, tin, tout = AzureOpenAIProvider("", client=client).complete("gpt-6-luna", "sys", "page text", 100)
    assert (text, tin, tout) == ('{"page_type": "bill"}', 120, 8)
    assert seen["model"] == "gpt-6-luna" and seen["messages"][1] == {"role": "user", "content": "page text"}
    assert seen["max_completion_tokens"] >= 1000


def test_check_command_reports_ok_and_failures(case, capsys, monkeypatch):
    from ime_copilot.check import run_check
    from ime_copilot.gateway import FakeProvider
    s = Settings(provider="azure_openai", models={"classify": "gpt-6-luna", "extract": "gpt-6-luna",
                                                  "questions": "gpt-6.1-sol"})
    assert run_check(s, provider=FakeProvider(lambda m, u: '{"ok": true}')) is True
    out = capsys.readouterr().out
    assert "OK" in out and "gpt-6-luna" in out and "gpt-6.1-sol" in out

    def broken(model, user):
        raise RuntimeError("Error code: 404 - DeploymentNotFound")
    s_fail = Settings(provider="azure_openai", models={"classify": "wrong-name"})
    from ime_copilot import gateway
    monkeypatch.setattr(gateway.time, "sleep", lambda *_: None)   # no retry back-off in tests
    assert run_check(s_fail, provider=FakeProvider(broken)) is False
    assert "Deployment name not found" in capsys.readouterr().out

    assert run_check(Settings(provider="none", ocr="azure_di"), di_client=FakeDI(case[0])) is True


@pytest.fixture
def client(tmp_path, monkeypatch):
    s = Settings(db_url=f"sqlite:///{tmp_path / 'api.db'}", ocr="pdftext", storage="local",
                 local_storage_dir=str(tmp_path / "blobs"), api_key="test-key")
    monkeypatch.setattr(api, "settings", lambda: s)
    return TestClient(api.app)


def test_api_upload_then_read_pages(case, client):
    pdf, _ = case
    h = {"X-API-Key": "test-key"}
    r = client.post("/cases", headers=h, data={"case_id": "api1"},
                    files={"file": ("case.pdf", pdf.read_bytes(), "application/pdf")})
    assert r.status_code == 200, r.text
    assert r.json()["pages"] == 12 and r.json()["service_lines"] == 39

    pages = client.get("/cases/api1/pages", headers=h).json()
    assert len(pages) == 12 and "text" not in pages[0]
    page1 = client.get("/cases/api1/pages/1", headers=h).json()
    assert page1["page_type"] == "referral_letter" and "Independent Medical Examination" in page1["text"]
    ledger = client.get("/cases/api1/ledger", headers=h).json()
    assert len(ledger["services"]) == 39 and [q["number"] for q in ledger["questions"]][:3] == ["1", "2", "3"]


def test_api_rejects_missing_key_bad_files_and_bad_ids(client):
    assert client.get("/cases").status_code == 401
    h = {"X-API-Key": "test-key"}
    r = client.post("/cases", headers=h, files={"file": ("x.pdf", b"not a pdf", "application/pdf")})
    assert r.status_code == 415
    assert client.get("/cases/..%2Fetc/pages", headers=h).status_code in (400, 404)
    assert client.get("/cases/nope", headers=h).status_code == 404


def test_foundry_claude_provider_and_per_task_routing():
    from ime_copilot.gateway import FoundryClaudeProvider, Gateway

    seen = {}

    class FakeMessages:
        def create(self, **kw):
            seen.update(kw)
            return NS(content=[NS(type="thinking", thinking="..."), NS(type="text", text='{"ok": true}')],
                      usage=NS(input_tokens=40, output_tokens=6))

    claude = FoundryClaudeProvider(client=NS(messages=FakeMessages()))
    assert claude.complete("claude-sonnet-5-5", "sys", "hi", 50) == ('{"ok": true}', 40, 6)
    assert seen["system"] == "sys" and seen["model"] == "claude-sonnet-5-5" and seen["max_tokens"] >= 2000

    calls = []

    class Recorder:
        def __init__(self, name):
            self.name = name

        def complete(self, model_id, system, user, max_tokens):
            calls.append((self.name, model_id))
            return '{"ok": true}', 10, 2

    s = Settings(provider="azure_openai",
                 models={"extract": "gpt-6-luna", "questions": "foundry_claude:claude-sonnet-5-5"})
    gw = Gateway(s)
    gw._providers = {"azure_openai": Recorder("azure_openai"), "foundry_claude": Recorder("foundry_claude")}
    gw.complete_json("extract", "s", "u")
    gw.complete_json("questions", "s", "u")
    assert calls == [("azure_openai", "gpt-6-luna"), ("foundry_claude", "claude-sonnet-5-5")]
    assert gw.cost_usd > 0                                   # Sonnet price found from the model name
