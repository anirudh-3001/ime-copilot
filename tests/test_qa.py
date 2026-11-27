import json
import re

import pytest

from ime_copilot.config import Settings
from ime_copilot.gateway import FakeProvider, Gateway, LocalProvider
from ime_copilot.pipeline import process_case
from ime_copilot.rag.evalqa import QUESTIONS, evaluate
from ime_copilot.extract.citations import normalize
from ime_copilot.rag.qa import NOT_FOUND, CaseQA
from ime_copilot.synthetic import build_case


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("qa")
    pdf, _ = build_case(tmp)
    s = Settings(db_url=f"sqlite:///{tmp / 'qa.db'}", ocr="auto")
    r = process_case(pdf, "qa1", s, progress=False)
    return r.pages, r.services


def _settings(**models):
    return Settings(provider="local", models=models)


def _first_passage_quote(user: str) -> tuple[int, str]:
    m = re.search(r"\[\d+\] \(p\. (\d+)[^)]*\)\n(.+)", user)
    return int(m.group(1)), m.group(2).strip()[:60]


def test_extractive_mode_finds_pages_and_cpts(processed):
    qa = CaseQA(*processed, gateway=None)
    rows = evaluate(qa)
    assert all(r["page_found"] for r in rows), rows
    assert all(r["cpt_recall"] == 1.0 for r in rows), rows


def test_model_answer_is_verified(processed):
    def good(model, user):
        page, quote = _first_passage_quote(user)
        return json.dumps({"answer": f"The records show it (p. {page}).",
                           "citations": [{"page": page, "quote": quote}], "cpt_codes": ["64483"]})
    gw = Gateway(_settings(answer="gemma4:26b-a4b"), provider=FakeProvider(good))
    r = CaseQA(*processed, gateway=gw, use_embeddings=False).ask("lumbar epidural steroid injections dates")
    assert r.method == "model" and r.sources and not r.warnings
    assert "64483" in {c.code for c in r.cpt_codes}


def test_made_up_citations_and_codes_are_rejected(processed):
    def bad(model, user):
        return json.dumps({"answer": "Twenty injections were given (p. 999).",
                           "citations": [{"page": 6, "quote": "twenty injections were administered weekly"}],
                           "cpt_codes": ["64490"]})
    gw = Gateway(_settings(answer="gemma4:26b-a4b"), provider=FakeProvider(bad))
    r = CaseQA(*processed, gateway=gw, use_embeddings=False).ask("How many epidural injections were done?")
    assert r.method == "unverified"                     # falls back to the real passages
    assert "Twenty" not in r.answer and r.sources
    assert all(s.page != 999 for s in r.sources)
    assert "64490" not in {c.code for c in r.cpt_codes}
    assert any("were removed" in w for w in r.warnings)


def test_escalation_when_local_answer_fails(processed):
    def responder(model, user):
        if model == "gemma4:26b-a4b":
            return "not json at all"
        page, quote = _first_passage_quote(user)
        return json.dumps({"answer": f"Answer from the second model (p. {page}).",
                           "citations": [{"page": page, "quote": quote}], "cpt_codes": []})
    gw = Gateway(_settings(answer="gemma4:26b-a4b", answer_escalate="gpt-6.1-sol"), provider=FakeProvider(responder))
    r = CaseQA(*processed, gateway=gw, use_embeddings=False).ask("What did the MRI show?")
    assert r.method == "escalated" and r.model == "gpt-6.1-sol"


def test_not_found_answer_is_allowed(processed):
    gw = Gateway(_settings(answer="gemma4:26b-a4b"),
                 provider=FakeProvider(lambda m, u: json.dumps({"answer": NOT_FOUND, "citations": [], "cpt_codes": []})))
    r = CaseQA(*processed, gateway=gw, use_embeddings=False).ask("What is the claimant's blood type?")
    assert r.method == "model" and r.answer == NOT_FOUND


def test_hybrid_search_with_embeddings(processed):
    vocab = ["therapy", "injection", "mri", "visit", "diagnos", "bill"]

    def embed(texts):
        return [[float(t.lower().count(w)) + 0.01 for w in vocab] for t in texts]
    gw = Gateway(_settings(answer="", embed="qwen3-embedding:0.6b"))
    gw.embed = embed                                         # stand-in for the local embedding model
    qa = CaseQA(*processed, gateway=gw)
    assert qa.index.vectors is not None
    hits = qa.index.search("physical therapy visits", k=5)
    assert any(h.vector_rank is not None for h in hits)


def test_local_provider_uses_openai_style_client():
    seen = {}

    class Chat:
        def create(self, **kw):
            seen.update(kw)
            from types import SimpleNamespace as NS
            return NS(choices=[NS(message=NS(content='{"ok": true}'))], usage=NS(prompt_tokens=5, completion_tokens=2))

    from types import SimpleNamespace as NS
    p = LocalProvider(client=NS(chat=NS(completions=Chat())))
    assert p.complete("gemma4:26b-a4b", "s", "u", 50) == ('{"ok": true}', 5, 2)
    assert seen["model"] == "gemma4:26b-a4b" and seen["temperature"] == 0


def test_question_set_is_well_formed():
    assert len(QUESTIONS) >= 5 and all(q["pages"] for q in QUESTIONS)


def test_local_provider_uses_ollama_native_api_with_json_and_context(monkeypatch):
    import io
    import json as _json
    import urllib.request
    sent = {}

    def fake_urlopen(req, timeout=None):
        sent["url"], sent["body"] = req.full_url, _json.loads(req.data)
        reply = {"message": {"content": '{"ok": true}'}, "prompt_eval_count": 7, "eval_count": 3}
        return io.BytesIO(_json.dumps(reply).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    p = LocalProvider("http://localhost:11434/v1", num_ctx=8192)
    assert p.complete("gemma4:26b-a4b", "s", "u", 50) == ('{"ok": true}', 7, 3)
    assert sent["url"] == "http://localhost:11434/api/chat"
    body = sent["body"]
    assert body["format"] == "json" and body["think"] is False and body["stream"] is False
    assert body["options"]["num_ctx"] == 8192 and body["options"]["temperature"] == 0


def test_passage_number_citations_become_pages_with_real_quotes(processed):
    """The model cites [n]; the system fills in the page number and takes the quote from the page."""
    def answer(model, user):
        n_pages = dict(re.findall(r"\[(\d+)\] \(p\. (\d+)", user))
        bill = next(n for n, p in n_pages.items() if p == "12")
        return json.dumps({"answer": f"Six lumbar epidural injections (CPT 64483) were billed, the first on "
                                     f"05/06/2025 for $1,150.00 [{bill}]. Another was on 01/01/2030 [{bill}][99].",
                           "cpt_codes": ["64483"]})
    pages = {p.page_no: p for p in processed[0]}
    gw = Gateway(_settings(answer="gemma4:26b-a4b"), provider=FakeProvider(answer))
    r = CaseQA(*processed, gateway=gw, use_embeddings=False).ask(
        "How many lumbar epidural injections were billed and on which dates?")
    assert r.method == "model"
    assert "(p. 12)" in r.answer and "[" not in r.answer
    assert r.sources and all(normalize(s.quote) in normalize(pages[s.page].text) for s in r.sources)
    assert any("not in the search results" in w for w in r.warnings)          # [99] removed
    assert any("01/01/2030" in w for w in r.warnings)                         # unsupported date flagged
    assert not any("05/06/2025" in w or "1,150" in w for w in r.warnings)     # supported values pass


def test_answer_without_any_usable_marker_falls_back(processed):
    gw = Gateway(_settings(answer="gemma4:26b-a4b"),
                 provider=FakeProvider(lambda m, u: json.dumps({"answer": "Some injections were done.", "cpt_codes": []})))
    r = CaseQA(*processed, gateway=gw, use_embeddings=False).ask("How many epidural injections were done?")
    assert r.method == "unverified" and r.sources


def test_timeline_question_uses_case_facts_from_the_whole_file(processed):
    """'How long has the patient been treated?' needs the whole file's bills, not just the top passages."""
    seen = {}

    def answer(model, user):
        seen["user"] = user
        return json.dumps({"answer": "The billed treatment runs from 03/18/2025 to 10/07/2025, about "
                                     "7 months [F1].", "cpt_codes": []})
    gw = Gateway(_settings(answer="gemma4:26b-a4b"), provider=FakeProvider(answer))
    qa = CaseQA(*processed, gateway=gw, use_embeddings=False)
    assert "first 2025-03-18" in qa.facts[0]["text"] and "most recent 2025-10-07" in qa.facts[0]["text"]
    r = qa.ask("How long has this patient been under treatment, from the start until now?")
    assert "CASE FACTS" in seen["user"] and "[F1]" in seen["user"]
    assert r.method == "model" and "(p. " in r.answer and "[F1]" not in r.answer
    assert {s.page for s in r.sources} & {8, 9, 10, 11, 12}      # bill pages, quotes from the page itself
    assert not any("not found in the cited text" in w for w in r.warnings)


def test_follow_up_question_is_planned_searched_and_answered_with_context(processed):
    """'And what did it show?' after an MRI question: the planner resolves 'it', searches for the MRI,
    and the answer prompt carries the conversation, the case overview and page types."""
    prompts = {}

    def model(name, user):
        if "PASSAGES" not in user:                                   # the search-planning call
            prompts["plan"] = user
            return json.dumps({"standalone_question": "What did the lumbar MRI show?",
                               "searches": ["MRI lumbar spine impression", "disc protrusion"]})
        prompts["answer"] = user
        n = re.search(r"\[(\d+)\] \(p\. 3, imaging report\)", user).group(1)
        return json.dumps({"answer": f"The lumbar MRI showed findings at L4-L5 [{n}].", "cpt_codes": ["72148"]})

    gw = Gateway(_settings(answer="gemma4:26b-a4b"), provider=FakeProvider(model))
    history = [{"question": "Was an MRI done?", "answer": "Yes, a lumbar MRI was done (p. 3)."}]
    r = CaseQA(*processed, gateway=gw, use_embeddings=False).ask(
        "And what did it show?", history=history, overview="## Overview\nRear-end collision, low back pain.")
    assert "RECENT CONVERSATION" in prompts["plan"] and "Was an MRI done?" in prompts["plan"]
    for part in ("CASE OVERVIEW", "RECENT CONVERSATION", "(Meaning: What did the lumbar MRI show?)", "imaging report"):
        assert part in prompts["answer"]
    assert 3 in r.searched_pages and r.method == "model" and "(p. 3)" in r.answer
    assert "72148" in {c.code for c in r.cpt_codes}


def test_bad_search_plan_falls_back_to_the_question(processed):
    def model(name, user):
        if "PASSAGES" not in user:
            return "I cannot plan"                                   # not JSON: plan is skipped
        page, quote = _first_passage_quote(user)
        return json.dumps({"answer": f"Found (p. {page}).", "citations": [{"page": page, "quote": quote}]})
    gw = Gateway(_settings(answer="gemma4:26b-a4b"), provider=FakeProvider(model))
    r = CaseQA(*processed, gateway=gw, use_embeddings=False).ask("What did the MRI show?")
    assert r.method == "model" and 3 in r.searched_pages
