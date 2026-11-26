"""Ask a question about one case file; get an answer, the related CPT codes and source pages.

Flow:
  1. Search (RAG): the most relevant passages from the file, with page numbers.
  2. Answer: the local model (default) writes an answer from those passages only.
  3. Verify: every cited page must be among the searched pages and its quote must be
     on that page; every CPT code must exist in the document. Anything else is dropped.
  4. If the answer fails verification and an escalation model is configured
     (IME_MODEL_ANSWER_ESCALATE, e.g. Azure OpenAI), the same passages go to it,
     and its answer is verified the same way. Otherwise the best-matching
     passages are returned with a clear note. Nothing unverified is presented as fact.
  5. No model configured at all: an extractive answer (best-matching lines + pages).
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from ..extract.citations import normalize, quote_on_page
from ..gateway import Gateway, GatewayError
from ..schemas import Citation, Page, ServiceRow
from .cpt import codes_for_question, find_code_mentions, summarize
from .index import CaseIndex, Hit, tokenize

NOT_FOUND = "The retrieved pages of this document do not answer this question."


class SourceRef(BaseModel):
    page: int
    quote: str


class CptRef(BaseModel):
    code: str
    category: str
    pages: list[int]
    billed_lines: int = 0
    units: int = 0
    billed_total: float = 0.0
    first_date: str | None = None
    last_date: str | None = None
    why: str = ""


class AskResult(BaseModel):
    question: str
    answer: str
    method: str                          # model | escalated | extractive | unverified
    model: str | None = None
    cpt_codes: list[CptRef] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)
    searched_pages: list[int] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    model_cost_usd: float = 0.0


_SYSTEM = """You answer questions about ONE medical claim file, using ONLY the numbered passages given.
Rules:
1. Use only facts stated in the passages. Never use outside knowledge about this patient.
2. Every factual statement must cite its page like (p. 12).
3. For each citation, copy a short EXACT quote (5 to 25 words) from that page's passage, character for character.
4. If the passages do not contain the answer, reply exactly: "%s"
5. Do not give medical opinions (causation, necessity, MMI); report what the records say.
6. List CPT/HCPCS codes from the passages that relate to the question.
Reply with JSON only:
{"answer": "<answer with (p. N) citations>", "citations": [{"page": <int>, "quote": "<exact text>"}], "cpt_codes": ["<code>", ...]}""" % NOT_FOUND


class CaseQA:
    """Holds one case's pages, ledger, search index and code mentions; answers questions."""

    def __init__(self, pages: list[Page], services: list[ServiceRow], gateway: Gateway | None = None,
                 use_embeddings: bool = True, vector_cache: dict | None = None):
        self.pages = pages
        self.pages_by_no = {p.page_no: p for p in pages}
        self.services = services
        self.gateway = gateway
        embedder = None
        self.notes: list[str] = []
        if use_embeddings and gateway is not None and gateway.has_model("embed"):
            embedder = gateway.embed
        try:
            self.index = CaseIndex.build(pages, embedder=embedder, vector_cache=vector_cache)
        except GatewayError as err:        # e.g. Ollama not running: keyword search still works
            self.notes.append(f"Meaning-based search unavailable ({err}); using keyword search only.")
            self.index = CaseIndex.build(pages)
        self.mentions = find_code_mentions(pages, {s.cpt for s in services})

    def ask(self, question: str, k: int = 8) -> AskResult:
        cost_before = self.gateway.cost_usd if self.gateway else 0.0
        hits = self.index.search(question, k=k)
        searched = sorted({h.passage.page_no for h in hits})
        result = AskResult(question=question, answer=NOT_FOUND, method="extractive", searched_pages=searched,
                           warnings=list(self.notes))
        if not hits:
            result.warnings.append("No matching text found in the document.")
            return result

        model_reply, llm_codes = None, []
        if self.gateway is not None and self.gateway.has_model("answer"):
            model_reply = self._model_answer("answer", question, hits, result)
            if model_reply is None and self.gateway.has_model("answer_escalate"):
                result.warnings.append("Local answer could not be verified; escalated to the second model.")
                model_reply = self._model_answer("answer_escalate", question, hits, result)
                if model_reply is not None:
                    result.method = "escalated"

        if model_reply is not None:
            answer, sources, llm_codes, model_name = model_reply
            result.answer, result.sources, result.model = answer, sources, model_name
            if result.method != "escalated":
                result.method = "model"
        else:
            if self.gateway is not None and self.gateway.has_model("answer"):
                result.method = "unverified"
                result.warnings.append("No answer could be verified against the document; "
                                       "showing the best-matching passages instead.")
            result.answer, result.sources = self._extractive(question, hits)

        cited_pages = {s.page for s in result.sources}
        why = codes_for_question(question, set(searched) | cited_pages, self.mentions, self.services)
        in_doc = {m.code for m in self.mentions} | {s.cpt for s in self.services}
        for code in llm_codes:
            code = code.strip().upper()
            if code in in_doc:
                why.setdefault(code, "named in the answer")
            elif code:
                result.warnings.append(f"Model mentioned CPT {code}, which is not in the document; dropped.")
        result.cpt_codes = [CptRef(**vars(s)) for s in summarize(set(why), self.mentions, self.services, why)]
        if self.gateway:
            result.model_cost_usd = round(self.gateway.cost_usd - cost_before, 6)
        return result

    # ------------------------------------------------------------------ model answer
    def _model_answer(self, task: str, question: str, hits: list[Hit], result: AskResult):
        searched = {h.passage.page_no for h in hits}
        passages = "\n\n".join(f"[p. {h.passage.page_no}]\n{h.passage.text}" for h in hits)
        codes_here = sorted({m.code for m in self.mentions if m.page_no in searched})
        user = (f"QUESTION: {question}\n\nPASSAGES:\n{passages}\n\n"
                f"CPT/HCPCS codes found on these pages: {', '.join(codes_here) or 'none'}")
        try:
            reply = self.gateway.complete_json(task, _SYSTEM, user, max_tokens=1500)
        except (GatewayError, ValueError) as err:
            result.warnings.append(f"{task} model unavailable: {err}")
            return None
        if not isinstance(reply, dict):
            result.warnings.append(f"{task} model returned an unexpected format.")
            return None

        answer = str(reply.get("answer", "")).strip()
        sources, dropped = [], 0
        for c in reply.get("citations") or []:
            try:
                page, quote = int(c.get("page")), str(c.get("quote", "")).strip()
            except (TypeError, ValueError, AttributeError):
                dropped += 1
                continue
            if page in searched and _quote_ok(page, quote, self.pages_by_no):
                sources.append(SourceRef(page=page, quote=quote))
            else:
                dropped += 1
        if dropped:
            result.warnings.append(f"{dropped} citation(s) from the {task} model were not found in the document "
                                   "and were removed.")

        named_pages = {int(n) for n in re.findall(r"\bp(?:age|g)?\.?\s*(\d{1,5})\b", answer, re.IGNORECASE)}
        bad_pages = sorted(named_pages - {s.page for s in sources} - searched)
        not_found = normalize(answer).startswith(normalize(NOT_FOUND)[:40])
        if not answer or bad_pages or (not sources and not not_found):
            if bad_pages:
                result.warnings.append(f"Answer referred to page(s) {bad_pages} that were not in the search results.")
            return None
        model_name = self.gateway.settings.models.get(task)
        return answer, sources, [str(x) for x in reply.get("cpt_codes") or []], model_name

    # ------------------------------------------------------------------ extractive answer
    def _extractive(self, question: str, hits: list[Hit], n: int = 5) -> tuple[str, list[SourceRef]]:
        terms = set(tokenize(question))
        sources: list[SourceRef] = []
        seen_pages: set[int] = set()
        for h in hits:
            if len(sources) >= n:
                break
            best = _best_line(h.passage.text, terms)
            if not best or h.passage.page_no in seen_pages:
                continue
            if quote_on_page(Citation(page_no=h.passage.page_no, quote=best), self.pages_by_no):
                sources.append(SourceRef(page=h.passage.page_no, quote=best))
                seen_pages.add(h.passage.page_no)
        if not sources:
            return NOT_FOUND, []
        lines = "\n".join(f"- p. {s.page}: “{s.quote}”" for s in sources)
        return f"Best-matching passages in the document (no AI model used):\n{lines}", sources


def _best_line(text: str, terms: set[str], max_len: int = 300) -> str:
    candidates = [c.strip() for c in re.split(r"\n|(?<=[.!?])\s+", text) if len(c.strip()) > 15]
    if not candidates:
        return text.strip()[:max_len]
    best = max(candidates, key=lambda c: (len(terms & set(tokenize(c))), -len(c)))
    return best[:max_len]


def _quote_ok(page: int, quote: str, pages_by_no: dict[int, Page]) -> bool:
    """Exact match after whitespace/quote normalization; quotes with '...' are checked piece by piece."""
    if len(quote) < 8:
        return False
    parts = [p.strip() for p in re.split(r"\.\.\.|…", quote) if len(p.strip()) >= 8]
    return bool(parts) and all(quote_on_page(Citation(page_no=page, quote=p), pages_by_no) for p in parts)
