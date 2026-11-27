"""Ask a question about one case file; get an answer, the related CPT codes and source pages.

Flow:
  1. Search (RAG): the most relevant passages from the file, with page numbers.
  2. Answer: the local model (default) writes an answer from those passages only and
     marks each fact with the number of the passage it came from, like [3].
  3. Verify: passage numbers are turned into page numbers by the system, and the quote
     shown for each page is taken from the page itself, so a page or quote can't be made
     up. Dates, amounts and codes in the answer are checked against the cited passages
     (flagged if missing); every CPT code must exist in the document.
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


_SYSTEM = """You are an experienced medical-legal analyst assisting a physician who is preparing an
independent medical examination (IME). Answer the question about ONE claim file.

You are given: a CASE OVERVIEW (background, may be missing), CASE FACTS computed from every bill in
the whole file, the RECENT CONVERSATION (for follow-up questions), and numbered PASSAGES from the
pages most related to the question. Each passage shows its page number and page type.

How to answer:
- Think like an analyst, not a search engine: connect information across passages, put events in
  date order, work out time spans and counts, and point out conflicts, gaps or missing records.
- Keep care for the claimed injury separate from records dated before the injury and from
  unrelated care (other illnesses, routine visits). Leave those out unless they matter to the
  question (for example prior injuries to the same body part), and label them when you include them.
- When you state a time span, give both dates (e.g. "from 02/21/2024 to 08/03/2026, about ...").
- Answer the question directly in the first sentence, then give the supporting details.
- Write clear, complete sentences in a professional clinical tone. Use short bullet points ("- ")
  for lists of visits, dates, treatments or findings.
- Be specific: dates, providers, body parts, findings, CPT codes, units and billed amounts.
- After every fact, put the number of the passage or case fact that directly supports it in square
  brackets, e.g. [2] or [F3]: cite the one or two best sources, not every passage. Use only numbers
  that exist. Never write page numbers yourself. The overview
  and conversation are background only: don't cite them, cite the passage or fact instead.
- For dates, timelines, "how long", counts and totals, use the CASE FACTS. "Until now" means up
  to the most recent date in the file; say what that date is.
- Use only what the given material says. If it partly answers, give that part and say what is
  not found. If it does not answer the question at all, reply exactly: "%s"
- Do not give your own medical opinion on causation, necessity, MMI or impairment; report and
  organize what the records say, including any opinions the records themselves contain.

Reply with JSON only:
{"answer": "<answer with [n] markers>", "cpt_codes": ["<CPT/HCPCS codes from the material that relate to the question>"]}""" % NOT_FOUND

_PLAN_SYSTEM = """You plan searches in a long medical claim file (medical records, bills, reports).
Given a question (and possibly the recent conversation, for follow-ups), reply with JSON only:
{"standalone_question": "<the question rewritten to stand alone, resolving words like 'it' or 'that'>",
 "searches": ["<2 to 4 short search phrases using words the records would contain>"]}
Example: "How long was she treated?" -> searches like "initial evaluation date", "most recent visit
follow-up", "discharged from care", "date of injury"."""

_PAGE_TYPE_LABEL = {"bill": "bill", "office_note": "office note", "pt_note": "therapy note",
                    "imaging_report": "imaging report", "operative_report": "procedure report",
                    "referral_letter": "IME referral letter", "other": "record"}


class CaseQA:
    """Holds one case's pages, ledger, search index and code mentions; answers questions."""

    def __init__(self, pages: list[Page], services: list[ServiceRow], gateway: Gateway | None = None,
                 use_embeddings: bool = True, vector_cache: dict | None = None, on_progress=None,
                 details: dict | None = None):
        self.pages = pages
        self.pages_by_no = {p.page_no: p for p in pages}
        self.services = services
        self.gateway = gateway
        embedder = None
        self.notes: list[str] = []
        if use_embeddings and gateway is not None and gateway.has_model("embed"):
            embedder = gateway.embed
        try:
            self.index = CaseIndex.build(pages, embedder=embedder, vector_cache=vector_cache,
                                         on_progress=on_progress)
        except GatewayError as err:        # e.g. Ollama not running: keyword search still works
            self.notes.append(f"Meaning-based search unavailable ({err}); using keyword search only.")
            self.index = CaseIndex.build(pages)
        self.mentions = find_code_mentions(pages, {s.cpt for s in services})
        self.facts = case_facts(services, details or {})

    def ask(self, question: str, k: int = 10, history: list[dict] | None = None,
            overview: str | None = None) -> AskResult:
        """history: recent turns [{"question", "answer"}] for follow-ups; overview: the case summary text."""
        cost_before = self.gateway.cost_usd if self.gateway else 0.0
        history = [h for h in (history or []) if h.get("question")][-3:]
        standalone, searches = self._plan(question, history)

        # Code-aware search: a bill line says "64483 ... 1,150.00", not "epidural injection", so
        # the codes in this file that match the question's topic are added to the search words.
        topic_codes = sorted(codes_for_question(standalone, set(), self.mentions, self.services))
        query = standalone + (" " + " ".join(topic_codes[:20]) if topic_codes else "")
        hits = _merge_hits([self.index.search(query, k=k)] + [self.index.search(q, k=5) for q in searches],
                           limit=k + 2)
        searched = sorted({h.passage.page_no for h in hits})
        result = AskResult(question=question, answer=NOT_FOUND, method="extractive", searched_pages=searched,
                           warnings=list(self.notes))
        if not hits:
            result.warnings.append("No matching text found in the document.")
            return result

        model_reply, llm_codes = None, []
        if self.gateway is not None and self.gateway.has_model("answer"):
            context = (history, overview, standalone)
            model_reply = self._model_answer("answer", question, hits, result, context)
            if model_reply is None and self.gateway.has_model("answer_escalate"):
                result.warnings.append("Local answer could not be verified; escalated to the second model.")
                model_reply = self._model_answer("answer_escalate", question, hits, result, context)
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
            if result.method == "unverified" and result.sources:
                result.answer = result.answer.replace(
                    "Best-matching passages in the document (no AI model used):",
                    "The AI answer could not be checked against the file, so here are the best-matching passages:")

        cited_pages = {s.page for s in result.sources}
        why = codes_for_question(standalone, cited_pages or set(searched), self.mentions, self.services)
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

    # ------------------------------------------------------------------ search planning
    def _plan(self, question: str, history: list[dict]) -> tuple[str, list[str]]:
        """Ask the model to turn the question into a standalone question + a few search phrases.
        Any problem (no model, bad reply) just means: search with the question as asked."""
        if self.gateway is None or not self.gateway.has_model("answer"):
            return question, []
        convo = "\n".join(f"Q: {h['question']}\nA: {str(h.get('answer', ''))[:400]}" for h in history)
        user = (f"RECENT CONVERSATION:\n{convo}\n\n" if convo else "") + f"QUESTION: {question}"
        try:
            reply = self.gateway.complete_json("answer", _PLAN_SYSTEM, user, max_tokens=300, attempts=1)
        except (GatewayError, ValueError):
            return question, []
        if not isinstance(reply, dict):
            return question, []
        standalone = str(reply.get("standalone_question") or "").strip() or question
        searches = [str(x).strip() for x in (reply.get("searches") or []) if str(x).strip()][:4]
        return (standalone if len(standalone) < 500 else question), searches

    # ------------------------------------------------------------------ model answer
    def _model_answer(self, task: str, question: str, hits: list[Hit], result: AskResult, context=None):
        history, overview, standalone = context or ([], None, question)
        searched = {h.passage.page_no for h in hits}

        def header(h: Hit) -> str:
            page = self.pages_by_no.get(h.passage.page_no)
            kind = _PAGE_TYPE_LABEL.get(getattr(page, "page_type", "other") or "other", "record")
            return f"(p. {h.passage.page_no}, {kind})"
        passages = "\n\n".join(f"[{i}] {header(h)}\n{h.passage.text}" for i, h in enumerate(hits, 1))
        codes_here = sorted({m.code for m in self.mentions if m.page_no in searched})
        facts = "\n".join(f"[F{i}] {f['text']}" for i, f in enumerate(self.facts, 1)) or "none"
        convo = "\n".join(f"Q: {h['question']}\nA: {str(h.get('answer', ''))[:600]}" for h in history or [])
        q_line = question if standalone == question else f"{question}\n(Meaning: {standalone})"
        user = ((f"CASE OVERVIEW (background, don't cite):\n{overview[:4000]}\n\n" if overview else "")
                + (f"RECENT CONVERSATION (background, don't cite):\n{convo}\n\n" if convo else "")
                + f"QUESTION: {q_line}\n\nCASE FACTS (whole file):\n{facts}\n\nPASSAGES:\n{passages}\n\n"
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
        not_found = normalize(answer).startswith(normalize(NOT_FOUND)[:40])
        answer, sources, bad = _resolve_passage_markers(answer, hits, self.pages_by_no, self.facts)

        # Older reply style (page + quote) is still accepted, with forgiving quote matching.
        for c in reply.get("citations") or []:
            try:
                page, quote = int(c.get("page")), str(c.get("quote", "")).strip()
            except (TypeError, ValueError, AttributeError):
                bad += 1
                continue
            if page in searched and _quote_ok(page, quote, self.pages_by_no):
                sources.append(SourceRef(page=page, quote=quote))
            elif page in searched and (q := _closest_line(page, quote, self.pages_by_no)):
                sources.append(SourceRef(page=page, quote=q))
            else:
                bad += 1
        if bad:
            result.warnings.append(f"{bad} citation(s) from the {task} model pointed to text that was not "
                                   "in the search results and were removed.")

        named_pages = {int(n) for n in re.findall(r"\bp(?:age|g)?\.?\s*(\d{1,5})\b", answer, re.IGNORECASE)}
        bad_pages = sorted(named_pages - {s.page for s in sources} - searched)
        if not answer or bad_pages or (not sources and not not_found):
            if bad_pages:
                result.warnings.append(f"Answer referred to page(s) {bad_pages} that were not in the search results.")
            elif not not_found:
                result.warnings.append(f"The {task} model's answer had no usable source markers.")
            return None

        answer, span_notes = fix_time_spans(answer)
        result.warnings += span_notes
        cited_pages = {s.page for s in sources}
        cited_text = (" ".join(self.pages_by_no[p].text for p in cited_pages if p in self.pages_by_no) + " "
                      + " ".join(h.passage.text for h in hits) + " " + " ".join(f["text"] for f in self.facts))
        missing = _unsupported_values(answer, cited_text)
        if missing:
            result.warnings.append("Check these in the source pages; they were not found in the cited text: "
                                   + ", ".join(missing[:6]))
        model_name = self.gateway.settings.models.get(task)
        return answer, _dedupe(sources), [str(x) for x in reply.get("cpt_codes") or []], model_name

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


def quote_for(text: str, context: str, max_chars: int = 220) -> str:
    """The short stretch of a passage that best supports a sentence: lines that contain the
    sentence's own dates score highest, then shared words. Long lines are cut to a window."""
    want_dates = set(_dates(context))
    terms = set(tokenize(context))
    candidates = []
    for piece in re.split(r"\n|(?<=[.!?])\s+", text):
        words = piece.split()
        if len(" ".join(words)) <= max_chars:
            candidates.append(" ".join(words))
        else:                                   # long run-on OCR line: look at overlapping windows
            for start in range(0, max(1, len(words) - 8), 10):
                candidates.append(" ".join(words[start:start + 24]))
    candidates = [c for c in candidates if len(c) > 15] or [" ".join(text.split()[:30])]

    def score(c: str):
        return (5 * len(want_dates & set(_dates(c))) + len(terms & set(tokenize(c))), -abs(len(c) - 120))
    best = max(candidates, key=score)
    if len(best) > max_chars:
        best = best[:max_chars].rsplit(" ", 1)[0]
    return best


_NUM_WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}
_SPAN = re.compile(r"(?:(?:approximately|about|roughly|nearly|almost|over|just over|around)\s+)?"
                   r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(years?|months?|weeks?|days?)"
                   r"(?:,?\s*(?:and\s+)?(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
                   r"(months?|weeks?|days?))?", re.IGNORECASE)
_UNIT_DAYS = {"year": 365.25, "month": 30.44, "week": 7, "day": 1}


def _span_days(m: re.Match) -> float:
    total = 0.0
    for num, unit in ((m.group(1), m.group(2)), (m.group(3), m.group(4))):
        if num and unit:
            n = _NUM_WORDS.get(num.lower(), None)
            n = int(num) if n is None else n
            total += n * _UNIT_DAYS[unit.lower().rstrip("s")]
    return total


def _format_span(days: float) -> str:
    if days < 60:
        return f"about {round(days)} days"
    years, months = int(days // 365.25), round((days % 365.25) / 30.44)
    if months == 12:
        years, months = years + 1, 0
    parts = ([f"{years} year{'s' if years != 1 else ''}"] if years else []) + \
            ([f"{months} month{'s' if months != 1 else ''}"] if months else [])
    return "about " + " and ".join(parts)


def fix_time_spans(text: str) -> tuple[str, list[str]]:
    """Models are poor at date arithmetic. Where a sentence gives two or more dates and a length of
    time, recompute the length from the earliest and latest date and correct it if it's off."""
    import datetime as _dt
    notes, out = [], []
    for sentence in re.split(r"(?<=[.!?])\s+|(?<=\n)", text):
        dates = sorted(_dates(sentence))
        span = _SPAN.search(sentence)
        if len(dates) >= 2 and span:
            try:
                d1, d2 = _dt.date(*dates[0]), _dt.date(*dates[-1])
            except ValueError:
                out.append(sentence)
                continue
            actual, stated = (d2 - d1).days, _span_days(span)
            if actual > 0 and abs(actual - stated) > max(31, 0.1 * actual):
                fixed = _format_span(actual)
                notes.append(f"Corrected a time span: the model wrote \"{span.group(0).strip()}\", but "
                             f"{d1:%m/%d/%Y} to {d2:%m/%d/%Y} is {fixed}.")
                sentence = sentence[:span.start()] + fixed + sentence[span.end():]
        out.append(sentence)
    joined = ""
    for i, part in enumerate(out):                 # re-join with the spaces the split removed
        joined += part if (i == 0 or joined.endswith("\n")) else " " + part
    return joined, notes


def _best_line(text: str, terms: set[str], max_len: int = 300) -> str:
    candidates = [c.strip() for c in re.split(r"\n|(?<=[.!?])\s+", text) if len(c.strip()) > 15]
    if not candidates:
        return text.strip()[:max_len]
    best = max(candidates, key=lambda c: (len(terms & set(tokenize(c))), -len(c)))
    return best[:max_len]


def _merge_hits(results: list[list[Hit]], limit: int) -> list[Hit]:
    """Combine several searches: take the best of each in turn, without repeats."""
    out, seen = [], set()
    for rank in range(max((len(r) for r in results), default=0)):
        for res in results:
            if rank < len(res) and res[rank].passage.pid not in seen and len(out) < limit:
                seen.add(res[rank].passage.pid)
                out.append(res[rank])
    return out


_MARKER = re.compile(r"\[(F?\d{1,2}(?:\s*[,;-]\s*F?\d{1,2})*)\]", re.IGNORECASE)


def _resolve_passage_markers(answer: str, hits: list[Hit], pages_by_no: dict[int, Page], facts=()):
    """Turn [n] markers into (p. N) and pick each source quote from the page itself.

    Returns (answer with page numbers, sources, number of invalid markers)."""
    sources: list[SourceRef] = []
    bad = 0
    sentences = re.split(r"(?<=[.!?])\s+|\n", answer)

    def numbers(group: str) -> list:
        out = []
        for part in re.split(r"\s*[,;]\s*", group.upper()):
            fact = part.startswith("F")
            part = part.replace("F", "")
            if "-" in part:
                a, b = (int(x) for x in part.split("-", 1))
                nums = list(range(a, b + 1)) if 0 < b - a < 10 else [a, b]
            else:
                nums = [int(part)] if part.strip() else []
            out += [("F", n) if fact else n for n in nums]
        return out

    def repl(m: re.Match) -> str:
        nonlocal bad
        pages = []
        for n in numbers(m.group(1)):
            if isinstance(n, tuple):                       # [F3]: a case fact from the bills
                if not 1 <= n[1] <= len(facts):
                    bad += 1
                    continue
                for page, quote in facts[n[1] - 1]["sources"]:
                    if quote_on_page(Citation(page_no=page, quote=quote), pages_by_no):
                        sources.append(SourceRef(page=page, quote=quote))
                    if page not in pages:
                        pages.append(page)
                continue
            if not 1 <= n <= len(hits):
                bad += 1
                continue
            passage = hits[n - 1].passage
            context = next((s for s in sentences if m.group(0) in s), answer)
            quote = quote_for(passage.text, _MARKER.sub(" ", context))
            if quote and quote_on_page(Citation(page_no=passage.page_no, quote=quote), pages_by_no):
                sources.append(SourceRef(page=passage.page_no, quote=quote))
            if passage.page_no not in pages:
                pages.append(passage.page_no)
        return "".join(f" (p. {p})" for p in pages)

    text = _MARKER.sub(repl, answer)
    text = re.sub(r"\s+(\(p\. \d+\))", r" \1", text)
    text = re.sub(r"(\(p\. (\d+)\))(\s*\(p\. \2\))+", r"\1", text)     # same page twice in a row
    text = re.sub(r"[ \t]+([.,;:])", r"\1", text).strip()
    return text, sources, bad


def _months(d1: str, d2: str) -> str:
    y1, m1, dd1 = (int(x) for x in d1.split("-"))
    y2, m2, dd2 = (int(x) for x in d2.split("-"))
    days = (y2 - y1) * 365.25 + (m2 - m1) * 30.44 + (dd2 - dd1)
    return f"{days / 30.44:.0f} months" if days >= 60 else f"{days:.0f} days"


def case_facts(services: list[ServiceRow], details: dict) -> list[dict]:
    """A short fact sheet computed from the whole file's bills (and the date of injury).
    Each fact keeps the pages and verified quotes it rests on, so the model can cite it as [F1]."""
    from .cpt import category
    facts: list[dict] = []
    doi = details.get("date_of_injury") if isinstance(details, dict) else None
    if doi and doi.get("value") and doi.get("citation"):
        c = doi["citation"]
        facts.append({"text": f"Date of injury: {doi['value']}", "sources": [(c["page_no"], c["quote"])]})
    rows = sorted((s for s in services if s.date_of_service), key=lambda s: s.date_of_service)
    if not rows:
        return facts
    first, last = rows[0], rows[-1]
    total = sum(s.billed_amount or 0 for s in rows)
    dates = {s.date_of_service for s in rows}
    facts.append({"text": f"Billed treatment dates: first {first.date_of_service} (CPT {first.cpt}), most recent "
                          f"{last.date_of_service} (CPT {last.cpt}); a span of about "
                          f"{_months(first.date_of_service, last.date_of_service)}; {len(dates)} dates of service, "
                          f"{len(rows)} billed lines, ${total:,.2f} billed in total",
                  "sources": [(first.citation.page_no, first.citation.quote), (last.citation.page_no, last.citation.quote)]})
    groups: dict[str, list[ServiceRow]] = {}
    for s in rows:
        groups.setdefault(category(s.cpt)[0], []).append(s)
    for label, g in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:10]:
        codes = sorted({s.cpt for s in g})
        g_dates = sorted({s.date_of_service for s in g})
        facts.append({"text": f"{label} (CPT {', '.join(codes[:6])}): {len(g_dates)} dates of service, {len(g)} lines, "
                              f"{sum(s.units for s in g)} units, ${sum(s.billed_amount or 0 for s in g):,.2f} billed, "
                              f"from {g[0].date_of_service} to {g[-1].date_of_service}",
                      "sources": [(g[0].citation.page_no, g[0].citation.quote),
                                  (g[-1].citation.page_no, g[-1].citation.quote)]})
    providers: dict[str, list[ServiceRow]] = {}
    for s in rows:
        if s.provider:
            providers.setdefault(s.provider.strip(), []).append(s)
    for name, g in sorted(providers.items(), key=lambda kv: -len(kv[1]))[:6]:
        facts.append({"text": f"Billing provider {name}: {len(g)} lines from {g[0].date_of_service} to "
                              f"{g[-1].date_of_service}",
                      "sources": [(g[0].citation.page_no, g[0].citation.quote)]})
    return facts


def _dedupe(sources: list[SourceRef]) -> list[SourceRef]:
    seen, out = set(), []
    for s in sources:
        key = (s.page, normalize(s.quote))
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
                                       "nov", "dec"], 1)}
_DATE_NUM = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b|\b(\d{4})-(\d{2})-(\d{2})\b")
_DATE_TXT = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})\b",
                       re.IGNORECASE)
_MONEY = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d+(?:\.\d{2})?)")
_AMOUNT = re.compile(r"\b(\d{1,3}(?:,\d{3})+(?:\.\d{2})?|\d+\.\d{2})\b")


def _dates(text: str) -> dict[tuple, str]:
    out = {}
    for m in _DATE_NUM.finditer(text):
        if m.group(4):
            y, mo, d = int(m.group(4)), int(m.group(5)), int(m.group(6))
        else:
            mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            y += 2000 if y < 100 else 0
        out[(y, mo, d)] = m.group(0)
    for m in _DATE_TXT.finditer(text):
        out[(int(m.group(3)), _MONTHS[m.group(1).lower()[:3]], int(m.group(2)))] = m.group(0)
    return out


def _amounts(text: str, money_only: bool) -> dict[float, str]:
    out = {}
    for rx in ((_MONEY,) if money_only else (_MONEY, _AMOUNT)):
        for m in rx.finditer(text):
            try:
                out[round(float(m.group(1).replace(",", "")), 2)] = m.group(0)
            except ValueError:
                pass
    return out


def _unsupported_values(answer: str, source_text: str) -> list[str]:
    """Dates, dollar amounts and CPT codes in the answer that don't appear in the source passages."""
    missing = []
    src_dates = _dates(source_text)
    missing += [shown for key, shown in _dates(answer).items() if key not in src_dates]
    src_amounts = _amounts(source_text, money_only=False)
    missing += [shown for value, shown in _amounts(answer, money_only=True).items()
                if value not in src_amounts and value != 0]
    src_norm = normalize(source_text)
    for code in set(re.findall(r"\b(\d{4}[0-9FTU]|[A-V]\d{4})\b", answer)):
        if code.lower() not in src_norm:
            missing.append(code)
    return missing


def _closest_line(page: int, quote: str, pages_by_no: dict[int, Page]) -> str | None:
    """For a slightly misquoted line (OCR typos, spacing): the page line that best matches it."""
    p = pages_by_no.get(page)
    words = set(tokenize(quote))
    if p is None or len(words) < 3:
        return None
    lines = [ln.strip() for ln in p.text.splitlines() if len(ln.strip()) > 10]
    if not lines:
        return None
    best = max(lines, key=lambda ln: len(words & set(tokenize(ln))))
    return best[:300] if len(words & set(tokenize(best))) >= max(3, int(0.6 * len(words))) else None


def _quote_ok(page: int, quote: str, pages_by_no: dict[int, Page]) -> bool:
    """Exact match after whitespace/quote normalization; quotes with '...' are checked piece by piece."""
    if len(quote) < 8:
        return False
    parts = [p.strip() for p in re.split(r"\.\.\.|…", quote) if len(p.strip()) >= 8]
    return bool(parts) and all(quote_on_page(Citation(page_no=page, quote=p), pages_by_no) for p in parts)
