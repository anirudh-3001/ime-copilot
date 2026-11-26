"""Case questions from the IME company's referral letter, kept verbatim and in order.

The signed report must answer every question in the original numbering
(1, 2, 3a, 3b ...), so numbering and wording are captured exactly.
"""

from __future__ import annotations

import re

from ..gateway import Gateway, GatewayError
from ..schemas import QUESTION_CATEGORIES, CaseQuestion, Citation, Page
from .citations import normalize

_ITEM = re.compile(r"^\s*(?:Q(?:uestion)?\s*)?\(?(?P<num>\d{1,2}[a-z]?|[a-z])[\.\)]\s+(?P<text>\S.*)$")
_STOP = re.compile(r"^\s*(sincerely|thank you|regards|respectfully|cc:)", re.IGNORECASE)
_INSTRUCTION = re.compile(
    r"[^.?]*(?:AMA Guides[^.?]*|address only[^.?]*|\b\d+(?:st|nd|rd|th) Edition[^.?]*)[.?]?",
    re.IGNORECASE,
)

# Checked in order: the first matching category wins.
_CATEGORY_RULES: list[tuple[str, str]] = [
    ("apportionment", r"apportion"),
    ("impairment", r"impairment|ama guides|whole person"),
    ("mmi", r"maximum medical improvement|\bmmi\b"),
    ("work_status", r"return to work|work status|light duty|modified duty"),
    ("requested_services", r"\brequested\b|\bproposed\b|\brecommended\b"),
    ("treatment_to_date", r"treatment (rendered|to date|provided|received)|to date\b.*(reasonable|necessary)"),
    ("diagnosis", r"\bwhat (is|are) the diagnos|diagnos[ie]s (related|attributable)"),
    ("causation", r"causal|caused|related to the (work|motor|accident|injury)|mechanism of injury"),
    ("future_care", r"future|further (treatment|care)"),
    ("diagnosis", r"diagnos"),
]


def categorize(text: str) -> str:
    low = text.lower()
    for category, pattern in _CATEGORY_RULES:
        if re.search(pattern, low):
            return category
    return "other"


def instructions_in(text: str) -> list[str]:
    return [m.group(0).strip() for m in _INSTRUCTION.finditer(text) if m.group(0).strip()]


def parse_questions_rules(letter_pages: list[Page]) -> list[CaseQuestion]:
    questions: list[CaseQuestion] = []
    current: dict | None = None
    last_parent: str | None = None

    def close():
        if current:
            text = " ".join(current["lines"]).strip()
            questions.append(CaseQuestion(
                number=current["num"], text=text, category=categorize(text),
                instructions=instructions_in(text),
                citation=Citation(page_no=current["page"], quote=text), method="rules",
            ))

    for page in letter_pages:
        for line in page.text.splitlines():
            if _STOP.match(line):
                close()
                current = None
                continue
            m = _ITEM.match(line)
            if m:
                close()
                num = m.group("num")
                if num.isalpha() and last_parent:       # "a)" under question 3 -> "3a"
                    num = f"{last_parent}{num}"
                elif num[0].isdigit():
                    last_parent = re.match(r"\d+", num).group(0)
                current = {"num": num, "lines": [m.group("text").strip()], "page": page.page_no}
            elif current and line.strip():
                current["lines"].append(line.strip())
            elif current and not line.strip():
                close()
                current = None
    close()
    _inherit_parent_category(questions)
    return questions


def _inherit_parent_category(questions: list[CaseQuestion]) -> None:
    """'3a) Physical therapy' has no keywords of its own; it belongs to question 3's category."""
    by_number = {q.number: q for q in questions}
    for q in questions:
        parent = re.match(r"\d+", q.number)
        if q.category == "other" and parent and parent.group(0) != q.number:
            q.category = by_number.get(parent.group(0), q).category


_SYSTEM = """You read a referral letter sent to an independent medical examiner and list the questions the examiner must answer.
Return JSON only: a list of objects with keys
  number (exactly as numbered in the letter, sub-parts joined to their parent, e.g. "3b"),
  text (the question copied EXACTLY, word for word, without the number),
  category (one of: """ + ", ".join(QUESTION_CATEGORIES) + """),
  instructions (list of instructions inside the question, e.g. "use the AMA Guides 6th Edition"; [] if none),
  page_no (the page number the question appears on).
Keep the original order. Do not merge, split, rephrase or add questions."""


def parse_questions_llm(letter_pages: list[Page], gateway: Gateway) -> list[CaseQuestion]:
    body = "\n\n".join(f"PAGE {p.page_no}:\n{p.text}" for p in letter_pages)
    reply = gateway.complete_json("questions", _SYSTEM, body, max_tokens=4000)
    out = []
    for item in reply if isinstance(reply, list) else []:
        try:
            category = item.get("category") if item.get("category") in QUESTION_CATEGORIES else categorize(item["text"])
            out.append(CaseQuestion(
                number=str(item["number"]), text=str(item["text"]).strip(), category=category,
                instructions=[str(x) for x in item.get("instructions") or []],
                citation=Citation(page_no=int(item.get("page_no") or letter_pages[0].page_no),
                                  quote=str(item["text"]).strip()),
                method="llm",
            ))
        except (KeyError, ValueError, TypeError):
            continue
    return out


def verify_question(q: CaseQuestion, letter_pages: list[Page]) -> bool:
    """Verbatim check across the letter (questions may wrap onto the next page)."""
    letter = normalize(" ".join(p.text for p in letter_pages))
    return len(q.text) > 3 and normalize(q.text) in letter


def parse_questions(letter_pages: list[Page], mode: str, gateway: Gateway | None) -> tuple[list[CaseQuestion], int]:
    if not letter_pages:
        return [], 0
    qs: list[CaseQuestion] | None = None
    if mode == "llm" and gateway is not None and gateway.enabled:
        try:
            qs = parse_questions_llm(letter_pages, gateway)
        except (GatewayError, ValueError):
            qs = None
    if qs is None:
        qs = parse_questions_rules(letter_pages)
    kept = [q for q in qs if verify_question(q, letter_pages)]
    return kept, len(qs) - len(kept)
