"""Finding CPT/HCPCS codes in a case file and deciding which ones a question is about.

A 5-digit number is not automatically a CPT code: ZIP codes (WI 53783),
amounts, claim and account numbers look the same. A code is accepted when:
  - it appears on a bill line already extracted into the ledger, or
  - its line has code context ("CPT", "HCPCS", "procedure code", a modifier
    like -LT / -GP, or "x4" units), and the number isn't a ZIP code, phone,
    date or money amount.

Category labels below are our own short descriptions of standard CPT code
ranges (not AMA descriptor text, which requires an AMA license).
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from ..schemas import Page, ServiceRow

_CODE = re.compile(r"(?<![\w$.,/-])([0-9]{4}[0-9FTU]|[A-V][0-9]{4})(?![\w/]|[.,]\d)(?:-([A-Z0-9]{2}))?")
_CONTEXT = re.compile(r"\bcpt\b|\bhcpcs\b|procedure\s*code|proc\.?\s*code|\bcode[sd]?\b|\bbilled\b|"
                      r"\bunits?\b|\bx\s?\d\b|\bmod(?:ifier)?\b|\bcharge", re.IGNORECASE)
_ZIP_BEFORE = re.compile(r"\b(?:[A-Z]{2}|zip|zip code|suite|ste|box)\s*$", re.IGNORECASE)
_PHONE = re.compile(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}")

# (low, high, label, question keywords) - checked in order, specific ranges first.
_CATEGORIES: list[tuple[str, str, str, tuple[str, ...]]] = [
    ("62320", "62327", "Epidural injection", ("epidural", "esi", "injection", "steroid")),
    ("64479", "64484", "Epidural injection (transforaminal)", ("epidural", "esi", "transforaminal", "injection")),
    ("64490", "64495", "Facet joint injection", ("facet", "injection", "medial branch")),
    ("64633", "64636", "Facet nerve ablation (RFA)", ("ablation", "rfa", "radiofrequency", "facet")),
    ("20550", "20553", "Trigger point / tendon injection", ("trigger point", "injection")),
    ("20600", "20611", "Joint injection or aspiration", ("joint injection", "injection", "aspiration")),
    ("97010", "97799", "Physical medicine / therapy", ("physical therapy", "therapy", "pt", "rehab",
                                                       "exercise", "manual therapy", "modalit")),
    ("98940", "98943", "Chiropractic manipulation", ("chiropractic", "chiro", "manipulation", "adjustment")),
    ("99202", "99499", "Office / hospital visit (E/M)", ("visit", "office", "consult", "evaluation",
                                                         "follow-up", "follow up", "appointment")),
    ("95860", "95913", "Nerve testing (EMG / NCS)", ("emg", "nerve conduction", "electromyography", "ncs")),
    ("70010", "79999", "Radiology / imaging", ("mri", "x-ray", "xray", "ct ", "imaging", "radiolog", "scan")),
    ("80047", "89398", "Laboratory / pathology", ("lab", "urine", "drug screen", "blood test", "toxicology")),
    ("90785", "90899", "Psychiatry / psychotherapy", ("psych", "counsel", "therapy session")),
    ("96360", "96549", "Infusion / injection (drug administration)", ("infusion", "injection")),
    ("10004", "69990", "Surgery / procedure", ("surgery", "surgical", "procedure", "operative", "fusion")),
    ("00100", "01999", "Anesthesia", ("anesthesia", "sedation")),
]
_HCPCS = {"J": ("Drug (HCPCS J-code)", ("drug", "medication", "injection")),
          "E": ("Durable medical equipment", ("equipment", "dme", "brace", "tens")),
          "L": ("Orthotics / prosthetics", ("brace", "orthotic", "prosthe")),
          "A": ("Supplies / transport (HCPCS)", ("supplies", "ambulance", "transport")),
          "G": ("HCPCS G-code service", ()), "Q": ("HCPCS Q-code", ()), "S": ("HCPCS S-code", ())}


def category(code: str) -> tuple[str, tuple[str, ...]]:
    if code[0].isalpha():
        return _HCPCS.get(code[0], ("HCPCS code", ()))
    if code[-1] in "FTU":
        return ("CPT category II/III code", ())
    for low, high, label, words in _CATEGORIES:
        if low <= code <= high:
            return label, words
    return ("Other CPT code", ())


@dataclass
class CodeMention:
    code: str
    page_no: int
    line: str


@dataclass
class CodeSummary:
    code: str
    category: str
    pages: list[int] = field(default_factory=list)          # pages mentioning the code
    billed_lines: int = 0
    units: int = 0
    billed_total: float = 0.0
    first_date: str | None = None
    last_date: str | None = None
    why: str = ""                                            # how it relates to the question


def find_code_mentions(pages: list[Page], ledger_codes: set[str]) -> list[CodeMention]:
    out = []
    for page in pages:
        if page.duplicate_of is not None:
            continue
        for line in page.text.splitlines():
            context = bool(_CONTEXT.search(line))
            for m in _CODE.finditer(line):
                code = m.group(1)
                before = line[:m.start()]
                if code in ledger_codes:
                    out.append(CodeMention(code, page.page_no, line.strip()))
                    continue
                if not (context or m.group(2)):                    # no code context at all
                    continue
                if _ZIP_BEFORE.search(before) or _PHONE.search(line[max(0, m.start() - 6):m.end() + 6]):
                    continue
                if re.match(r"\s*/\d", line[m.end():]) or re.search(r"\$\s*$", before):
                    continue                                       # part of a date or money amount
                out.append(CodeMention(code, page.page_no, line.strip()))
    return out


def summarize(codes: set[str], mentions: list[CodeMention], services: list[ServiceRow],
              why: dict[str, str]) -> list[CodeSummary]:
    pages_by_code: dict[str, set[int]] = defaultdict(set)
    for m in mentions:
        if m.code in codes:
            pages_by_code[m.code].add(m.page_no)
    out = []
    for code in sorted(codes):
        s = CodeSummary(code=code, category=category(code)[0], why=why.get(code, ""))
        rows = [r for r in services if r.cpt == code]
        for r in rows:
            pages_by_code[code].add(r.citation.page_no)
            s.billed_lines += 1
            s.units += r.units
            s.billed_total = round(s.billed_total + (r.billed_amount or 0), 2)
        dates = sorted(r.date_of_service for r in rows)
        s.first_date, s.last_date = (dates[0], dates[-1]) if dates else (None, None)
        s.pages = sorted(pages_by_code[code])
        out.append(s)
    return out


_NOT_BILLING = re.compile(r"\b(diagnos\w*|icd|dx|history|symptoms?|complaints?|restrictions?|work status|"
                          r"medications?|allerg\w*|employer|insurer|claim number|date of (?:birth|injury))\b")


def codes_for_question(question: str, retrieved_pages: set[int], mentions: list[CodeMention],
                       services: list[ServiceRow]) -> dict[str, str]:
    """Codes the question is about -> reason. Only codes that exist in the document qualify."""
    in_doc = {m.code for m in mentions} | {s.cpt for s in services}
    q = question.lower()
    why: dict[str, str] = {}

    for code in re.findall(r"\b([0-9]{4}[0-9FTU]|[A-V][0-9]{4})\b", question.upper()):
        if code in in_doc:
            why[code] = "named in the question"

    asks_all = re.search(r"\ball\b.*\b(cpt|codes?|billed|services|procedures)\b|\bevery\b.*\bcodes?\b", q)
    for code in in_doc:
        if code in why:
            continue
        label, words = category(code)
        if asks_all:
            why[code] = "all codes requested"
        elif any(w in q for w in words):
            why[code] = f"question is about {label.lower()}"

    if why:          # the question named a topic or codes: don't pad with every code on the same bill page
        return why
    if _NOT_BILLING.search(q):   # diagnoses, history, restrictions...: billing codes aren't the answer
        return why
    for m in mentions:
        if m.page_no in retrieved_pages and m.code not in why:
            why[m.code] = f"appears on source page {m.page_no}"
    return why
