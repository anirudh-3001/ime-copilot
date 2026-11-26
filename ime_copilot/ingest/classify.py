"""Page classification: what kind of document is each page?

Two implementations with the same output:
  rules - keyword scoring, fully offline; the baseline every model must beat
  llm   - a small model through the gateway (Haiku-class today; our own
          fine-tuned classifier later)
"""

from __future__ import annotations

import re

from ..gateway import Gateway
from ..schemas import PAGE_TYPES, Page

# (pattern, weight) per page type. Patterns are matched case-insensitively.
_RULES: dict[str, list[tuple[str, float]]] = {
    "bill": [
        (r"health insurance claim form", 5), (r"cms[- ]?1500", 4), (r"\bub[- ]?04\b", 4),
        (r"itemized statement", 4), (r"total charges?", 2), (r"\bcpt\b|\bhcpcs\b", 1.5),
        (r"date\(s\) of service|dates? of service", 1.5), (r"amount due|balance due", 1.5),
    ],
    "referral_letter": [
        (r"independent medical (examination|evaluation)", 3), (r"please (address|answer|respond)", 3),
        (r"questions?\b", 1), (r"\bime\b", 1.5), (r"claim (no|number|#)", 1), (r"re:\s", 1),
    ],
    "operative_report": [
        (r"operative report|procedure note|procedure report", 5), (r"pre-?operative diagnosis", 3),
        (r"epidural|transforaminal|facet|radiofrequency|fluoroscop", 2), (r"anesthesia", 1),
    ],
    "pt_note": [
        (r"physical therapy|physiotherapy|chiropractic", 4), (r"daily note|treatment note", 2),
        (r"therapeutic exercise|manual therapy|97110|97140|98941", 2), (r"visit #?\d+", 1),
    ],
    "imaging_report": [
        (r"\bmri\b|\bct\b scan|x-?ray|radiograph", 3), (r"impression:", 2), (r"technique:", 2),
        (r"radiolog", 2), (r"findings:", 1),
    ],
    "office_note": [
        (r"\bsubjective\b|chief complaint|history of present illness", 3), (r"\bassessment\b", 2),
        (r"\bplan\b", 1), (r"physical exam|examination:", 2), (r"\bobjective\b", 1),
    ],
    "prior_ime": [
        (r"independent medical examination report", 5), (r"maximum medical improvement", 2),
        (r"records reviewed", 2), (r"impairment rating", 2),
    ],
    "legal": [
        (r"deposition|subpoena|attorney at law|law offices?|plaintiff|defendant", 3),
        (r"court|docket", 2),
    ],
}


def classify_rules(page: Page) -> tuple[str, float]:
    text = page.text.lower()
    if not text.strip():
        return "other", 0.0
    scores = {
        ptype: sum(w for pat, w in rules if re.search(pat, text))
        for ptype, rules in _RULES.items()
    }
    best = max(scores, key=scores.get)
    top = scores[best]
    if top < 2:
        return "other", 0.3
    second = sorted(scores.values(), reverse=True)[1]
    confidence = min(0.99, 0.5 + 0.5 * (top - second) / top)
    return best, round(confidence, 2)


_SYSTEM = (
    "You classify single pages from workers' compensation and auto injury claim files. "
    "Reply with JSON only: {\"page_type\": <one of " + ", ".join(PAGE_TYPES) + ">, "
    "\"confidence\": <0 to 1>}."
)


def classify_llm(page: Page, gateway: Gateway) -> tuple[str, float]:
    reply = gateway.complete_json("classify", _SYSTEM, page.text[:6000], max_tokens=100)
    ptype = str(reply.get("page_type", "other"))
    if ptype not in PAGE_TYPES:
        ptype = "other"
    return ptype, float(reply.get("confidence", 0.5))


def classify_pages(pages: list[Page], mode: str = "rules", gateway: Gateway | None = None) -> list[Page]:
    for page in pages:
        if page.duplicate_of is not None:
            continue  # inherits the type of its original below
        if mode == "llm" and gateway is not None and gateway.enabled:
            page.page_type, page.type_confidence = classify_llm(page, gateway)
        else:
            page.page_type, page.type_confidence = classify_rules(page)
    by_no = {p.page_no: p for p in pages}
    for page in pages:
        if page.duplicate_of is not None:
            original = by_no[page.duplicate_of]
            page.page_type, page.type_confidence = original.page_type, original.type_confidence
    return pages
