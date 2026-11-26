"""Citation verifier: cite or omit.

A row survives only if its quote is found on the page it cites, and the key
values of the row (code, date, amount) appear inside that quote. This blocks
the two failure modes that hurt in a deposition: a made-up source, and a real
source attached to a made-up number.
"""

from __future__ import annotations

import re

from ..schemas import Citation, Page

_TRANSLATE = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", " ": " ",
})


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(_TRANSLATE)).strip().casefold()


def quote_on_page(citation: Citation, pages_by_no: dict[int, Page]) -> bool:
    page = pages_by_no.get(citation.page_no)
    quote = normalize(citation.quote)
    if page is None or len(quote) < 4:
        return False
    return quote in normalize(page.text)


def values_in_quote(quote: str, *values: object) -> bool:
    """Every non-empty value must appear in the quote (amounts compared as digits)."""
    q = normalize(quote)
    q_digits = re.sub(r"[^0-9]", "", q)
    for value in values:
        if value is None or value == "":
            continue
        if isinstance(value, float):
            digits = re.sub(r"[^0-9]", "", f"{value:.2f}")
            if digits not in q_digits:
                return False
        elif normalize(str(value)) not in q:
            return False
    return True


def us_date_variants(iso_date: str) -> list[str]:
    """2025-03-20 -> ['03/20/2025', '03/20/25', '3/20/2025']"""
    y, m, d = iso_date.split("-")
    return [f"{m}/{d}/{y}", f"{m}/{d}/{y[2:]}", f"{int(m)}/{int(d)}/{y}"]


def date_in_quote(quote: str, iso_date: str) -> bool:
    q = normalize(quote)
    return any(v in q for v in us_date_variants(iso_date))
