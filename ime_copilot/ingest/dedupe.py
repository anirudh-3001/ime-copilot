"""Near-duplicate page detection (the same note faxed three times counts once)."""

from __future__ import annotations

import hashlib
import re

from ..schemas import Page

SIMILARITY_THRESHOLD = 0.90
SHINGLE = 5


_STAMP_LINE = re.compile(r"\b(fax|received|sent|transmitted)\b|page \d+ of \d+", re.IGNORECASE)
_DATE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")


def _strip_stamps(text: str) -> str:
    """Fax headers and page stamps differ between copies of the same page; drop those lines."""
    return "\n".join(l for l in text.splitlines() if not (len(l) < 120 and _STAMP_LINE.search(l)))


def _normalize(text: str) -> str:
    text = re.sub(r"[^a-z0-9 ]+", " ", _strip_stamps(text).lower())
    return re.sub(r"\s+", " ", text).strip()


def _dates(text: str) -> frozenset[str]:
    return frozenset(_DATE.findall(_strip_stamps(text)))


def _shingles(text: str) -> set[str]:
    words = text.split()
    if len(words) < SHINGLE:
        return {text} if text else set()
    return {" ".join(words[i:i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def mark_duplicates(pages: list[Page]) -> list[Page]:
    """Set page.duplicate_of to the first page each near-duplicate copies."""
    seen_hash: dict[str, int] = {}
    kept: list[tuple[int, int, set[str], frozenset[str]]] = []   # (page_no, length, shingles, dates)

    for page in pages:
        norm = _normalize(page.text)
        if len(norm) < 40:                          # blank or near-blank pages
            continue
        digest = hashlib.sha256(norm.encode()).hexdigest()
        if digest in seen_hash:
            page.duplicate_of = seen_hash[digest]
            continue
        sh, dates = _shingles(norm), _dates(page.text)
        for page_no, length, other, other_dates in kept:
            # Templated notes (PT, chiro) are often identical except the visit date:
            # different dates mean different visits, never duplicates.
            if dates != other_dates:
                continue
            if abs(length - len(norm)) > 0.15 * max(length, len(norm)):
                continue
            if jaccard(sh, other) >= SIMILARITY_THRESHOLD:
                page.duplicate_of = page_no
                break
        if page.duplicate_of is None:
            seen_hash[digest] = page.page_no
            kept.append((page.page_no, len(norm), sh, dates))
    return pages
