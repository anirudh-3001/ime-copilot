"""Diagnoses (ICD-10-CM) with the first page and date they appear on.

Stage 1 uses the rules extractor in both modes; an LLM extractor for narrative
diagnoses ("disc herniation at L5-S1") comes in the next stage.
"""

from __future__ import annotations

import re

from ..schemas import Citation, DiagnosisRow, Page
from .services import to_iso

# ICD-10-CM: letter (not U), digit, digit/A/B, optional .xxxx
_ICD = re.compile(r"\b([A-TV-Z][0-9][0-9AB](?:\.[0-9A-Z]{1,4})?)\b")
_CONTEXT = re.compile(r"diagnos|\bdx\b|icd|assessment|impression", re.IGNORECASE)
_PAGE_DATE = re.compile(
    r"(?:date of (?:service|visit|exam|procedure)|dos|visit date|exam date)\s*[:#]?\s*(\d{1,2}/\d{1,2}/\d{2,4})",
    re.IGNORECASE,
)
CLINICAL_TYPES = {"office_note", "operative_report", "imaging_report", "pt_note", "bill", "prior_ime"}


def _page_date(page: Page) -> str | None:
    m = _PAGE_DATE.search(page.text)
    try:
        return to_iso(m.group(1)) if m else None
    except ValueError:
        return None


def extract_diagnoses(pages: list[Page]) -> list[DiagnosisRow]:
    found: dict[str, DiagnosisRow] = {}
    for page in pages:
        if page.page_type not in CLINICAL_TYPES:
            continue
        page_date = _page_date(page)
        in_dx_block = False
        for line in page.text.splitlines():
            if _CONTEXT.search(line):
                in_dx_block = True
            elif not line.strip():
                in_dx_block = False
            for m in _ICD.finditer(line):
                code = m.group(1)
                # A dot or a diagnosis context guards against matches like room numbers.
                if "." not in code and not in_dx_block:
                    continue
                desc = line[m.end():].strip(" -:\t") or None
                row = DiagnosisRow(
                    code=code, description=desc, first_seen_date=page_date,
                    citation=Citation(page_no=page.page_no, quote=line.strip()),
                    confidence=0.9 if "." in code else 0.7,
                )
                prev = found.get(code)
                if prev is None or _earlier(row, prev):
                    found[code] = row
    return sorted(found.values(), key=lambda d: (d.first_seen_date or "9999", d.citation.page_no))


def _earlier(a: DiagnosisRow, b: DiagnosisRow) -> bool:
    if a.first_seen_date and b.first_seen_date and a.first_seen_date != b.first_seen_date:
        return a.first_seen_date < b.first_seen_date
    if a.first_seen_date and not b.first_seen_date:
        return True
    return a.citation.page_no < b.citation.page_no and not (b.first_seen_date and not a.first_seen_date)
