"""Case header details: who the claimant is and what the claim is.

Pulls labelled fields that appear on almost every page of a claim file
(patient/claimant name, date of birth, claim number, date of injury, employer,
insurer). Each field is voted on across pages: the value seen most often wins,
and every value keeps its citation (page + exact line) plus how many pages agree.

Social Security numbers are never stored in full: only the last 4 digits.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from ..schemas import CaseDetails, Citation, FieldValue, Page
from .services import to_iso

_DATE = r"(\d{1,2}[/-]\d{1,2}[/-](?:\d{4}|\d{2}))"
_STOP = r"(?=\s{2,}|\s*\(|\s+(?:DOB|D\.O\.B|Date of|Claim|Policy|Sex|Gender|MRN|Account|Acct|Age|SSN|Employer|Insurer|Carrier|Phone)\b|[|;]|$)"

# field -> list of regexes; group(1) is the value
_PATTERNS: dict[str, list[str]] = {
    "patient_name": [
        rf"\b(?:patient(?:'s)?\s*name|patient|claimant(?:'s)?\s*name|claimant|employee\s*name|injured\s*worker)\s*[:#]\s*([A-Za-z][A-Za-z.,'\- ]{{2,60}}?){_STOP}",
        r"\b(?:RE|Re)\s*:\s*(?!Independent)([A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){1,3})\b",
    ],
    "date_of_birth": [rf"\b(?:DOB|D\.O\.B\.?|date\s*of\s*birth|birth\s*date)\s*[:#]?\s*{_DATE}"],
    "claim_number": [
        r"\b(?:claim|policy)\s*(?:no\.?|number|num|#)?\s*(?:/\s*policy\s*no\.?)?\s*[:#]\s*([A-Z0-9][A-Z0-9\-]{3,24})\b",
    ],
    "date_of_injury": [
        rf"\b(?:date\s*of\s*(?:injury|loss|accident|incident)|DOI|DOL|injury\s*date|accident\s*date)\s*[:#]?\s*{_DATE}",
        rf"\b(?:motor\s*vehicle\s*accident|accident|injury)\s+on\s+{_DATE}",
    ],
    "employer": [rf"\bemployer\s*(?:name)?\s*[:#]\s*([A-Za-z0-9][A-Za-z0-9&.,'\- ]{{2,60}}?){_STOP}"],
    "insurer": [rf"\b(?:insurer|insurance\s*carrier|carrier|insurance\s*company)\s*[:#]\s*([A-Za-z0-9][A-Za-z0-9&.,'\- ]{{2,60}}?){_STOP}"],
    "ssn_last4": [r"\b(?:SSN|SS#|social\s*security(?:\s*(?:no|number))?)\s*[:#]?\s*(?:\d{3}|X{3}|\*{3})[- ]?(?:\d{2}|X{2}|\*{2})[- ]?(\d{4})\b"],
}
_COMPILED = {k: [re.compile(p, re.IGNORECASE) for p in v] for k, v in _PATTERNS.items()}
_DATE_FIELDS = {"date_of_birth", "date_of_injury"}
_NOT_NAMES = {"name", "patient", "claimant", "unknown", "n/a", "na", "self", "same"}
_LABEL_WORDS = {"patient", "name", "claimant", "employee", "insured", "member", "examinee", "injured",
                "worker", "first", "last", "middle", "date", "birth", "address", "signature", "print"}
_STOPWORDS = {"of", "the", "and", "for", "to", "in", "on", "at", "by", "is", "my", "your", "with"}
_PARTICLES = {"de", "da", "van", "von", "del", "la", "le", "di", "du", "st", "mc", "mac"}
_FREE_TEXT = {"patient_name", "employer", "insurer"}
MIN_OCR_CONF_FREE_TEXT = 0.80      # names/companies from worse OCR are too often garbage


def _plausible_name(value: str) -> bool:
    tokens = value.replace(",", " ").split()
    if len(tokens) < 2 or len(tokens) > 5:
        return False
    if all(t.lower().strip(".") in _LABEL_WORDS for t in tokens):          # "Patient Name"
        return False
    for t in tokens:
        bare = t.strip(".'-")
        if not bare.isalpha():
            return False
        if bare.lower() in _STOPWORDS:                                     # "MOY OF eg"
            return False
        if bare.islower() and bare not in _PARTICLES:                      # names are capitalized
            return False
    return True


def _plausible_org(value: str) -> bool:
    """Reject OCR fragments like 'th Wl CHU Chin ee': mostly 1-2 letter pieces."""
    tokens = value.split()
    if not tokens or not any(c.isalpha() for c in value):
        return False
    short = sum(1 for t in tokens if len(t.strip(".,&'-")) <= 2)
    return short <= len(tokens) / 2 and any(len(t) >= 3 and t[0].isupper() for t in tokens)


def _clean(field: str, raw: str) -> str | None:
    value = re.sub(r"\s+", " ", raw).strip(" .,:-")
    if not value:
        return None
    if field in _DATE_FIELDS:
        try:
            return to_iso(value.replace("-", "/"))
        except ValueError:
            return None
    if field == "patient_name":
        value = re.sub(r"\((?:synthetic|test)\)", "", value, flags=re.IGNORECASE).strip()
        if value.lower() in _NOT_NAMES or any(c.isdigit() for c in value) or not _plausible_name(value):
            return None
        if "," in value:                       # "SAMPLE, JORDAN Q" -> "Jordan Q Sample"
            last, first = [s.strip() for s in value.split(",", 1)]
            value = f"{first} {last}"
        return value.title() if value.isupper() else value
    if field == "claim_number":
        if not re.search(r"\d", value):
            return None
        return value.upper()
    if field in ("employer", "insurer") and not _plausible_org(value):
        return None
    return value


def _vote_key(field: str, value: str) -> str:
    """Spellings of the same value vote together.

    name:  'Jordan Q. Sample', 'JORDAN SAMPLE', 'Sample, Jordan Q' -> first + last name
    claim: '01-007-261013', '01007261013-3' -> base number without separators or a -N suffix
    """
    plain = re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
    if field == "patient_name":
        tokens = plain.split()
        return f"{tokens[0]} {tokens[-1]}" if len(tokens) >= 2 else plain
    if field == "claim_number":
        base = re.sub(r"-\d{1,2}$", "", value.upper())
        return re.sub(r"[^A-Z0-9]", "", base)
    return plain


def extract_case_details(pages: list[Page]) -> CaseDetails:
    votes: dict[str, Counter] = defaultdict(Counter)                     # field -> key -> pages
    spellings: dict[tuple[str, str], Counter] = defaultdict(Counter)     # (field, key) -> variants
    first_seen: dict[tuple[str, str], Citation] = {}

    for page in pages:
        seen_on_page: set[tuple[str, str]] = set()
        weak_ocr = page.ocr_confidence is not None and page.ocr_confidence < MIN_OCR_CONF_FREE_TEXT
        for line in page.text.splitlines():
            for field, patterns in _COMPILED.items():
                if weak_ocr and field in _FREE_TEXT:
                    continue
                for pattern in patterns:
                    for m in pattern.finditer(line):
                        value = _clean(field, m.group(1))
                        if not value:
                            continue
                        key = _vote_key(field, value)
                        spellings[(field, key)][value] += 1
                        if (field, key) in seen_on_page:
                            continue
                        seen_on_page.add((field, key))
                        votes[field][key] += 1         # one vote per page
                        first_seen.setdefault((field, key),
                                              Citation(page_no=page.page_no, quote=line.strip()))

    def shown(field: str, key: str) -> str:
        variants = spellings[(field, key)]
        if field == "patient_name":       # the fullest spelling: keep the middle initial if any
            return max(variants, key=lambda v: (len(v.split()), variants[v]))
        if field == "claim_number":       # the base number, without a -1/-2 sub-claim suffix
            return re.sub(r"-\d{1,2}$", "", variants.most_common(1)[0][0])
        return variants.most_common(1)[0][0]

    details = CaseDetails()
    for field, counter in votes.items():
        (best, count), *rest = counter.most_common()
        setattr(details, field, FieldValue(
            value=shown(field, best), citation=first_seen[(field, best)], pages_agreeing=count,
            other_values=[shown(field, k) for k, _ in rest][:5],
            variants=sorted(spellings[(field, best)], key=lambda v: -spellings[(field, best)][v])[:6],
        ))
    return details
