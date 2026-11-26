"""Treatment ledger rows from bill pages (CMS-1500 box 24 lines, itemized statements)."""

from __future__ import annotations

import re
from datetime import datetime

from ..gateway import Gateway, GatewayError
from ..schemas import Citation, Page, ServiceRow
from .citations import date_in_quote, quote_on_page, values_in_quote

_DATE = r"\d{1,2}/\d{1,2}/(?:\d{4}|\d{2})"
_CODE = r"\d{5}|\d{4}[A-Z]|[A-Z]\d{4}"          # CPT, CPT cat. III, HCPCS
_AMOUNT = r"\d{1,3}(?:,\d{3})*\.\d{2}"

# One CMS-1500 service line: from [to] POS CPT [MOD] PTR CHARGE UNITS
_LINE = re.compile(
    rf"(?P<from>{_DATE})\s+(?:(?P<to>{_DATE})\s+)?(?P<pos>\d{{2}})\s+"
    rf"(?P<cpt>{_CODE})(?:\s+(?P<mod>[A-Z0-9]{{2}}))?\s+(?P<ptr>[A-L]{{1,4}})\s+"
    rf"\$?(?P<charge>{_AMOUNT})\s+(?P<units>\d{{1,3}})\b"
)
_PROVIDER = re.compile(r"billing provider[^:\n]*:\s*(?P<name>[^\n]+)", re.IGNORECASE)


def to_iso(us_date: str) -> str:
    fmt = "%m/%d/%Y" if len(us_date.split("/")[-1]) == 4 else "%m/%d/%y"
    return datetime.strptime(us_date, fmt).date().isoformat()


def extract_services_rules(page: Page) -> list[ServiceRow]:
    provider_match = _PROVIDER.search(page.text)
    provider = (re.split(r"\s+NPI\b", provider_match.group("name"))[0].strip()
                if provider_match else None)
    rows = []
    for line in page.text.splitlines():
        m = _LINE.search(line)
        if not m:
            continue
        rows.append(ServiceRow(
            date_of_service=to_iso(m.group("from")),
            cpt=m.group("cpt"),
            modifier=m.group("mod"),
            units=int(m.group("units")),
            billed_amount=float(m.group("charge").replace(",", "")),
            dx_pointer=m.group("ptr"),
            provider=provider,
            citation=Citation(page_no=page.page_no, quote=line.strip()),
            confidence=0.95,
            method="rules",
        ))
    return rows


_SYSTEM = """You extract billed service lines from one page of a medical bill (CMS-1500, UB-04 or itemized statement).
Return JSON only: a list of objects with keys
  date_of_service (MM/DD/YYYY), cpt, modifier (or null), units (integer),
  billed_amount (number), dx_pointer (or null), provider (or null),
  quote (the full service line copied EXACTLY, character for character, from the page).
Rules: copy values only from the page; never infer or calculate. If a value is not on the page, use null.
If the page has no service lines, return []."""


def extract_services_llm(page: Page, gateway: Gateway) -> list[ServiceRow]:
    reply = gateway.complete_json("extract", _SYSTEM, f"PAGE {page.page_no}:\n{page.text}", max_tokens=4000)
    rows = []
    for item in reply if isinstance(reply, list) else []:
        try:
            amount = item.get("billed_amount")
            rows.append(ServiceRow(
                date_of_service=to_iso(str(item["date_of_service"])),
                cpt=str(item["cpt"]).strip().upper(),
                modifier=item.get("modifier") or None,
                units=int(item.get("units") or 1),
                billed_amount=float(str(amount).replace(",", "").replace("$", "")) if amount is not None else None,
                dx_pointer=item.get("dx_pointer") or None,
                provider=item.get("provider") or None,
                citation=Citation(page_no=page.page_no, quote=str(item.get("quote", ""))),
                confidence=0.85,
                method="llm",
            ))
        except (KeyError, ValueError, TypeError):
            continue   # malformed row: drop rather than guess
    return rows


def verify_service(row: ServiceRow, pages_by_no: dict[int, Page]) -> bool:
    q = row.citation.quote
    return (
        quote_on_page(row.citation, pages_by_no)
        and date_in_quote(q, row.date_of_service)
        and values_in_quote(q, row.cpt, row.billed_amount)
    )


def extract_services(bill_pages: list[Page], pages_by_no: dict[int, Page], mode: str,
                     gateway: Gateway | None) -> tuple[list[ServiceRow], int]:
    """Returns (verified unique rows, number dropped by the citation verifier)."""
    rows: list[ServiceRow] = []
    for page in bill_pages:
        if mode == "llm" and gateway is not None and gateway.enabled:
            try:
                rows.extend(extract_services_llm(page, gateway))
                continue
            except (GatewayError, ValueError):
                pass   # fall back to the rules baseline for this page
        rows.extend(extract_services_rules(page))

    kept, dropped, seen = [], 0, set()
    for row in rows:
        if not verify_service(row, pages_by_no):
            dropped += 1
            continue
        # The same line on a resubmitted bill must not be counted twice.
        key = (row.date_of_service, row.cpt, row.modifier, row.units, row.billed_amount)
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    kept.sort(key=lambda r: (r.date_of_service, r.cpt))
    return kept, dropped
