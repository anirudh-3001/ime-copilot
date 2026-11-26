"""Typed records that flow through the pipeline.

Every extracted fact carries a Citation (page number + the exact quoted text).
Rows whose quote cannot be found on the cited page are dropped (principle P2:
cite or omit), so nothing unsupported reaches the ledger.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

PAGE_TYPES = [
    "referral_letter",   # IME company's cover letter / questionnaire
    "office_note",
    "operative_report",  # procedures, injections
    "pt_note",           # physical therapy / chiropractic visit notes
    "imaging_report",
    "bill",              # CMS-1500, UB-04, itemized statements
    "prior_ime",
    "legal",
    "other",
]

QUESTION_CATEGORIES = [
    "diagnosis", "causation", "treatment_to_date", "requested_services", "mmi",
    "impairment", "work_status", "future_care", "apportionment", "other",
]


class Citation(BaseModel):
    page_no: int = Field(description="1-based page number in the case file")
    quote: str = Field(description="Exact text copied from that page")


class PageTable(BaseModel):
    """A table found on a page (Document Intelligence), as a grid of cell text."""
    row_count: int
    column_count: int
    cells: list[list[str]]                  # cells[row][column]
    continues_on_pages: list[int] = Field(default_factory=list)


class Page(BaseModel):
    page_no: int
    text: str
    text_source: str = "pdftext"            # pdftext | tesseract | azure_di | textract
    ocr_confidence: Optional[float] = None  # 0-1 when OCR reports it
    low_confidence_words: float = 0.0       # share of words with confidence < 0.5
    angle: Optional[float] = None           # page rotation detected by OCR, degrees
    tables: list[PageTable] = Field(default_factory=list)
    page_type: str = "other"
    type_confidence: float = 0.0
    duplicate_of: Optional[int] = None      # page_no of the first copy
    needs_review: bool = False              # low OCR confidence or empty page


class ServiceRow(BaseModel):
    date_of_service: str                    # ISO yyyy-mm-dd
    cpt: str                                # CPT or HCPCS code
    modifier: Optional[str] = None
    units: int = 1
    billed_amount: Optional[float] = None
    dx_pointer: Optional[str] = None
    provider: Optional[str] = None
    citation: Citation
    confidence: float = 1.0
    method: str = "rules"                   # rules | llm


class DiagnosisRow(BaseModel):
    code: str                               # ICD-10-CM
    description: Optional[str] = None
    first_seen_date: Optional[str] = None
    citation: Citation
    confidence: float = 1.0
    method: str = "rules"


class CaseQuestion(BaseModel):
    number: str                             # exactly as the IME company numbered it, e.g. "3b"
    text: str                               # verbatim
    category: str = "other"
    instructions: list[str] = Field(default_factory=list)
    citation: Citation
    method: str = "rules"


class FieldValue(BaseModel):
    value: str
    citation: Citation
    pages_agreeing: int = 1                 # how many pages show this same value
    other_values: list[str] = Field(default_factory=list)   # disagreements worth a look
    variants: list[str] = Field(default_factory=list)       # spellings grouped into this value


class CaseDetails(BaseModel):
    """Claimant and claim header, voted across pages."""
    patient_name: Optional[FieldValue] = None
    date_of_birth: Optional[FieldValue] = None
    claim_number: Optional[FieldValue] = None
    date_of_injury: Optional[FieldValue] = None
    employer: Optional[FieldValue] = None
    insurer: Optional[FieldValue] = None
    ssn_last4: Optional[FieldValue] = None


class CaseResult(BaseModel):
    case_id: str
    source_file: str
    pages: list[Page]
    details: CaseDetails = Field(default_factory=CaseDetails)
    services: list[ServiceRow]
    diagnoses: list[DiagnosisRow]
    questions: list[CaseQuestion]
    dropped_uncited: int = 0                # rows removed by the citation verifier
    model_cost_usd: float = 0.0
