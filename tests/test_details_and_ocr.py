import time

import pytest
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from ime_copilot.extract.demographics import extract_case_details
from ime_copilot.ingest.pdf import parse_range, read_pages, tesseract_available
from ime_copilot.schemas import Page


def test_details_from_real_world_label_styles():
    pages = [
        Page(page_no=1, text="PATIENT NAME: SAMPLE, JORDAN Q\nD.O.B. 6/2/1981   SSN: 123-45-6789\n"
                             "Claim #: WC-88812-A   DOI: 03/14/25\nEmployer: Harborview Logistics"),
        Page(page_no=2, text="Patient: Jordan Q Sample   DOB: 06/02/1981\nClaim Number: WC-88812-A"),
        Page(page_no=3, text="Patient: Jordan Q Sample\nDate of Injury: 03/14/2025\nInsurer: Summit Mutual"),
        Page(page_no=4, text="Patient: Jamie Other   DOB: 01/01/1990"),     # a stray page from another file
    ]
    d = extract_case_details(pages)
    assert d.patient_name.value == "Jordan Q Sample" and d.patient_name.pages_agreeing == 3
    assert "Jamie Other" in d.patient_name.other_values          # disagreement surfaced, not hidden
    assert d.date_of_birth.value == "1981-06-02" and d.date_of_birth.pages_agreeing == 2
    assert d.claim_number.value == "WC-88812-A"
    assert d.date_of_injury.value == "2025-03-14"
    assert d.employer.value == "Harborview Logistics" and d.insurer.value == "Summit Mutual"
    assert d.ssn_last4.value == "6789"                            # never the full SSN
    assert "123-45" not in d.model_dump_json().replace(d.ssn_last4.citation.quote, "")


def test_details_group_variants_and_reject_ocr_garbage():
    """Patterns seen on a real 1,587-page file, reproduced with invented values."""
    pages = [Page(page_no=n, text="Patient Name: Doe, Jane R\nClaim #: 01009990001-3") for n in range(1, 4)]
    pages += [
        Page(page_no=4, text="Patient: Jane Doe   DOB: 02/03/1960\nClaim #: 01009990001-1"),
        Page(page_no=5, text="Patient: JANE DOE\nClaim # 01-009-990001\nEmployer name: Riverside Bakery Co"),
        Page(page_no=6, text="Claimant: Patient Name\nPatient: MOY OF eg"),                 # label / garbage
        Page(page_no=7, text="My insurance company: th Wl CHU Chin ee |"),                # handwritten scrawl
        Page(page_no=8, text="Patient: Jxne Dqe Smithe", text_source="tesseract", ocr_confidence=0.55),
        Page(page_no=9, text="Employer name: Lakeview Storage LLC"),
    ]
    d = extract_case_details(pages)
    assert d.patient_name.value == "Jane R Doe"                 # fullest spelling
    assert d.patient_name.pages_agreeing == 5                   # all variants counted together
    assert d.patient_name.other_values == []                    # no label words, garbage or bad-OCR names
    assert d.claim_number.value == "01009990001"
    assert d.claim_number.pages_agreeing == 5
    assert "01009990001-3" in d.claim_number.variants and "01-009-990001" in d.claim_number.variants
    assert d.insurer is None                                    # garbage rejected
    assert d.employer.value in ("Riverside Bakery Co", "Lakeview Storage LLC")
    assert len(d.employer.other_values) == 1                    # disagreement still surfaced


def test_parse_range():
    assert parse_range("1-50") == (1, 50) and parse_range("7") == (7, 7) and parse_range("") is None


def _scanned_pdf(path, n_pages):
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 34)
    except OSError:
        font = ImageFont.load_default()
    c = canvas.Canvas(str(path), pagesize=letter)
    for i in range(1, n_pages + 1):
        img = Image.new("L", (1275, 1650), 255)
        d = ImageDraw.Draw(img)
        for j, line in enumerate([f"PHYSICAL THERAPY DAILY NOTE {i}", "Patient: Jordan Q Sample",
                                  f"Date of service: 04/{i:02d}/2025", "Therapeutic exercise 97110"]):
            d.text((90, 90 + 60 * j), line, fill=0, font=font)
        c.drawImage(ImageReader(img), 0, 0, *letter)
        c.showPage()
    c.save()


@pytest.mark.skipif(not tesseract_available(), reason="Tesseract/Poppler not installed")
def test_parallel_ocr_page_range_and_resume(tmp_path):
    pdf = tmp_path / "scanned.pdf"
    _scanned_pdf(pdf, 6)
    cache = tmp_path / "ocr_cache"

    pages, _ = read_pages(pdf, "auto", page_range=(2, 5), workers=2, cache_dir=cache, dpi=200, progress=False)
    assert [p.page_no for p in pages] == [2, 3, 4, 5]
    assert all(p.text_source == "tesseract" for p in pages)
    assert "Jordan" in pages[0].text and "04/02/2025" in pages[0].text
    assert len(list(cache.rglob("page_*.json"))) == 4

    t0 = time.monotonic()
    again, _ = read_pages(pdf, "auto", page_range=(2, 5), cache_dir=cache, dpi=200, progress=False)
    assert time.monotonic() - t0 < 2                               # served from cache
    assert [p.text for p in again] == [p.text for p in pages]
