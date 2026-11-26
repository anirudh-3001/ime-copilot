"""Synthetic IME case files with known answers, for building and testing without PHI.

Every name, provider and claim number here is invented. The case is shaped
like a real auto-injury IME referral: a question letter, office notes, an MRI,
injection procedure notes, PT notes (one scanned), a faxed duplicate, and
CMS-1500 bills with deliberately inflated PT charges.
"""

from __future__ import annotations

import json
import random
import textwrap
from datetime import date, timedelta
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

W, H = letter
LEFT, TOP, LINE = 54, H - 60, 14

CLAIMANT = "Jordan Q. Sample (SYNTHETIC)"
CLAIM_NO = "SYN-2025-000417"
DOB = "06/02/1981"
EMPLOYER = "Harborview Logistics (fictional)"
INSURER = "Summit Mutual Insurance (fictional)"
DOI = "03/14/2025"
CLINIC = "Lakeside Spine & Pain Associates (fictional)"
PT_CLINIC = "Riverbend Physical Therapy (fictional)"
NPI = "1999999999"

QUESTIONS = [
    ("1", "What are the diagnoses related to the motor vehicle accident of 03/14/2025?"),
    ("2", "Is the claimant's current lumbar condition causally related to the motor vehicle accident of "
          "03/14/2025? Please explain the basis for your opinion, including any pre-existing findings."),
    ("3", "Was the treatment rendered to date reasonable and necessary? Please address each of the "
          "following separately:"),
    ("3a", "Physical therapy"),
    ("3b", "Lumbar epidural steroid injections"),
    ("4", "Has the claimant reached maximum medical improvement? If not, when is it anticipated?"),
    ("5", "If the claimant is at maximum medical improvement, please provide a permanent impairment rating "
          "using the AMA Guides, 6th Edition. Address only the lumbar spine."),
    ("6", "Can the claimant return to work? Please list any restrictions and their expected duration."),
]

QUESTION_CATEGORIES = ["diagnosis", "causation", "treatment_to_date", "treatment_to_date",
                       "treatment_to_date", "mmi", "impairment", "work_status"]


def _us(d: date) -> str:
    return d.strftime("%m/%d/%Y")


class _Doc:
    def __init__(self, path: Path):
        self.c = canvas.Canvas(str(path), pagesize=letter)
        self.page_no = 0
        self.truth_types: dict[int, str] = {}

    def text_page(self, ptype: str, lines: list[str], header: str | None = None) -> int:
        self.page_no += 1
        self.truth_types[self.page_no] = ptype
        y = TOP
        if header:
            self.c.setFont("Helvetica", 8)
            self.c.drawString(LEFT, H - 30, header)
        self.c.setFont("Helvetica", 10)
        for line in lines:
            for chunk in (textwrap.wrap(line, 98, subsequent_indent="   ") or [""]):
                self.c.drawString(LEFT, y, chunk)
                y -= LINE
        self.c.showPage()
        return self.page_no

    def image_page(self, ptype: str, lines: list[str]) -> int:
        """A 'scanned' page: text rendered into an image, so it has no text layer."""
        from PIL import Image, ImageDraw, ImageFont

        self.page_no += 1
        self.truth_types[self.page_no] = ptype
        img = Image.new("L", (1700, 2200), 255)
        draw = ImageDraw.Draw(img)
        font = None
        for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "arial.ttf"):
            try:
                font = ImageFont.truetype(name, 30)
                break
            except OSError:
                continue
        if font is None:
            try:
                font = ImageFont.load_default(size=30)
            except TypeError:
                font = ImageFont.load_default()
        y = 120
        for line in lines:
            draw.text((110, y), line, fill=0, font=font)
            y += 48
        self.c.drawImage(ImageReader(img), 0, 0, W, H)
        self.c.showPage()
        return self.page_no

    def save(self):
        self.c.save()


def _bill_lines(provider: str, dx: list[str], services: list[tuple]) -> list[str]:
    lines = [
        "HEALTH INSURANCE CLAIM FORM (CMS-1500)  -  SYNTHETIC TEST DOCUMENT",
        f"Patient: {CLAIMANT}    DOB: {DOB}    Claim/Policy No: {CLAIM_NO}    Date of injury: {DOI}",
        f"Billing provider: {provider}   NPI {NPI}",
        "21. Diagnosis or nature of illness or injury (ICD-10):",
    ]
    lines += [f"   {chr(65 + i)}. {code}" for i, code in enumerate(dx)]
    lines += ["", "24. A. DOS FROM   TO          B. POS  D. CPT/HCPCS  MOD  E. PTR  F. CHARGES  G. UNITS", ""]
    total = 0.0
    for dos, cpt, mod, ptr, charge, units in services:
        mod_txt = f"{mod}  " if mod else ""
        lines.append(f"{_us(dos)}  {_us(dos)}  11  {cpt}  {mod_txt}{ptr}  {charge:,.2f}  {units}")
        total += charge
    lines += ["", f"28. TOTAL CHARGES  ${total:,.2f}"]
    return lines


def build_case(out_dir: str | Path, case_id: str = "synthetic_001", seed: int = 7) -> tuple[Path, Path]:
    rng = random.Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pdf_path, truth_path = out / f"{case_id}.pdf", out / f"{case_id}.truth.json"
    doc = _Doc(pdf_path)
    truth_services: list[dict] = []
    duplicates: dict[int, int] = {}

    # 1. Referral letter with the case questions
    letter_lines = [
        "Northgate Independent Review Services, LLC (fictional)",
        "",
        "RE: Independent Medical Examination",
        f"Claimant: {CLAIMANT}      DOB: {DOB}      Claim No: {CLAIM_NO}      Date of loss: {DOI}",
        f"Employer: {EMPLOYER}      Insurer: {INSURER}",
        "",
        "Dear Doctor,",
        "Thank you for agreeing to perform an independent medical examination of the above claimant.",
        "Enclosed are the medical records and billing received to date. Please address the following:",
        "",
    ]
    for num, text in QUESTIONS:
        if num[-1].isalpha():
            letter_lines.append(f"   {num[-1]}) {text}")
        else:
            letter_lines.append("")
            letter_lines.append(f"{num}. {text}")
    letter_lines += ["", "Sincerely,", "Claims Review Coordinator (fictional)"]
    doc.text_page("referral_letter", letter_lines)

    # 2. Initial office visit
    visit1 = date(2025, 3, 18)
    office1 = [
        f"{CLINIC}",
        "OFFICE VISIT NOTE",
        f"Patient: {CLAIMANT}    DOB: {DOB}    Date of service: {_us(visit1)}",
        "Chief complaint: Low back pain radiating to the left leg after a motor vehicle accident on 03/14/2025.",
        "Subjective: Rear-ended at a stop light. Pain 7/10. Numbness in the left foot.",
        "Objective: Positive straight leg raise on the left at 40 degrees. Reduced left ankle reflex.",
        "Assessment / Diagnoses:",
        "   M54.16 - Radiculopathy, lumbar region",
        "   S33.5XXA - Sprain of ligaments of lumbar spine, initial encounter",
        "Plan: MRI lumbar spine. Physical therapy 3 times per week. Follow up in 3 weeks.",
        "Electronically signed: A. Rivera, MD (fictional)",
    ]
    office1_page = doc.text_page("office_note", office1)

    # 3. MRI report
    doc.text_page("imaging_report", [
        "Lakeside Imaging Center (fictional) - RADIOLOGY REPORT",
        f"Patient: {CLAIMANT}    Exam date: 04/02/2025",
        "Exam: MRI lumbar spine without contrast",
        "Technique: Multiplanar multisequence images of the lumbar spine.",
        "Findings: L4-L5 broad-based disc bulge. L5-S1 left paracentral disc protrusion contacting the "
        "traversing left S1 nerve root. Mild facet arthropathy L4-S1, chronic in appearance.",
        "Impression: L5-S1 left paracentral protrusion with S1 nerve root contact. Degenerative facet changes.",
    ])

    # 4. Follow-up office visit
    doc.text_page("office_note", [
        f"{CLINIC}",
        "OFFICE VISIT NOTE",
        f"Patient: {CLAIMANT}    Date of service: 04/10/2025",
        "Subjective: Back pain 7/10, leg pain improved slightly with therapy.",
        "Objective: Straight leg raise positive on the left.",
        "Assessment:",
        "   M54.16 - Radiculopathy, lumbar region",
        "   M51.27 - Other intervertebral disc displacement, lumbosacral region",
        "Plan: Left L5-S1 transforaminal epidural steroid injection. Continue physical therapy.",
    ])

    # 5. Faxed duplicate of the first office note
    dup_page = doc.text_page("office_note", office1, header="FAX RECEIVED 04/12/2025 09:14  Page 1 of 1")
    duplicates[dup_page] = office1_page

    # 6-7. Injection procedure notes
    esi_dates = [date(2025, 5, 6), date(2025, 6, 3), date(2025, 7, 1), date(2025, 8, 5),
                 date(2025, 9, 2), date(2025, 10, 7)]
    for d in esi_dates[:2]:
        doc.text_page("operative_report", [
            f"{CLINIC}",
            "PROCEDURE NOTE",
            f"Patient: {CLAIMANT}    Date of procedure: {_us(d)}",
            "Pre-operative diagnosis: M54.16 Radiculopathy, lumbar region",
            "Procedure: Left L5-S1 transforaminal epidural steroid injection under fluoroscopic guidance.",
            "Pre-procedure pain 7/10. Post-procedure pain 5/10. No complications.",
        ])

    # 8-9. PT daily notes (the second one is a scanned image)
    pt_dates = [date(2025, 3, 20) + timedelta(days=7 * i + rng.choice([0, 2])) for i in range(10)]
    doc.text_page("pt_note", [
        f"{PT_CLINIC}",
        "PHYSICAL THERAPY DAILY NOTE",
        f"Patient: {CLAIMANT}    Date of service: {_us(pt_dates[0])}    Visit #1",
        "Treatment: Therapeutic exercise 97110 x4, manual therapy 97140 x2, hot pack 97010.",
        "Pain before 7/10, after 6/10. Oswestry 48%.",
    ])
    doc.image_page("pt_note", [
        f"{PT_CLINIC}",
        "PHYSICAL THERAPY DAILY NOTE",
        f"Date of service: {_us(pt_dates[5])}    Visit #6",
        "Treatment: Therapeutic exercise, manual therapy, hot pack.",
        "Pain before 7/10, after 7/10. Oswestry 46%.",
    ])

    # 10. PT bill: about $1,150 a visit against a local cash price near $100
    pt_services = []
    for d in pt_dates:
        pt_services += [(d, "97110", "GP", "A", 600.00, 4), (d, "97140", "GP", "A", 350.00, 2),
                        (d, "97010", None, "A", 200.00, 1)]
    bill_pages = [
        (PT_CLINIC, ["M54.16", "S33.5XXA"], pt_services[:15]),
        (PT_CLINIC, ["M54.16", "S33.5XXA"], pt_services[15:]),
        (CLINIC, ["M54.16", "M51.27"],
         [(d, "64483", "LT", "A", 1150.00, 1) for d in esi_dates]
         + [(visit1, "99204", None, "A", 425.00, 1), (date(2025, 4, 10), "99214", None, "AB", 275.00, 1),
            (date(2025, 4, 2), "72148", None, "B", 2400.00, 1)]),
    ]
    for provider, dx, svc in bill_pages:
        doc.text_page("bill", _bill_lines(provider, dx, svc))
        for dos, cpt, mod, _ptr, charge, units in svc:
            truth_services.append({"date_of_service": dos.isoformat(), "cpt": cpt, "modifier": mod,
                                   "units": units, "billed_amount": charge})

    doc.save()
    truth = {
        "case_id": case_id,
        "page_types": {str(k): v for k, v in doc.truth_types.items()},
        "duplicates": {str(k): v for k, v in duplicates.items()},
        "services": truth_services,
        "diagnoses": ["M54.16", "S33.5XXA", "M51.27"],
        "details": {"patient_name": "Jordan Q. Sample", "date_of_birth": "1981-06-02",
                    "claim_number": CLAIM_NO, "date_of_injury": "2025-03-14",
                    "employer": EMPLOYER.replace(" (fictional)", ""),
                    "insurer": INSURER.replace(" (fictional)", "")},
        "questions": [{"number": n, "text": t, "category": c}
                      for (n, t), c in zip(QUESTIONS, QUESTION_CATEGORIES)],
    }
    truth_path.write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return pdf_path, truth_path
