"""Document catalog: per-page extraction checked against the page, and the lists built from it."""

import json
import re

import pytest

from ime_copilot.catalog import (bill_rows, build_catalog, catalog_facts, catalog_views, load_catalog,
                                 timeline_facts, verify_entry)
from ime_copilot.config import Settings
from ime_copilot.gateway import FakeProvider, Gateway
from ime_copilot.schemas import Page

PAGES = [
    Page(page_no=1, text="Lakeside Spine & Pain Associates\nDate of service: 03/01/2024\nProvider: Alan Smith, MD\n"
                         "Chief complaint: low back pain 7/10\nPlan: lumbar MRI"),
    Page(page_no=2, text="Lakeside Imaging Center\nExam: MRI lumbar spine 03/05/2024\n"
                         "Impression: L4-L5 disc protrusion\nRadiologist: Maria Lopez, MD"),
    Page(page_no=3, text="Valley Surgery Center OPERATIVE REPORT\nDate: 04/10/2024\nSurgeon: Dr. Alan Smith\n"
                         "Procedure: L4-L5 microdiscectomy\nAnesthesia: general"),
    Page(page_no=4, text="Lakeside Spine & Pain Associates\nDate of service: 05/20/2024\nProvider: Alan Smith, MD\n"
                         "Follow-up after surgery, pain 2/10"),
    Page(page_no=5, text="ITEMIZED STATEMENT Valley Surgery Center\n04/10/2024 63030 Microdiscectomy 1 $12,500.00\n"
                         "03/01/2024 99204 Office visit 1 $350.00"),
]

REPLIES = {
    1: {"doc_type": "office visit", "title": "Lakeside Spine & Pain Associates", "date": "03/01/2024",
        "providers": [{"name": "Alan Smith, MD", "role": "treating physician"},
                      {"name": "Robert Jones", "role": "physician"}],            # not on the page: dropped
        "facility": "Lakeside Spine & Pain Associates", "procedures": [], "imaging": [],
        "bill_lines": [], "pain": "low back pain 7/10", "note": "Initial visit."},
    2: {"doc_type": "imaging report", "date": "March 5, 2024",                    # written differently: still found
        "providers": [{"name": "Maria Lopez, MD", "role": "radiologist"}], "facility": "Lakeside Imaging Center",
        "imaging": [{"study": "MRI lumbar spine", "impression": "L4-L5 disc protrusion"},
                    {"study": "CT head", "impression": "normal"}],                # not on the page: dropped
        "procedures": [], "bill_lines": [], "pain": "", "note": "MRI."},
    3: {"doc_type": "operative report", "date": "04/10/2024", "providers": [{"name": "Dr. Alan Smith", "role": "surgeon"}],
        "facility": "Valley Surgery Center",
        "procedures": [{"name": "L4-L5 microdiscectomy", "cpt": "63030", "surgery": True}],   # code not on page: blanked
        "imaging": [], "bill_lines": [], "pain": "", "note": "Surgery."},
    4: {"doc_type": "office visit", "date": "05/20/2024", "providers": [{"name": "Alan Smith, MD", "role": "treating physician"}],
        "facility": "Lakeside Spine & Pain Associates", "procedures": [], "imaging": [], "bill_lines": [],
        "pain": "pain 2/10", "note": "Follow-up."},
    5: {"doc_type": "bill", "date": "", "providers": [], "facility": "Valley Surgery Center", "procedures": [], "imaging": [],
        "bill_lines": [{"date": "04/10/2024", "cpt": "63030", "description": "Microdiscectomy", "units": 1,
                        "amount": 12500, "provider": "Valley Surgery Center"},
                       {"date": "03/01/2024", "cpt": "99204", "units": 1, "amount": 350.0, "provider": ""},
                       {"date": "03/01/2024", "cpt": "99214", "units": 1, "amount": 999.0}],   # not on page: dropped
        "pain": "", "note": "Bill."},
}


def _gateway(calls):
    def reply(model, user):
        n = int(re.match(r"PAGE (\d+):", user).group(1))
        calls.append(n)
        return json.dumps(REPLIES[n])
    return Gateway(Settings(provider="local", models={"extract": "gemma4:26b-a4b", "answer": "gemma4:26b-a4b"}),
                   provider=FakeProvider(reply))


@pytest.fixture
def catalog(tmp_path):
    calls = []
    progress = []
    entries = build_catalog(PAGES, _gateway(calls), tmp_path, "c1", on_progress=lambda d, t: progress.append((d, t)))
    return entries, calls, progress, tmp_path


def test_only_what_is_on_the_page_is_kept(catalog):
    entries, calls, progress, _ = catalog
    assert sorted(calls) == [1, 2, 3, 4, 5] and progress[-1] == (5, 5)
    assert [p["name"] for p in entries[1]["providers"]] == ["Alan Smith, MD"] and entries[1]["dropped"] == 1
    assert entries[2]["date"] == "2024-03-05" and [i["study"] for i in entries[2]["imaging"]] == ["MRI lumbar spine"]
    assert entries[3]["procedures"][0]["cpt"] == "" and entries[3]["procedures"][0]["surgery"] is True
    assert [(b["cpt"], b["amount"]) for b in entries[5]["bill_lines"]] == [("63030", 12500.0), ("99204", 350.0)]


def test_lists_for_the_summary(catalog):
    entries, _, _, _ = catalog
    by_no = {p.page_no: p for p in PAGES}
    rows = bill_rows(entries, by_no, [])
    assert {(r.cpt, r.billed_amount) for r in rows} == {("63030", 12500.0), ("99204", 350.0)}
    assert {r.provider for r in rows} == {"Valley Surgery Center"}           # facility named on the bill
    views = catalog_views(entries, rows, "2024-02-20")
    smith = next(p for p in views["providers"] if "Smith" in p["name"])
    assert smith["visits"] == 3 and smith["first_date"] == "2024-03-01" and smith["last_date"] == "2024-05-20"
    assert "L4-L5 microdiscectomy" in smith["performed"]
    assert [d["date"] for d in views["dates_of_service"]] == ["2024-03-01", "2024-03-05", "2024-04-10", "2024-05-20"]
    assert views["imaging"][0]["study"] == "MRI lumbar spine" and views["imaging"][0]["date"] == "2024-03-05"
    surgery = views["surgeries"][0]
    assert surgery["name"] == "L4-L5 microdiscectomy" and surgery["cost"] == 12500.0
    ba = views["before_after"]
    assert ba["before"]["visits"] == 2 and ba["after"]["visits"] == 1
    assert ba["before"]["pain_notes"][0]["pain"] == "low back pain 7/10" and ba["after"]["pain_notes"][0]["pain"] == "pain 2/10"
    assert ba["before"]["billed"] == 350.0
    assert views["billed_by"][0] == {"name": "Valley Surgery Center", "billed": 12850.0, "lines": 2,
                                     "first_date": "2024-03-01", "last_date": "2024-04-10",
                                     "codes": ["63030", "99204"], "pages": [5]}
    assert views["unattributed_billed"] == 0
    assert [d["doc_type"] for d in views["documents"]][:2] == ["office visit", "imaging report"]
    facts = catalog_facts(views, by_no) + timeline_facts(views, entries, by_no)
    assert any("Surgery: L4-L5 microdiscectomy on 2024-04-10" in f["text"] for f in facts)
    assert all(by_no[p].text and q for f in facts for p, q in f["sources"])


def test_catalog_resumes_and_skips_unchanged_pages(catalog):
    entries, _, _, tmp = catalog
    calls = []
    again = build_catalog(PAGES, _gateway(calls), tmp, "c1")
    assert calls == [] and set(again) == set(entries) and set(load_catalog(tmp, "c1")) == {1, 2, 3, 4, 5}
    changed = [Page(page_no=4, text=PAGES[3].text + " Discharged.")] + PAGES[:3] + PAGES[4:]
    build_catalog(changed, _gateway(calls), tmp, "c1")
    assert calls == [4]


def test_bad_reply_is_harmless():
    e = verify_entry("not a dict", PAGES[0])
    assert e["providers"] == [] and e["date"] is None and e["doc_type"] == "other"
