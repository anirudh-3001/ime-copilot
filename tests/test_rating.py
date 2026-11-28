"""Reference library (AMA / Colorado) and the rating worksheets."""

import json
import re

import pytest
from fastapi.testclient import TestClient
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from ime_copilot import api
from ime_copilot.config import Settings
from ime_copilot.gateway import FakeProvider, Gateway
from ime_copilot.guides import build_libraries, cite_label, forget_indexes, load_library, printed_page
from ime_copilot.pipeline import process_case
from ime_copilot.rag.qa import CaseQA
from ime_copilot.rating import combine, guide_guidance, record_findings
from ime_copilot.synthetic import build_case

FILLER = " Physicians should record the measurements carefully and repeat them for consistency." * 8


def _pdf(path, pages):
    c = canvas.Canvas(str(path), pagesize=letter)
    for t in pages:
        y = 740
        for line in t.split("\n"):
            c.drawString(50, y, line[:110])
            y -= 16
        c.showPage()
    c.save()


@pytest.fixture
def library(tmp_path):
    guides = tmp_path / "guides"
    (guides / "AMA").mkdir(parents=True)
    (guides / "Colorado").mkdir()
    _pdf(guides / "AMA" / "guides part 1.pdf", [
        "86 GUIDES TO THE EVALUATION OF PERMANENT IMPAIRMENT\nThe Lumbar Spine. Range of motion of the lumbar spine "
        "is measured with two inclinometers." + FILLER,
        "THE MUSCULOSKELETAL SYSTEM 87\nTable 53. Impairments Due to Specific Disorders of the Spine. Lumbar disc "
        "surgery with residual pain is rated from this table." + FILLER,
        "254 GUIDES TO THE EVALUATION OF PERMANENT IMPAIRMENT\nCombined Values Chart. Combine the larger value with "
        "the smaller value." + FILLER])
    _pdf(guides / "Colorado" / "curriculum.pdf", [
        "Impairment Rating Tips (Desk Aid 11)\nLumbar spine: the physician must report the tables used and range of "
        "motion validity." + FILLER + "\n12"])
    (guides / "AMA" / "copy of curriculum.pdf").write_bytes((guides / "Colorado" / "curriculum.pdf").read_bytes())
    s = Settings(guides_dir=str(guides), index_dir=str(tmp_path / "index"), db_url=f"sqlite:///{tmp_path / 'r.db'}")
    forget_indexes()
    counts = build_libraries(s)
    return s, counts


def test_combined_values():
    assert combine([10, 40]) == (46, ["40% combined with 10% = 46%"])
    assert combine([20, 10, 5])[0] == 32
    assert combine([]) == (0, []) and combine([0, 25]) == (25, [])


def test_printed_page_numbers():
    assert printed_page("38 GUIDES TO THE EVALUATION OF PERMANENT IMPAIRMENT\ntext") == 38
    assert printed_page("THE VISUAL SYSTEM 163\ntext") == 163
    assert printed_page("Chapter text\nmore\n12") == 12
    assert printed_page("no number here\nat all") is None


def test_library_is_built_deduplicated_and_labelled(library):
    s, counts = library
    assert counts == {"colorado": 1, "ama": 3}                  # the copied curriculum is indexed once
    ama = load_library(s, "ama")
    assert [p["printed_page"] for p in ama] == [86, 87, 254]
    assert cite_label("ama", ama[1]) == "AMA Guides 3rd Ed. Rev., p. 87"
    assert cite_label("colorado", load_library(s, "colorado")[0]) == "curriculum, p. 12"
    again = build_libraries(s)                                    # second run reuses what was read
    assert again == counts


def test_worksheet_findings_and_guidance_are_linked(library, tmp_path):
    s, _ = library
    pdf, _ = build_case(tmp_path / "case")
    r = process_case(pdf, "rate1", s, progress=False)
    prompts = {}

    def model(name, user):
        if "RATING INPUTS" in user:
            prompts["guide"] = user
            g = {lab: n for n, lab in re.findall(r"\[G(\d+)\] \(([^)]+)\)", user)}
            t53 = g["AMA Guides 3rd Ed. Rev., p. 87"]
            tips = next(n for lab, n in g.items() if lab.startswith("curriculum"))
            return json.dumps({"body_parts": [{
                "name": "Lumbar spine", "method": f"Rate with Table 53 plus range of motion [G{t53}].",
                "tables": [{"name": "Table 53", "use": f"Specific disorders of the spine [G{t53}]"}],
                "measure_at_exam": [f"Lumbar range of motion with two inclinometers [G{t53}]", "Straight leg raise [G99]"],
                "inputs_present": ["MRI findings"], "inputs_missing": ["Inclinometer ROM"],
                "notes": [f"Report the tables used [G{tips}]"]}],
                "general": ["Combine values largest first"]})
        n = re.search(r"\[(\d+)\] \(p\. 3\)", user).group(1)
        return json.dumps({"body_parts": [{"name": "Lumbar spine",
                                           "imaging": [f"MRI lumbar spine [{n}]"],
                                           "diagnoses": ["Radiculopathy without a source"]}]})
    gw = Gateway(Settings(provider="local", models={"answer": "gemma4:26b-a4b"}), provider=FakeProvider(model))
    qa = CaseQA(r.pages, r.services, gateway=gw, use_embeddings=False)
    findings = record_findings(qa, s, "rate1")
    bp = findings["body_parts"][0]
    assert bp["imaging"][0]["sources"][0]["page"] == 3 and "(p. 3)" in bp["imaging"][0]["text"]
    assert bp["diagnoses"] == [] and findings["dropped"] == 1      # no source: left out
    out = guide_guidance("colorado", findings, s, gw)
    assert "Lumbar spine" in prompts["guide"] and "MRI lumbar spine" in prompts["guide"]
    g = out["body_parts"][0]
    assert g["method"]["cited"] and g["method"]["cites"][0]["label"] == "AMA Guides 3rd Ed. Rev., p. 87"
    assert "(AMA Guides 3rd Ed. Rev., p. 87)" in g["method"]["text"] and "[G" not in g["method"]["text"]
    assert [x["cited"] for x in g["measure_at_exam"]] == [True, False]   # [G99] isn't a real passage
    assert g["notes"][0]["cites"][0]["library"] == "colorado"
    assert out["general"][0]["cited"] is False and out["uncited"] == 2
    assert out["libraries"] == ["colorado", "ama"]


def test_rating_api(library, tmp_path, monkeypatch):
    s, _ = library
    pdf, _ = build_case(tmp_path / "case2")
    process_case(pdf, "rate2", s, progress=False)
    monkeypatch.setattr(api, "settings", lambda: s)
    c = TestClient(api.app)
    d = c.get("/cases/rate2/rating/ama").json()
    assert d["status"] == "not_started" and d["library_ready"] and d["model_available"] is False
    assert c.post("/cases/rate2/rating/ama", json={}).status_code == 400          # no model configured
    assert c.get("/cases/rate2/rating/nope").status_code == 404
    page = c.get("/guides/ama/pages/2").json()
    assert page["label"] == "AMA Guides 3rd Ed. Rev., p. 87" and "Table 53" in page["text"]
    assert c.get("/guides/ama/pages/99").status_code == 404
