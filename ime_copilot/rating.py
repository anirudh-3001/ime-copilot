"""Impairment rating worksheets: AMA Guides and Colorado.

A DRAFT to help the rating physician; it never gives the rating itself. Three parts:

1. Rating inputs from the records (shared by both tabs): per body part, the diagnoses,
   surgeries, imaging, past range-of-motion and neurological findings, MMI statements, earlier
   ratings, prior injuries (apportionment) and restrictions, each linked to its page. Items the
   model gives without a valid source are left out.
2. Guidance from the reference library: which sections and tables of the Guides apply to each
   body part, what must be measured at the exam, which inputs are present or missing, and (for
   the Colorado tab) Colorado's rules from the DOWC curriculum and Desk Aid 11. Each statement
   is linked to a Guides page; statements without a valid Guides source are marked.
3. A combined-values calculator (in the web page) for the values the physician determines:
   combine(a, b) = a + b x (100 - a) / 100, largest first, as in the Guides' Combined Values Chart.

Saved per case: <index_dir>/<case>/rating_findings.json and rating_<tab>.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

TABS = {
    "ama": {"title": "AMA Guides rating", "libraries": ["ama"],
            "intro": "the AMA Guides to the Evaluation of Permanent Impairment (the edition in the library)"},
    "colorado": {"title": "Colorado rating", "libraries": ["colorado", "ama"],
                 "intro": "Colorado workers' compensation rules: the AMA Guides, 3rd Edition (Revised), as applied "
                          "by the Colorado Division of Workers' Compensation Level II curriculum and its Impairment "
                          "Rating Tips (Desk Aid 11)"},
}

FINDINGS_QUERIES = [
    "range of motion measurements degrees flexion extension inclinometer",
    "neurological examination strength reflexes sensation dermatome radiculopathy",
    "maximum medical improvement MMI date",
    "impairment rating whole person percent",
    "diagnosis assessment injured body part",
    "surgery operative report procedure",
    "MRI CT X-ray EMG findings impression",
    "prior injury previous accident pre-existing condition apportionment",
    "work restrictions permanent restrictions functional capacity",
]

FIELDS = ["diagnoses", "surgery", "imaging", "range_of_motion", "neuro", "mmi", "prior_ratings",
          "prior_injuries", "restrictions"]
FIELD_LABELS = {"diagnoses": "Diagnoses", "surgery": "Surgery / procedures", "imaging": "Imaging and tests",
                "range_of_motion": "Range of motion recorded", "neuro": "Neurological findings",
                "mmi": "MMI statements", "prior_ratings": "Earlier impairment ratings",
                "prior_injuries": "Prior injuries / apportionment", "restrictions": "Restrictions"}

_FINDINGS_SYSTEM = """You prepare impairment-rating inputs for a physician from ONE claim file.
From the FACT SHEET and numbered PASSAGES, list what the records say that matters for rating
permanent impairment, grouped by injured body part (e.g. "Lumbar spine", "Cervical spine",
"Right shoulder"). Use only what the material says.

Reply with JSON only:
{"body_parts": [{"name": "<body part>",
   "diagnoses": ["<finding with source marker>"], "surgery": [...], "imaging": [...],
   "range_of_motion": ["<measurements with dates and who measured>"], "neuro": [...],
   "mmi": [...], "prior_ratings": [...], "prior_injuries": [...], "restrictions": [...]}]}
Rules: every item must end with the passage or fact number it came from, e.g. [4] or [F2]; keep
items short and specific (values, dates, sides, levels); leave a list empty when the records say
nothing; at most 6 body parts, most important first."""

_GUIDE_SYSTEM = """You help a rating physician prepare an impairment rating under %s.
You are given the rating inputs found in the claim file (by body part) and numbered passages from
the reference library [G1], [G2] ... For each body part, explain how the reference says it should be
rated. Use only what the reference passages say; if they don't cover something, say so.

Reply with JSON only:
{"body_parts": [{"name": "<body part, as in the inputs>",
   "method": "<2-4 sentences: which chapter/section and approach applies, with [Gn] markers>",
   "tables": [{"name": "<table or figure name/number as written in the reference>", "use": "<what it is used for here> [Gn]"}],
   "measure_at_exam": ["<what the physician must measure or document at the exam, and how> [Gn]"],
   "inputs_present": ["<which needed inputs the records already have>"],
   "inputs_missing": ["<which needed inputs are missing from the records>"],
   "notes": ["<rules, cautions, validity criteria, apportionment or reporting requirements> [Gn]"]}],
 "general": ["<rules that apply to the whole rating, e.g. combining values, rounding, reporting> [Gn]"]}
Rules: cite the reference passage for every rule, table or method with [Gn]; never invent table
numbers or percentages; do not give the impairment percentage for this patient; that is the
physician's determination."""


def combine(values: list[float]) -> tuple[int, list[str]]:
    """Combine whole-person impairments, largest first: a + b(100 - a)/100, rounded at each step."""
    def half_up(x: float) -> int:          # same rounding as the calculator in the web page
        return int(x + 0.5)
    vals = sorted((half_up(v) for v in values if v and v > 0), reverse=True)
    if not vals:
        return 0, []
    total, steps = vals[0], []
    for v in vals[1:]:
        new = half_up(total + v * (100 - total) / 100)
        steps.append(f"{total}% combined with {v}% = {new}%")
        total = new
    return total, steps


def _path(settings, case_id: str, name: str) -> Path:
    return Path(settings.index_dir) / case_id / f"{name}.json"


def load(settings, case_id: str, name: str) -> dict | None:
    p = _path(settings, case_id, name)
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except (OSError, ValueError):
        return None


def _save(settings, case_id: str, name: str, data: dict) -> dict:
    p = _path(settings, case_id, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


# ----------------------------------------------------------------------------- part 1: record findings
def record_findings(qa, settings, case_id: str) -> dict:
    from .case_summary import gather_passages
    from .rag.qa import _resolve_passage_markers
    hits = gather_passages(qa, queries=FINDINGS_QUERIES, per_query=3, limit=18)
    facts = "\n".join(f"[F{i}] {f['text']}" for i, f in enumerate(qa.facts, 1)) or "none"
    passages = "\n\n".join(f"[{i}] (p. {h.passage.page_no})\n{h.passage.text}" for i, h in enumerate(hits, 1))
    out = {"body_parts": [], "dropped": 0, "warnings": []}
    try:
        reply = qa.gateway.complete_json("answer", _FINDINGS_SYSTEM,
                                         f"FACT SHEET:\n{facts}\n\nPASSAGES:\n{passages}", max_tokens=2500)
    except Exception as err:  # noqa: BLE001
        out["warnings"].append(f"The model could not read the rating inputs: {err}")
        return _save(settings, case_id, "rating_findings", out)
    for bp in (reply.get("body_parts") or []) if isinstance(reply, dict) else []:
        if not isinstance(bp, dict) or not str(bp.get("name") or "").strip():
            continue
        part = {"name": str(bp["name"]).strip()[:60]}
        for field in FIELDS:
            items = []
            for raw in bp.get(field) or []:
                text, sources, bad = _resolve_passage_markers(str(raw), hits, qa.pages_by_no, qa.facts)
                if sources:
                    items.append({"text": text, "sources": [s.model_dump() for s in sources]})
                else:
                    out["dropped"] += 1
            part[field] = items
        if any(part[f] for f in FIELDS):
            out["body_parts"].append(part)
    return _save(settings, case_id, "rating_findings", out)


# ----------------------------------------------------------------------------- part 2: guidance
_GMARK = re.compile(r"\[\s*G\s*(\d{1,3}(?:\s*[,;]\s*G?\s*\d{1,3})*)\s*\]", re.IGNORECASE)


def _resolve_guides(text: str, ghits: list[tuple], quote_for) -> tuple[str, list[dict], int]:
    """[G3] -> a citation of the reference page; returns (clean text, citations, invalid markers)."""
    cites, bad = [], 0

    def repl(m):
        nonlocal bad
        labels = []
        for n in re.findall(r"\d+", m.group(1)):
            n = int(n)
            if not 1 <= n <= len(ghits):
                bad += 1
                continue
            lib, hit, meta, label = ghits[n - 1]
            q = quote_for(hit.passage.text, _GMARK.sub(" ", text))
            cites.append({"library": lib, "gid": meta["gid"], "label": label, "quote": q})
            labels.append(label)
        return "".join(f" ({lb})" for lb in dict.fromkeys(labels))
    clean = _GMARK.sub(repl, text)
    return re.sub(r"\s+([.,;:])", r"\1", clean).strip(), cites, bad


def guide_guidance(tab: str, findings: dict, settings, gateway) -> dict:
    from .guides import cite_label, get_guide_index
    from .rag.qa import quote_for
    spec = TABS[tab]
    out = {"tab": tab, "title": spec["title"], "body_parts": [], "general": [], "warnings": [],
           "uncited": 0, "libraries": []}
    indexes = {lib: get_guide_index(settings, lib, gateway) for lib in spec["libraries"]}
    indexes = {k: v for k, v in indexes.items() if v is not None}
    out["libraries"] = list(indexes)
    if not indexes:
        out["warnings"].append("No reference library found: put the PDFs in the guides folder and run "
                               "'python -m ime_copilot guides'.")
        return out
    parts = findings.get("body_parts") or [{"name": "Injured body parts"}]
    per_part = max(3, 18 // max(1, len(parts)))
    ghits, seen = [], set()
    for bp in parts[:6]:
        name = bp["name"]
        queries = [f"{name} impairment evaluation", f"{name} range of motion measurement table impairment",
                   f"{name} specific disorders diagnosis table percent"]
        if tab == "colorado":
            queries += [f"{name} impairment rating tips Colorado", f"{name} rating example case"]
        taken = 0
        for q in queries:
            for lib, idx in indexes.items():
                for h in idx.search(q, k=3):
                    key = (lib, h.passage.pid)
                    if key in seen or taken >= per_part:
                        continue
                    seen.add(key)
                    meta = idx.meta[h.passage.page_no]
                    ghits.append((lib, h, meta, cite_label(lib, meta)))
                    taken += 1
    for q in (["combined values chart combining impairments whole person"]
              + (["impairment rating tips general rules report requirements"] if tab == "colorado" else [])):
        for lib, idx in indexes.items():
            for h in idx.search(q, k=2):
                if (lib, h.passage.pid) not in seen:
                    seen.add((lib, h.passage.pid))
                    meta = idx.meta[h.passage.page_no]
                    ghits.append((lib, h, meta, cite_label(lib, meta)))

    inputs = []
    for bp in parts[:6]:
        lines = [f"{FIELD_LABELS[f]}: " + "; ".join(i["text"] for i in bp.get(f, [])) for f in FIELDS if bp.get(f)]
        inputs.append(f"- {bp['name']}\n  " + "\n  ".join(lines or ["(no findings in the records)"]))
    passages = "\n\n".join(f"[G{i}] ({label})\n{h.passage.text}" for i, (_, h, _, label) in enumerate(ghits, 1))
    user = f"RATING INPUTS FROM THE RECORDS:\n" + "\n".join(inputs) + f"\n\nREFERENCE PASSAGES:\n{passages}"
    try:
        reply = gateway.complete_json("answer", _GUIDE_SYSTEM % spec["intro"], user, max_tokens=3000)
    except Exception as err:  # noqa: BLE001
        out["warnings"].append(f"The model could not prepare the guidance: {err}")
        return out
    if not isinstance(reply, dict):
        out["warnings"].append("The model returned an unexpected format.")
        return out

    def item(text):
        clean, cites, bad = _resolve_guides(str(text), ghits, quote_for)
        if not cites:
            out["uncited"] += 1
        return {"text": clean, "cites": cites, "cited": bool(cites)}

    for bp in reply.get("body_parts") or []:
        if not isinstance(bp, dict):
            continue
        out["body_parts"].append({
            "name": str(bp.get("name") or "").strip()[:60],
            "method": item(bp.get("method") or "") if bp.get("method") else None,
            "tables": [{"name": str(t.get("name") or "")[:120], **item(t.get("use") or "")}
                       for t in bp.get("tables") or [] if isinstance(t, dict) and t.get("name")],
            "measure_at_exam": [item(x) for x in bp.get("measure_at_exam") or []],
            "inputs_present": [str(x)[:200] for x in bp.get("inputs_present") or []],
            "inputs_missing": [str(x)[:200] for x in bp.get("inputs_missing") or []],
            "notes": [item(x) for x in bp.get("notes") or []],
        })
    out["general"] = [item(x) for x in reply.get("general") or []]
    out["reference_pages"] = sorted({label for _, _, _, label in ghits})
    return out


def prepare(case_id: str, settings, tab: str, on_progress=None) -> dict:
    """Build (or reuse) the record findings, then the guidance for this tab, and save it."""
    from .db import get_engine
    from .guides import build_libraries, load_library
    from .rag.service import get_qa
    if on_progress:
        on_progress(0, 3)
    if not all(load_library(settings, lib) for lib in TABS[tab]["libraries"]):
        build_libraries(settings)
    qa = get_qa(get_engine(settings.db_url), case_id, settings)
    if qa.gateway is None:
        raise RuntimeError("The rating worksheet needs a model (IME_PROVIDER=local).")
    if on_progress:
        on_progress(1, 3)
    findings = load(settings, case_id, "rating_findings")
    if findings is None or not findings.get("body_parts"):
        findings = record_findings(qa, settings, case_id)
    if on_progress:
        on_progress(2, 3)
    guidance = guide_guidance(tab, findings, settings, qa.gateway)
    guidance["model"] = qa.gateway.settings.models.get("answer")
    _save(settings, case_id, f"rating_{tab}", guidance)
    if on_progress:
        on_progress(3, 3)
    return guidance
