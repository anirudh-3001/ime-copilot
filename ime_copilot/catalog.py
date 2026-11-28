"""Document catalog: one careful read of every page, so the summary can list what is in the file.

For each page the extraction model notes: document type, date, providers, facility,
procedures (and whether they are surgery), imaging studies with impressions, bill lines,
and pain level. Every item is checked against the page text, and anything that isn't
on the page is dropped:
  - a date must appear on the page (any format: 02/21/2024, 2-21-24, February 21, 2024)
  - a person or facility name: its words must appear on the page
  - a procedure or imaging study: most of its words (or its CPT code) must appear on the page
  - a bill line: its CPT code, amount and date must all appear on the page
Results are cached per page (data/index/<case>/catalog.jsonl), so a long run that stops
continues where it left off, and a re-run only reads pages whose text changed.

From the page entries this module builds the lists shown in the summary: documents,
providers (what they did, dates, billed amounts), dates of service, procedures with cost,
imaging with dates, surgeries, and a before/after-surgery comparison.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .extract.citations import normalize
from .schemas import Citation, Page, ServiceRow

DOC_TYPES = ["office visit", "therapy note", "chiropractic note", "imaging report", "procedure report",
             "operative report", "lab result", "emergency/hospital record", "bill", "IME/legal document",
             "intake/questionnaire", "correspondence", "other"]

CATALOG_SYSTEM = """You catalog one page of a medical claim file. Read the page and reply with JSON only:
{"doc_type": "<one of: %s>",
 "title": "<document title or heading as written, short>",
 "date": "<the date of service, visit or report on this page, copied as written; empty if none>",
 "providers": [{"name": "<person's name as written>", "role": "<e.g. treating physician, surgeon, radiologist, chiropractor, physical therapist, nurse practitioner>"}],
 "facility": "<clinic, hospital or practice name as written; empty if none>",
 "procedures": [{"name": "<procedure or treatment performed on this date>", "cpt": "<CPT code if shown>", "surgery": <true only for an operation in an operating room or surgery center, false otherwise>}],
 "imaging": [{"study": "<e.g. MRI lumbar spine, X-ray cervical spine>", "impression": "<the impression or main finding, short>"}],
 "bill_lines": [{"date": "<as written>", "cpt": "<code>", "description": "<short>", "units": <number>, "amount": <billed amount, number>, "provider": "<billing or rendering provider if shown>"}],
 "pain": "<pain level or main complaint if stated, short; empty if none>",
 "note": "<one short sentence: what this page records>"}
Rules: use only what is on this page; copy names, codes and dates exactly as written; use empty
strings or empty lists when something is not on the page. Do not include the patient as a provider.
Signatures, ordering and referring doctors count as providers.""" % ", ".join(DOC_TYPES)

MAX_PAGE_WORDS = 900
_STOP = set("dr md do dc pt dpt np pa pac rn lpn ma mbbs phd mr mrs ms jr sr ii iii iv facs faaos the of and "
            "llc inc pc pllc pa ltd group center centre clinic".split())


# ----------------------------------------------------------------------------- helpers
def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize(text or ""))


def _page_words(page: Page) -> set[str]:
    return set(_words(page.text))


def _name_tokens(name: str) -> list[str]:
    return [w for w in _words(name) if w not in _STOP and len(w) >= 2 and not w.isdigit()]


def _on_page(text: str, page_words: set[str], share: float = 1.0) -> bool:
    toks = [w for w in _words(text) if len(w) >= 2]
    if not toks:
        return False
    hit = sum(1 for w in toks if w in page_words)
    return hit >= max(1, share * len(toks))


def _iso(date_text: str, page_text: str) -> str | None:
    """The date as ISO (YYYY-MM-DD) if it parses and the same date appears on the page."""
    from .rag.qa import _dates
    found = _dates(date_text or "")
    if not found:
        return None
    key = sorted(found)[0]
    if key not in _dates(page_text):
        return None
    y, m, d = key
    if not (1900 < y < 2100 and 1 <= m <= 12 and 1 <= d <= 31):
        return None
    return f"{y:04d}-{m:02d}-{d:02d}"


def _amount(value) -> float | None:
    try:
        return round(float(str(value).replace("$", "").replace(",", "").strip()), 2)
    except (TypeError, ValueError):
        return None


def _amount_on_page(amount: float, page_text: str) -> bool:
    digits = re.sub(r"[^0-9]", "", f"{amount:.2f}")
    whole = re.sub(r"[^0-9]", "", f"{amount:.0f}")
    page_digits = re.sub(r"[^0-9]", " ", page_text)
    runs = set(page_digits.split()) | {re.sub(r"[^0-9]", "", m) for m in
                                       re.findall(r"\d{1,3}(?:,\d{3})*(?:\.\d{2})?", page_text)}
    return digits in runs or (amount == int(amount) and whole in runs)


# ----------------------------------------------------------------------------- one page
def verify_entry(raw: dict, page: Page) -> dict:
    """Keep only what can be found on the page."""
    text, words = page.text, _page_words(page)
    out = {"page": page.page_no, "doc_type": "other", "title": "", "date": None, "providers": [],
           "facility": "", "procedures": [], "imaging": [], "bill_lines": [], "pain": "", "note": "",
           "dropped": 0}
    if not isinstance(raw, dict):
        return out
    dt = str(raw.get("doc_type") or "other").strip().lower()
    out["doc_type"] = next((t for t in DOC_TYPES if t.lower() == dt), "other")
    title = str(raw.get("title") or "").strip()[:120]
    out["title"] = title if title and _on_page(title, words, 0.6) else ""
    out["date"] = _iso(str(raw.get("date") or ""), text)

    for p in raw.get("providers") or []:
        if not isinstance(p, dict):
            continue
        name = " ".join(str(p.get("name") or "").split())[:80]
        toks = _name_tokens(name)
        if len(toks) >= 1 and all(t in words for t in toks) and any(len(t) >= 3 for t in toks):
            out["providers"].append({"name": name, "role": str(p.get("role") or "").strip()[:60]})
        elif name:
            out["dropped"] += 1
    fac = " ".join(str(raw.get("facility") or "").split())[:100]
    out["facility"] = fac if fac and _on_page(fac, words, 0.7) else ""

    for pr in raw.get("procedures") or []:
        if not isinstance(pr, dict):
            continue
        name = " ".join(str(pr.get("name") or "").split())[:120]
        cpt = str(pr.get("cpt") or "").strip().upper()
        cpt = cpt if re.fullmatch(r"\d{4}[0-9FTU]|[A-V]\d{4}", cpt) and cpt.lower() in normalize(text) else ""
        if name and (_on_page(name, words, 0.6) or cpt):
            surgery = bool(pr.get("surgery")) and bool(re.search(
                r"operative|surgery|surgical|anesthesia|incision|arthroscop|fusion|laminectomy|discectomy|"
                r"repair|replacement|reconstruction", text, re.IGNORECASE))
            out["procedures"].append({"name": name, "cpt": cpt, "surgery": surgery})
        elif name:
            out["dropped"] += 1

    for im in raw.get("imaging") or []:
        if not isinstance(im, dict):
            continue
        study = " ".join(str(im.get("study") or "").split())[:80]
        if study and _on_page(study, words, 0.5) and re.search(
                r"\b(mri|m\.r\.i|ct|x-?ray|radiograph|ultrasound|emg|ncs|nerve conduction|bone scan|"
                r"fluoroscop|mammogra|dexa|pet)\b", study + " " + text, re.IGNORECASE):
            impression = " ".join(str(im.get("impression") or "").split())[:300]
            out["imaging"].append({"study": study,
                                   "impression": impression if _on_page(impression, words, 0.6) else ""})
        elif study:
            out["dropped"] += 1

    for bl in raw.get("bill_lines") or []:
        if not isinstance(bl, dict):
            continue
        cpt = str(bl.get("cpt") or "").strip().upper()
        amount = _amount(bl.get("amount"))
        date = _iso(str(bl.get("date") or ""), text) or out["date"]
        if (re.fullmatch(r"\d{4}[0-9FTU]|[A-V]\d{4}", cpt) and cpt.lower() in normalize(text)
                and amount is not None and amount > 0 and _amount_on_page(amount, text) and date):
            try:
                units = max(1, int(float(bl.get("units") or 1)))
            except (TypeError, ValueError):
                units = 1
            prov = " ".join(str(bl.get("provider") or "").split())[:80]
            out["bill_lines"].append({"date": date, "cpt": cpt, "description": str(bl.get("description") or "")[:80],
                                      "units": units, "amount": amount,
                                      "provider": prov if prov and _on_page(prov, words, 0.7) else ""})
        elif cpt or amount:
            out["dropped"] += 1

    pain = " ".join(str(raw.get("pain") or "").split())[:120]
    out["pain"] = pain if pain and _on_page(pain, words, 0.5) else ""
    out["note"] = " ".join(str(raw.get("note") or "").split())[:200]
    return out


def catalog_path(index_dir: str | Path, case_id: str) -> Path:
    return Path(index_dir) / case_id / "catalog.jsonl"


def _page_key(page: Page, model: str) -> str:
    return hashlib.sha256(f"{model}\n{page.text}".encode()).hexdigest()[:20]


def load_catalog(index_dir: str | Path, case_id: str) -> dict[int, dict]:
    path = catalog_path(index_dir, case_id)
    entries: dict[int, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
                entries[e["page"]] = e
            except (ValueError, KeyError):
                continue
    return entries


def build_catalog(pages: list[Page], gateway, index_dir: str | Path, case_id: str,
                  on_progress=None, workers: int = 1) -> dict[int, dict]:
    """Read every (non-duplicate, non-empty) page once with the extraction model. Resumable."""
    model = gateway.settings.models.get("extract") or gateway.settings.models.get("answer") or ""
    task = "extract" if gateway.has_model("extract") else "answer"
    done = load_catalog(index_dir, case_id)
    todo = [p for p in pages if p.duplicate_of is None and len(p.text.strip()) >= 40
            and done.get(p.page_no, {}).get("key") != _page_key(p, model)]
    total = sum(1 for p in pages if p.duplicate_of is None and len(p.text.strip()) >= 40)
    count = total - len(todo)
    if on_progress:
        on_progress(count, total)
    path = catalog_path(index_dir, case_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    def read(page: Page) -> dict:
        words = page.text.split()
        body = " ".join(words[:MAX_PAGE_WORDS]) if len(words) > MAX_PAGE_WORDS else page.text
        try:
            raw = gateway.complete_json(task, CATALOG_SYSTEM, f"PAGE {page.page_no}:\n{body}",
                                        max_tokens=900, attempts=2)
        except Exception:  # noqa: BLE001 - one unreadable page must not stop the catalog
            raw = None
        entry = verify_entry(raw, page)
        entry["key"] = _page_key(page, model)
        entry["failed"] = raw is None
        return entry

    with path.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for entry in pool.map(read, todo):
            f.write(json.dumps(entry) + "\n")
            f.flush()
            done[entry["page"]] = entry
            count += 1
            if on_progress:
                on_progress(count, total)
    return done


# ----------------------------------------------------------------------------- lists for the summary
def _provider_key(name: str) -> frozenset:
    return frozenset(_name_tokens(name))


def _merge_names(names: Counter) -> dict[frozenset, str]:
    """Group spellings of the same person ('Dr. A. Smith', 'Alan Smith MD'): key -> display name."""
    keys = sorted({_provider_key(n) for n in names if _provider_key(n)}, key=len, reverse=True)
    canon: dict[frozenset, frozenset] = {}
    for k in keys:
        target = next((c for c in canon.values() if k <= c and len(k) >= 2), None)
        canon[k] = target or k
    display: dict[frozenset, Counter] = defaultdict(Counter)
    for n, c in names.items():
        k = _provider_key(n)
        if k:
            display[canon[k]][n] += c
    out = {}
    for k, target in canon.items():
        out[k] = max(display[target].items(), key=lambda kv: (kv[1], len(kv[0])))[0]
    return out


def bill_rows(entries: dict[int, dict], pages_by_no: dict[int, Page], existing: list[ServiceRow]) -> list[ServiceRow]:
    """Bill lines found by the catalog that the rule-based bill reader missed, as ledger rows."""
    have = {(s.date_of_service, s.cpt, round(s.billed_amount or 0, 2)) for s in existing}
    rows = []
    for e in entries.values():
        page = pages_by_no.get(e["page"])
        if page is None:
            continue
        for bl in e.get("bill_lines") or []:
            key = (bl["date"], bl["cpt"], round(bl["amount"], 2))
            if key in have:
                continue
            line = next((ln.strip() for ln in page.text.splitlines() if bl["cpt"] in ln.upper()), "")
            if not line:
                continue
            have.add(key)
            rows.append(ServiceRow(date_of_service=bl["date"], cpt=bl["cpt"], units=bl["units"],
                                   billed_amount=bl["amount"], provider=bl.get("provider") or e.get("facility") or None,
                                   citation=Citation(page_no=e["page"], quote=line[:200]),
                                   confidence=0.8, method="llm"))
    return rows


def catalog_views(entries: dict[int, dict], services: list[ServiceRow], date_of_injury: str | None = None) -> dict:
    """Documents, providers, dates of service, procedures, imaging, surgeries, before/after."""
    pages = sorted(entries.values(), key=lambda e: e["page"])

    # documents: consecutive pages that belong together
    docs: list[dict] = []
    for e in pages:
        prov = e["providers"][0]["name"] if e["providers"] else ""
        cur = docs[-1] if docs else None
        same = (cur is not None and e["page"] == cur["pages"][-1] + 1 and e["doc_type"] == cur["doc_type"]
                and (not e["date"] or not cur["date"] or e["date"] == cur["date"])
                and (not prov or not cur["provider"] or _provider_key(prov) == _provider_key(cur["provider"])))
        if same:
            cur["pages"].append(e["page"])
            cur["date"] = cur["date"] or e["date"]
            cur["provider"] = cur["provider"] or prov
            cur["facility"] = cur["facility"] or e["facility"]
            cur["title"] = cur["title"] or e["title"]
        else:
            docs.append({"doc_type": e["doc_type"], "title": e["title"], "date": e["date"], "provider": prov,
                         "facility": e["facility"], "pages": [e["page"]], "note": e["note"]})

    # providers
    names = Counter(p["name"] for e in pages for p in e["providers"])
    display = _merge_names(names)
    prov: dict[str, dict] = {}

    def P(name: str) -> dict | None:
        k = _provider_key(name)
        if not k or k not in display:
            return None
        d = display[k]
        return prov.setdefault(d, {"name": d, "roles": Counter(), "facilities": Counter(), "dates": set(),
                                   "pages": set(), "performed": Counter(), "billed": 0.0, "bill_lines": 0})
    for e in pages:
        for p in e["providers"]:
            rec = P(p["name"])
            if rec is None:
                continue
            if p["role"]:
                rec["roles"][p["role"].lower()] += 1
            if e["facility"]:
                rec["facilities"][e["facility"]] += 1
            if e["date"]:
                rec["dates"].add(e["date"])
            rec["pages"].add(e["page"])
            for pr in e["procedures"]:
                rec["performed"][pr["name"]] += 1
            for im in e["imaging"]:
                rec["performed"][im["study"]] += 1

    # billing (ledger rows from the rule-based reader + catalog bill lines) attributed by provider name
    all_rows = list(services)
    unattributed = 0.0
    billers: dict[str, dict] = {}
    for s in all_rows:
        rec = P(s.provider) if s.provider else None
        if rec is not None:
            rec["billed"] += s.billed_amount or 0
            rec["bill_lines"] += 1
        if s.provider:                       # every bill line is also counted under the name on the bill
            name = " ".join(s.provider.split())
            b = billers.setdefault(normalize(name), {"name": name, "billed": 0.0, "lines": 0, "dates": set(),
                                                     "codes": Counter(), "pages": set()})
            b["billed"] += s.billed_amount or 0
            b["lines"] += 1
            b["dates"].add(s.date_of_service)
            b["codes"][s.cpt] += 1
            b["pages"].add(s.citation.page_no)
        else:
            unattributed += s.billed_amount or 0
    billed_by = [{"name": b["name"], "billed": round(b["billed"], 2), "lines": b["lines"],
                  "first_date": min(b["dates"]), "last_date": max(b["dates"]),
                  "codes": [c for c, _ in b["codes"].most_common(6)], "pages": sorted(b["pages"])[:10]}
                 for b in billers.values()]
    billed_by.sort(key=lambda r: -r["billed"])
    providers = []
    for rec in prov.values():
        dates = sorted(rec["dates"])
        providers.append({"name": rec["name"], "role": rec["roles"].most_common(1)[0][0] if rec["roles"] else "",
                          "facility": rec["facilities"].most_common(1)[0][0] if rec["facilities"] else "",
                          "first_date": dates[0] if dates else None, "last_date": dates[-1] if dates else None,
                          "visits": len(dates), "pages": sorted(rec["pages"])[:40],
                          "performed": [n for n, _ in rec["performed"].most_common(6)],
                          "billed": round(rec["billed"], 2), "bill_lines": rec["bill_lines"]})
    providers.sort(key=lambda r: (-r["visits"], -len(r["pages"]), r["name"]))

    # dates of service
    by_date: dict[str, dict] = {}
    for e in pages:
        if not e["date"] or e["doc_type"] in ("bill", "IME/legal document", "correspondence"):
            continue
        d = by_date.setdefault(e["date"], {"date": e["date"], "providers": set(), "types": set(),
                                           "facilities": set(), "pages": []})
        d["providers"].update(p["name"] for p in e["providers"])
        d["types"].add(e["doc_type"])
        if e["facility"]:
            d["facilities"].add(e["facility"])
        d["pages"].append(e["page"])
    dates_of_service = [{"date": d["date"], "providers": sorted(d["providers"])[:4], "types": sorted(d["types"]),
                         "facilities": sorted(d["facilities"])[:2], "pages": d["pages"][:10]}
                        for d in sorted(by_date.values(), key=lambda x: x["date"])]

    # procedures with cost (bill line on the same date with the same code; else the only line that day)
    by_day_code: dict[tuple, float] = defaultdict(float)
    by_day: dict[str, list] = defaultdict(list)
    for s in all_rows:
        by_day_code[(s.date_of_service, s.cpt)] += s.billed_amount or 0
        by_day[s.date_of_service].append(s)
    procedures, seen = [], set()
    for e in pages:
        for pr in e["procedures"]:
            key = (e["date"], normalize(pr["name"]))
            if key in seen:
                continue
            seen.add(key)
            cost, how = None, ""
            if e["date"] and pr["cpt"] and (e["date"], pr["cpt"]) in by_day_code:
                cost, how = round(by_day_code[(e["date"], pr["cpt"])], 2), "same date and code"
            elif e["date"] and len(by_day.get(e["date"], [])) == 1:
                only = by_day[e["date"]][0]
                cost, how = only.billed_amount, f"only bill line that date (CPT {only.cpt})"
            procedures.append({"date": e["date"], "name": pr["name"], "cpt": pr["cpt"], "surgery": pr["surgery"],
                               "provider": e["providers"][0]["name"] if e["providers"] else "",
                               "facility": e["facility"], "page": e["page"], "cost": cost, "cost_basis": how})
    procedures.sort(key=lambda r: (r["date"] or "9999", r["page"]))

    imaging, seen = [], set()
    for e in pages:
        for im in e["imaging"]:
            key = (e["date"], normalize(im["study"]))
            if key in seen:
                continue
            seen.add(key)
            imaging.append({"date": e["date"], "study": im["study"], "impression": im["impression"],
                            "facility": e["facility"],
                            "provider": e["providers"][0]["name"] if e["providers"] else "", "page": e["page"]})
    imaging.sort(key=lambda r: (r["date"] or "9999", r["page"]))

    surgeries = [p for p in procedures if p["surgery"] and p["date"]]
    before_after = _before_after(pages, all_rows, surgeries, date_of_injury) if surgeries else None

    return {"documents": docs, "providers": providers, "billed_by": billed_by, "dates_of_service": dates_of_service,
            "procedures": procedures, "imaging": imaging, "surgeries": surgeries, "before_after": before_after,
            "unattributed_billed": round(unattributed, 2),
            "pages_read": len(pages), "pages_failed": sum(1 for e in pages if e.get("failed")),
            "items_dropped": sum(e.get("dropped", 0) for e in pages)}


def _before_after(pages: list[dict], rows: list[ServiceRow], surgeries: list[dict], doi: str | None) -> dict:
    first = min(surgeries, key=lambda s: s["date"])
    cut = first["date"]

    def side(entries, bills):
        dates = sorted({e["date"] for e in entries if e["date"]})
        types = Counter(e["doc_type"] for e in entries)
        provs = Counter(p["name"] for e in entries for p in e["providers"])
        pains = [{"date": e["date"], "pain": e["pain"], "page": e["page"]} for e in entries if e["pain"]]
        return {"first_date": dates[0] if dates else None, "last_date": dates[-1] if dates else None,
                "visits": len(dates), "providers": [n for n, _ in provs.most_common(6)],
                "treatment": [f"{t} ({n})" for t, n in types.most_common(6)],
                "procedures": sorted({p["name"] for e in entries for p in e["procedures"]})[:10],
                "billed": round(sum(b.billed_amount or 0 for b in bills), 2),
                "pain_notes": (pains[:3] + pains[-3:]) if len(pains) > 6 else pains}
    dated = [e for e in pages if e["date"] and (not doi or e["date"] >= doi)
             and e["doc_type"] not in ("bill", "IME/legal document", "correspondence")]
    before = [e for e in dated if e["date"] < cut]
    after = [e for e in dated if e["date"] > cut]
    return {"surgery": first, "before": side(before, [r for r in rows if r.date_of_service < cut
                                                      and (not doi or r.date_of_service >= doi)]),
            "after": side(after, [r for r in rows if r.date_of_service > cut])}


def timeline_facts(views: dict, entries: dict[int, dict], pages_by_no: dict[int, Page], limit: int = 90) -> list[dict]:
    """Compact dated timeline for the summary model, as citable facts with verified quotes."""
    from .rag.qa import quote_for
    facts = []
    for d in views["dates_of_service"]:
        bits = [d["date"], "/".join(d["types"])]
        if d["providers"]:
            bits.append(", ".join(d["providers"][:2]))
        if d["facilities"]:
            bits.append(d["facilities"][0])
        extra = []
        for pno in d["pages"][:3]:
            e = entries.get(pno, {})
            extra += [p["name"] + (" (surgery)" if p["surgery"] else "") for p in e.get("procedures", [])]
            extra += [f"{i['study']}: {i['impression']}" if i["impression"] else i["study"] for i in e.get("imaging", [])]
            if e.get("pain"):
                extra.append("pain: " + e["pain"])
        if extra:
            bits.append("; ".join(dict.fromkeys(extra))[:220])
        text = " · ".join(bits)
        page = pages_by_no.get(d["pages"][0])
        quote = quote_for(page.text, text) if page else ""
        facts.append({"text": "Encounter " + text, "sources": [(d["pages"][0], quote)] if quote else []})
    if len(facts) > limit:                       # keep the start, the end, and every encounter with content
        rich = [f for f in facts if ";" in f["text"] or "surgery" in f["text"]]
        keep = facts[:15] + [f for f in rich if f not in facts[:15] and f not in facts[-15:]][: limit - 30] + facts[-15:]
        facts = keep
    if views["surgeries"]:
        s = views["surgeries"][0]
        facts.append({"text": f"First surgery: {s['name']} on {s['date']}"
                              + (f" by {s['provider']}" if s["provider"] else ""),
                      "sources": [(s["page"], quote_for(pages_by_no[s['page']].text, s["name"]))]
                      if s["page"] in pages_by_no else []})
    return facts


def catalog_facts(views: dict, pages_by_no: dict[int, Page]) -> list[dict]:
    """Whole-file facts from the catalog for every question: encounter span, providers, surgeries, imaging."""
    from .rag.qa import quote_for
    facts = []

    def src(page_no: int, text: str):
        page = pages_by_no.get(page_no)
        q = quote_for(page.text, text) if page else ""
        return [(page_no, q)] if q else []
    dos = views["dates_of_service"]
    if dos:
        first, last = dos[0], dos[-1]
        facts.append({"text": f"Dated encounters in the records (all providers): first {first['date']}, most recent "
                              f"{last['date']}; {len(dos)} dates of service",
                      "sources": src(first["pages"][0], first["date"]) + src(last["pages"][0], last["date"])})
    for p in views["providers"][:10]:
        what = f"; performed: {', '.join(p['performed'][:3])}" if p["performed"] else ""
        span = f", {p['first_date']} to {p['last_date']}, {p['visits']} dates" if p["first_date"] else ""
        cost = f"; billed ${p['billed']:,.2f}" if p["billed"] else ""
        facts.append({"text": f"Provider {p['name']}" + (f" ({p['role']}" + (f", {p['facility']}" if p['facility'] else "") + ")" if p["role"] else "")
                              + span + what + cost,
                      "sources": src(p["pages"][0], p["name"]) if p["pages"] else []})
    for s in views["surgeries"][:5]:
        facts.append({"text": f"Surgery: {s['name']} on {s['date']}" + (f" by {s['provider']}" if s["provider"] else "")
                              + (f" at {s['facility']}" if s["facility"] else ""), "sources": src(s["page"], s["name"])})
    for im in views["imaging"][:10]:
        facts.append({"text": f"Imaging: {im['study']} on {im['date'] or 'date not stated'}"
                              + (f": {im['impression']}" if im["impression"] else ""), "sources": src(im["page"], im["study"])})
    return facts
