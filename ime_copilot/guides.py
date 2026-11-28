"""Reference library: the AMA Guides and Colorado rating documents, searchable like a claim file.

Put the PDFs in two folders (any number of files, any names, subfolders allowed):
    <IME_GUIDES_DIR>/AMA/        AMA Guides to the Evaluation of Permanent Impairment (scans are fine)
    <IME_GUIDES_DIR>/Colorado/   Colorado DOWC documents (Level II curriculum, Desk Aid 11 ...)
Then:  python -m ime_copilot guides      (or it builds itself the first time a rating tab is prepared)

- Text comes from the PDF's text layer; scanned pages are OCR'd with column-aware layout,
  because the Guides are printed in two columns.
- Each page keeps its file, PDF page and printed page number (read from the page header or
  footer), so citations read "AMA Guides, p. 87".
- A file that appears in both folders is indexed once (Colorado first).
- Built once and saved under <IME_INDEX_DIR>/_guides/; search vectors are cached like case files.

Copyright: the AMA Guides are the AMA's copyrighted work. This library is built and kept on the
local machine for the licensed user's own reference; the app shows short quotes with page numbers.
Confirm the license before using it beyond that.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pypdf import PdfReader

from .ingest.pdf import _parse_tesseract_tsv, tesseract_available
from .schemas import Page

LIBRARIES = {"colorado": "Colorado", "ama": "AMA"}     # order matters: shared files go to Colorado
LABELS = {"ama": "AMA Guides", "colorado": "Colorado DOWC"}
MIN_TEXT = 200


def guides_root(settings) -> Path:
    return Path(os.path.expanduser(settings.guides_dir))


def library_dir(settings) -> Path:
    return Path(settings.index_dir) / "_guides"


# ----------------------------------------------------------------------------- reading
def _ocr(pdf: Path, page_no: int, dpi: int = 200) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "p"
        subprocess.run(["pdftoppm", "-f", str(page_no), "-l", str(page_no), "-r", str(dpi), "-gray", "-png",
                        "-singlefile", str(pdf), str(prefix)], check=True, capture_output=True)
        tsv = subprocess.run(["tesseract", f"{prefix}.png", "stdout", "--psm", "3", "tsv"], check=True,
                             capture_output=True, text=True, env={**os.environ, "OMP_THREAD_LIMIT": "1"}).stdout
    return _parse_tesseract_tsv(tsv)[0]


_HEAD_EVEN = re.compile(r"^\W{0,3}(\d{1,3})\s+[‘'\"“]?[A-Z][A-Z ,’'&\-]{5,}")
_HEAD_ODD = re.compile(r"[A-Z][A-Z ,’'&\-]{5,}\s+(\d{1,3})\W{0,3}$")
_BARE = re.compile(r"^\W{0,2}(?:page\s+)?(\d{1,3})\W{0,2}$", re.IGNORECASE)


def printed_page(text: str) -> int | None:
    """The page number printed on the page: in a running header ('38 GUIDES TO THE ...',
    'THE VISUAL SYSTEM 163') or alone on a header/footer line."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines[:3]:
        for rx in (_HEAD_EVEN, _HEAD_ODD, _BARE):
            m = rx.search(ln)
            if m:
                return int(m.group(1))
    for ln in lines[-3:]:
        m = _BARE.search(ln)
        if m:
            return int(m.group(1))
    return None


def _fix_page_numbers(pages: list[dict]) -> None:
    """Scanned books are consecutive within a file: the printed number is the PDF page plus a fixed
    offset. Use the most common offset per file to fill pages where no number was read and to fix
    misreads; a read number that its neighbours confirm is kept."""
    from collections import Counter
    by_file: dict[str, list[dict]] = {}
    for d in pages:
        by_file.setdefault(d["file"], []).append(d)
    for items in by_file.values():
        offsets = Counter(d["printed_page"] - d["pdf_page"] for d in items if d["printed_page"])
        if not offsets:
            continue
        offset, votes = offsets.most_common(1)[0]
        if votes < max(3, 0.4 * sum(offsets.values())):
            continue                                   # no clear sequence (e.g. a mixed document)
        read = {d["pdf_page"]: d["printed_page"] for d in items if d["printed_page"]}
        for d in items:
            expected = d["pdf_page"] + offset
            got = d["printed_page"]
            if got == expected:
                continue
            confirmed = got is not None and (read.get(d["pdf_page"] - 1) == got - 1 or read.get(d["pdf_page"] + 1) == got + 1)
            if not confirmed and expected > 0:
                d["printed_page"] = expected
                d["page_estimated"] = True


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:20]


def build_libraries(settings, progress=None, workers: int | None = None) -> dict[str, int]:
    """Read every PDF in the guides folders (OCR when needed). Pages already read are kept."""
    root, out = guides_root(settings), library_dir(settings)
    out.mkdir(parents=True, exist_ok=True)
    seen_hashes: set[str] = set()
    counts = {}
    for name, folder in LIBRARIES.items():
        files = sorted(p for p in (root / folder).rglob("*") if p.suffix.lower() == ".pdf") if (root / folder).exists() else []
        path = out / f"{name}.jsonl"
        old = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    d = json.loads(line)
                    old[(d["hash"], d["pdf_page"])] = d
                except (ValueError, KeyError):
                    pass
        pages: list[dict] = []
        for f in files:
            fh = _file_hash(f)
            if fh in seen_hashes:
                continue
            seen_hashes.add(fh)
            reader = PdfReader(str(f))
            n = len(reader.pages)
            todo = []
            for i in range(1, n + 1):
                if (fh, i) in old:
                    pages.append(old[(fh, i)])
                    continue
                try:
                    text = (reader.pages[i - 1].extract_text() or "").strip()
                except Exception:  # noqa: BLE001
                    text = ""
                if len(text) >= MIN_TEXT or not tesseract_available():
                    pages.append({"hash": fh, "file": f.name, "pdf_page": i, "text": text, "ocr": False})
                else:
                    todo.append(i)
            if todo:
                if progress:
                    progress(f"{folder}/{f.name}: OCR of {len(todo)} pages")
                with ThreadPoolExecutor(max_workers=workers or max(1, os.cpu_count() or 1)) as pool:
                    for i, text in zip(todo, pool.map(lambda k: _ocr(f, k), todo)):
                        pages.append({"hash": fh, "file": f.name, "pdf_page": i, "text": text, "ocr": True})
        pages.sort(key=lambda d: (d["file"], d["pdf_page"]))
        for gid, d in enumerate(pages, 1):
            d["gid"] = gid
            d["printed_page"] = printed_page(d["text"])
        _fix_page_numbers(pages)
        with path.open("w", encoding="utf-8") as fo:
            for d in pages:
                fo.write(json.dumps(d) + "\n")
        counts[name] = len(pages)
        if progress:
            progress(f"{folder}: {len(pages)} pages from {len(files)} file(s)")
    return counts


def load_library(settings, name: str) -> list[dict]:
    path = library_dir(settings) / f"{name}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def cite_label(name: str, page: dict) -> str:
    if name == "ama" and page.get("printed_page"):
        return f"AMA Guides 3rd Ed. Rev., p. {page['printed_page']}"
    if page.get("printed_page"):
        return f"{Path(page['file']).stem}, p. {page['printed_page']}"
    return f"{Path(page['file']).stem}, PDF p. {page['pdf_page']}"


# ----------------------------------------------------------------------------- search
class GuideIndex:
    """Search over one library; hits carry the guide page's metadata."""

    def __init__(self, name: str, pages: list[dict], embedder=None, vector_cache=None):
        from .rag.index import CaseIndex
        self.name, self.meta = name, {p["gid"]: p for p in pages}
        self.pages = [Page(page_no=p["gid"], text=p["text"]) for p in pages]
        self.pages_by_no = {p.page_no: p for p in self.pages}
        try:
            self.index = CaseIndex.build(self.pages, embedder=embedder, vector_cache=vector_cache)
        except Exception:  # noqa: BLE001 - keyword search still works
            self.index = CaseIndex.build(self.pages)

    def search(self, query: str, k: int = 6):
        return self.index.search(query, k=k)


_INDEXES: dict[str, GuideIndex] = {}
_LOCK = threading.Lock()


def get_guide_index(settings, name: str, gateway=None) -> GuideIndex | None:
    from .rag.store import DiskVectorCache, index_path
    with _LOCK:
        if name not in _INDEXES:
            pages = load_library(settings, name)
            if not pages:
                return None
            embedder, cache = None, None
            if gateway is not None and gateway.has_model("embed"):
                embedder = gateway.embed
                cache = DiskVectorCache(index_path(settings.index_dir, f"_guides_{name}", settings.models["embed"]))
            _INDEXES[name] = GuideIndex(name, pages, embedder, cache)
        return _INDEXES[name]


def forget_indexes() -> None:
    with _LOCK:
        _INDEXES.clear()


def main(settings) -> int:
    root = guides_root(settings)
    print(f"Guides folder: {root}", file=sys.stderr)
    for name, folder in LIBRARIES.items():
        print(f"  {folder}/: {'found' if (root / folder).exists() else 'missing'}", file=sys.stderr)
    counts = build_libraries(settings, progress=lambda m: print("  " + m, file=sys.stderr, flush=True))
    forget_indexes()
    from .gateway import Gateway
    gw = Gateway(settings) if settings.provider != "none" else None
    for name in LIBRARIES:
        if counts.get(name):
            print(f"Indexing {LABELS[name]} for search...", file=sys.stderr, flush=True)
            get_guide_index(settings, name, gw)
    print(f"Done: {counts}", file=sys.stderr)
    return 0
