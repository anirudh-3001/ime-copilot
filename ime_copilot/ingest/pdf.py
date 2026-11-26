"""Split a case PDF into pages and get the text of each page.

Text sources, chosen by IME_OCR:
  azure_di  - Azure AI Document Intelligence (production; HIPAA-eligible under
              the Microsoft BAA). Text, tables, layout, confidence.
  pdftext   - the PDF's own text layer (fast; empty on scanned/faxed pages)
  tesseract - local OCR on every page (development on synthetic data only)
  auto      - text layer first, Tesseract only for pages with no usable text
  textract  - AWS Textract (alternative cloud)

Large files (1,000+ scanned pages):
  - page_range processes a slice first, e.g. (1, 50)
  - Tesseract runs in parallel, one process per CPU
  - cache_dir keeps each page's OCR result, so an interrupted run resumes
    where it stopped instead of starting over
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from pypdf import PdfReader, PdfWriter

from ..schemas import Page

MIN_TEXT_CHARS = 20          # below this a page is treated as scanned
REVIEW_CONFIDENCE = 0.80     # OCR confidence below this goes to human review


def page_count(pdf_path: str | Path) -> int:
    return len(PdfReader(str(pdf_path)).pages)


def resolve_range(page_range: tuple[int, int] | None, total: int) -> tuple[int, int]:
    if page_range is None:
        return 1, total
    start, end = page_range
    start, end = max(1, start), min(total, end)
    if start > end:
        raise ValueError(f"Page range {page_range} is outside 1-{total}.")
    return start, end


def parse_range(text: str | None) -> tuple[int, int] | None:
    """'1-50' -> (1, 50); '7' -> (7, 7); None/'' -> None"""
    if not text:
        return None
    if "-" in text:
        a, b = text.split("-", 1)
        return int(a), int(b)
    return int(text), int(text)


def read_pages(pdf_path: str | Path, ocr_mode: str = "auto", aws_region: str = "us-east-1",
               di_client: Any = None, page_range: tuple[int, int] | None = None,
               workers: int | None = None, cache_dir: str | Path | None = None,
               dpi: int = 300, progress: bool = True,
               on_progress: Any = None) -> tuple[list[Page], dict | None]:
    """on_progress(stage, done, total) is called as pages are read ('reading') and OCR'd ('ocr')."""
    """Returns (pages, layout). layout is only produced by Document Intelligence."""
    pdf_path = Path(pdf_path)

    if ocr_mode == "azure_di":
        from .azure_di import analyze_pdf
        if di_client is None:
            raise ValueError("IME_OCR=azure_di needs a Document Intelligence client (set AZURE_DI_ENDPOINT).")
        return analyze_pdf(str(pdf_path), di_client, page_range=page_range)

    reader = PdfReader(str(pdf_path))
    start, end = resolve_range(page_range, len(reader.pages))
    numbers = list(range(start, end + 1))

    pages: dict[int, Page] = {}
    for i, n in enumerate(numbers, 1):
        if on_progress and (i % 50 == 0 or i == len(numbers)):
            on_progress("reading", i, len(numbers))
        text = ""
        if ocr_mode in ("pdftext", "auto"):
            try:
                text = (reader.pages[n - 1].extract_text() or "").strip()
            except Exception:   # noqa: BLE001 - damaged page streams happen in faxed files
                text = ""
        pages[n] = Page(page_no=n, text=text, text_source="pdftext")

    if ocr_mode == "textract":
        textract = _TextractOCR(aws_region)
        for n in numbers:
            text, conf = textract.page(_single_page_pdf(reader, n - 1))
            pages[n] = Page(page_no=n, text=text, text_source="textract", ocr_confidence=conf)
    elif ocr_mode in ("tesseract", "auto"):
        todo = [n for n in numbers if ocr_mode == "tesseract" or len(pages[n].text) < MIN_TEXT_CHARS]
        if todo and tesseract_available():
            cache = _OcrCache(cache_dir, pdf_path, dpi) if cache_dir else None
            for n, (text, conf) in _ocr_many(pdf_path, todo, workers, cache, dpi, progress,
                                             on_progress).items():
                pages[n] = Page(page_no=n, text=text, text_source="tesseract", ocr_confidence=conf)
        elif todo and progress:
            print(f"[ocr] {len(todo)} scanned pages found but Tesseract/Poppler are not installed; "
                  "they are flagged for review.", file=sys.stderr, flush=True)

    out = []
    for n in numbers:
        p = pages[n]
        p.needs_review = len(p.text) < MIN_TEXT_CHARS or (
            p.ocr_confidence is not None and p.ocr_confidence < REVIEW_CONFIDENCE)
        out.append(p)
    return out, None


# ---------------------------------------------------------------- Tesseract

def tesseract_available() -> bool:
    return bool(shutil.which("tesseract") and shutil.which("pdftoppm"))


def _ocr_many(pdf_path: Path, numbers: list[int], workers: int | None, cache: "_OcrCache | None",
              dpi: int, progress: bool, on_progress: Any = None) -> dict[int, tuple[str, float | None]]:
    results: dict[int, tuple[str, float | None]] = {}
    todo = []
    for n in numbers:
        hit = cache.get(n) if cache else None
        if hit is not None:
            results[n] = hit
        else:
            todo.append(n)
    if progress:
        print(f"[ocr] {len(numbers)} pages need OCR: {len(numbers) - len(todo)} from cache, "
              f"{len(todo)} to run", file=sys.stderr, flush=True)
    if on_progress:
        on_progress("ocr", len(numbers) - len(todo), len(numbers))
    if not todo:
        return results

    workers = workers or max(1, os.cpu_count() or 1)
    started, done = time.monotonic(), 0
    with ThreadPoolExecutor(max_workers=workers) as pool:   # each task runs its own OCR process
        futures = {pool.submit(_tesseract_page, pdf_path, n, dpi): n for n in todo}
        for fut in as_completed(futures):
            n = futures[fut]
            try:
                results[n] = fut.result()
            except Exception as err:  # noqa: BLE001 - one bad page must not stop the file
                results[n] = ("", None)
                if progress:
                    print(f"[ocr] page {n} failed: {err}", file=sys.stderr, flush=True)
            if cache:
                cache.put(n, *results[n])
            done += 1
            if on_progress:
                on_progress("ocr", len(numbers) - len(todo) + done, len(numbers))
            if progress and (done % 25 == 0 or done == len(todo)):
                elapsed = time.monotonic() - started
                rate = elapsed / done
                print(f"[ocr] {done}/{len(todo)} pages  {rate:.1f}s/page  "
                      f"~{rate * (len(todo) - done) / 60:.0f} min left", file=sys.stderr, flush=True)
    return results


def _tesseract_page(pdf_path: Path, page_no: int, dpi: int = 300) -> tuple[str, float | None]:
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "page"
        subprocess.run(
            ["pdftoppm", "-f", str(page_no), "-l", str(page_no), "-r", str(dpi), "-gray", "-png",
             "-singlefile", str(pdf_path), str(prefix)],
            check=True, capture_output=True,
        )
        tsv = subprocess.run(
            ["tesseract", f"{prefix}.png", "stdout", "--psm", "6", "tsv"],
            check=True, capture_output=True, text=True,
            env={**os.environ, "OMP_THREAD_LIMIT": "1"},   # parallelism comes from our pool
        ).stdout
    return _parse_tesseract_tsv(tsv)


def _parse_tesseract_tsv(tsv: str) -> tuple[str, float | None]:
    lines: dict[tuple[str, str, str], list[str]] = {}
    confs: list[float] = []
    for row in tsv.splitlines()[1:]:
        cols = row.split("\t")
        if len(cols) < 12 or cols[0] != "5":       # level 5 = word
            continue
        word, conf = cols[11], float(cols[10])
        if not word.strip():
            continue
        lines.setdefault((cols[2], cols[3], cols[4]), []).append(word)
        if conf >= 0:
            confs.append(conf / 100.0)
    text = "\n".join(" ".join(words) for words in lines.values())
    return text, (sum(confs) / len(confs) if confs else None)


class _OcrCache:
    """One small JSON file per OCR'd page, keyed by a fingerprint of the PDF."""

    def __init__(self, root: str | Path, pdf_path: Path, dpi: int):
        self.dir = Path(root) / f"{_fingerprint(pdf_path)}_dpi{dpi}"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _file(self, n: int) -> Path:
        return self.dir / f"page_{n:05d}.json"

    def get(self, n: int) -> tuple[str, float | None] | None:
        f = self._file(n)
        if not f.exists():
            return None
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            return d["text"], d["conf"]
        except (ValueError, KeyError):
            return None

    def put(self, n: int, text: str, conf: float | None) -> None:
        tmp = self._file(n).with_suffix(".tmp")
        tmp.write_text(json.dumps({"text": text, "conf": conf}), encoding="utf-8")
        tmp.replace(self._file(n))


def _fingerprint(pdf_path: Path) -> str:
    """Size + first and last MB: fast even for very large files."""
    size = pdf_path.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with open(pdf_path, "rb") as fh:
        h.update(fh.read(1 << 20))
        if size > (1 << 20):
            fh.seek(max(0, size - (1 << 20)))
            h.update(fh.read(1 << 20))
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- Textract (AWS alternative)

def _single_page_pdf(reader: PdfReader, index: int) -> bytes:
    writer = PdfWriter()
    writer.add_page(reader.pages[index])
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


class _TextractOCR:
    def __init__(self, region: str):
        import boto3  # optional dependency: pip install ime-copilot[aws]
        self.client = boto3.client("textract", region_name=region)

    def page(self, pdf_bytes: bytes) -> tuple[str, float | None]:
        resp = self.client.detect_document_text(Document={"Bytes": pdf_bytes})
        lines = [b for b in resp.get("Blocks", []) if b.get("BlockType") == "LINE"]
        text = "\n".join(b.get("Text", "") for b in lines)
        conf = (sum(b.get("Confidence", 0) for b in lines) / len(lines) / 100.0) if lines else None
        return text, conf
