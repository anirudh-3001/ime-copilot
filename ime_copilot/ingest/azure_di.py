"""Azure AI Document Intelligence (prebuilt-layout): text, tables and layout per page.

- One analyze call per file (in chunks of up to 2,000 pages, the service's
  per-document limit), not one call per page.
- Keeps page numbers, word-level confidence, rotation and tables.
- Line polygons are returned separately as a layout document, saved to Blob
  Storage so the workbench can highlight a cited line on the original page.
- Authentication: Microsoft Entra ID (managed identity / az login) by default;
  a key only for local development.

HIPAA: Document Intelligence is the HIPAA-eligible step that turns PDF pages
into text. Only that text is sent on to Azure OpenAI; never PDFs or page images.
"""

from __future__ import annotations

from typing import Any

from pypdf import PdfReader

from ..schemas import Page, PageTable

MODEL_ID = "prebuilt-layout"
MAX_PAGES_PER_CALL = 2000
REVIEW_CONFIDENCE = 0.80          # average word confidence below this -> human review
LOW_WORD_CONFIDENCE = 0.50
REVIEW_LOW_WORD_SHARE = 0.20      # >20% of words below 0.5 -> human review


def make_client(endpoint: str, key: str = ""):
    from azure.ai.documentintelligence import DocumentIntelligenceClient  # pip install ime-copilot[azure]
    if key:
        from azure.core.credentials import AzureKeyCredential
        credential = AzureKeyCredential(key)
    else:
        from azure.identity import DefaultAzureCredential
        credential = DefaultAzureCredential()
    return DocumentIntelligenceClient(endpoint=endpoint, credential=credential)


def _request(pdf_bytes: bytes) -> Any:
    try:
        from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
        return AnalyzeDocumentRequest(bytes_source=pdf_bytes)
    except ImportError:            # tests with a fake client
        return {"base64Source": pdf_bytes}


def analyze_pdf(pdf_path: str, client: Any, chunk_size: int = MAX_PAGES_PER_CALL,
                page_range: tuple[int, int] | None = None) -> tuple[list[Page], dict]:
    """Returns (pages, layout). layout = {"pages": [{page_number, width, height, unit, angle, lines}]}."""
    from .pdf import resolve_range

    with open(pdf_path, "rb") as fh:
        data = fh.read()
    first, last = resolve_range(page_range, len(PdfReader(pdf_path).pages))

    pages: list[Page] = []
    layout: dict = {"model": MODEL_ID, "pages": []}
    for start in range(first, last + 1, chunk_size):
        end = min(start + chunk_size - 1, last)
        poller = client.begin_analyze_document(MODEL_ID, _request(data), pages=f"{start}-{end}")
        result = poller.result()
        chunk_pages, chunk_layout = _convert(result)
        pages.extend(chunk_pages)
        layout["pages"].extend(chunk_layout)

    # Pages Document Intelligence returned nothing for (blank or unreadable) still get a row.
    seen = {p.page_no for p in pages}
    for n in range(first, last + 1):
        if n not in seen:
            pages.append(Page(page_no=n, text="", text_source="azure_di", needs_review=True))
    pages.sort(key=lambda p: p.page_no)
    return pages, layout


def _convert(result: Any) -> tuple[list[Page], list[dict]]:
    tables_by_page: dict[int, list[PageTable]] = {}
    for table in getattr(result, "tables", None) or []:
        regions = table.bounding_regions or []
        if not regions:
            continue
        grid = [["" for _ in range(table.column_count)] for _ in range(table.row_count)]
        for cell in table.cells:
            grid[cell.row_index][cell.column_index] = (cell.content or "").strip()
        first = regions[0].page_number
        later = sorted({r.page_number for r in regions[1:] if r.page_number != first})
        tables_by_page.setdefault(first, []).append(PageTable(
            row_count=table.row_count, column_count=table.column_count, cells=grid,
            continues_on_pages=later,
        ))

    pages, layout = [], []
    for p in getattr(result, "pages", None) or []:
        lines = p.lines or []
        words = p.words or []
        text = "\n".join(line.content for line in lines)
        confs = [w.confidence for w in words if w.confidence is not None]
        avg = sum(confs) / len(confs) if confs else None
        low_share = (sum(1 for c in confs if c < LOW_WORD_CONFIDENCE) / len(confs)) if confs else 0.0
        needs_review = (
            len(text.strip()) < 20
            or (avg is not None and avg < REVIEW_CONFIDENCE)
            or low_share > REVIEW_LOW_WORD_SHARE
        )
        pages.append(Page(
            page_no=p.page_number, text=text, text_source="azure_di",
            ocr_confidence=round(avg, 4) if avg is not None else None,
            low_confidence_words=round(low_share, 4), angle=p.angle,
            tables=tables_by_page.get(p.page_number, []), needs_review=needs_review,
        ))
        layout.append({
            "page_number": p.page_number, "width": p.width, "height": p.height,
            "unit": p.unit, "angle": p.angle,
            "lines": [{"content": line.content, "polygon": list(line.polygon or [])} for line in lines],
        })
    return pages, layout
