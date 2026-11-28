# IME Copilot

Ask questions about a 1,500–2,000 page claim file and get an **answer**, the **related CPT codes** and the **page numbers** it came from. Everything can run on a Mac, with nothing sent to the internet.

## Start here: Mac with Ollama (10 minutes)

You need: **Ollama** with `gemma4:26b-a4b` and `qwen3-embedding:0.6b` pulled, plus **Python 3.12**. For scanned pages, also install `brew install tesseract poppler`.

In VS Code, open this folder, open the terminal (*View → Terminal*), and run:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # already set up for the local Mac models
pytest -q                     # all tests should pass (1 skips until Tesseract is installed)
python -m ime_copilot check   # checks Ollama answers + search model
```

**Ask questions in the terminal:**
```bash
python -m ime_copilot synth                                  # makes a fake 12-page test case
python -m ime_copilot ask data/synthetic/synthetic_001.pdf   # then type questions, one per line
python -m ime_copilot ask data/MY_FILE.pdf "What physical therapy was billed?"
```
The first time, a PDF is read and OCR'd, which can take a while for 2,000 pages. After that, questions are quick.

**Or use the web page (best for demos):**
```bash
python -m ime_copilot serve       # then open http://127.0.0.1:8000
```
Drop a PDF on the left. A progress card shows each step (reading, OCR, analysis, search index) with time left. Then ask questions: every answer has clickable page numbers that open the page with the quote highlighted, a CPT table (code, type, billed lines, units, total, dates, pages) and a badge showing whether the answer was verified against the file.

**Pinned case summary.** Every file gets a summary pinned at the top of its page, with three tabs:
- *Summary*: the model's own understanding of the case, in its own structure: what happened, the course of treatment, current status, and what the examiner should watch for (gaps, inconsistencies, prior injuries). It reads a fact sheet from every bill plus about 24 key passages; every statement links to its page. Written automatically after upload.
- *Billing*: totals by type and by CPT code (lines, units, amounts, dates, pages) and the top billing providers.
- *Case & file*: patient, DOB, claim number, date of injury, employer, insurer, the IME referral questions, diagnosis codes, and pages that were hard to read.

Below the written summary, the **document catalog** lists: surgeries (with cost), a **before and after surgery** comparison (visits, providers, treatment, billed amounts, pain notes), providers (role, facility, dates, visits, what they did, billed), billed amounts by provider or facility, procedures (date, CPT, provider, cost), imaging such as MRI with dates and impressions, all dates of service, and every document in the file with its page range. The catalog comes from one read of every page by the model (about 5–8 seconds per page, once per file, resumable); every item is checked against its page and dropped if it isn't there. It also adds bill lines from bill layouts the rule-based reader doesn't know, and gives the Q&A whole-file facts (all encounter dates, providers, surgeries, imaging).

*Print* saves it as a PDF.

**AMA and Colorado rating tabs (draft for the rating physician).** Put the reference PDFs in `IME_GUIDES_DIR` (`AMA/` and `Colorado/` folders; scans are fine) and run `python -m ime_copilot guides` once to read and index them (scanned pages are OCR'd; citations use the printed page numbers). Each tab, on request, shows: (1) rating inputs from the records by body part (diagnoses, surgery, imaging, recorded range of motion, neurological findings, MMI statements, earlier ratings, prior injuries, restrictions), each with its page; (2) how to rate it: the sections and tables that apply, what to measure at the exam, what is present or missing, and cautions (Colorado: the DOWC Level II curriculum and Desk Aid 11 rules), each linked to a Guides page you can open; (3) a combined-values calculator (A + B × (100 − A) / 100, largest first). It never gives the rating itself. The AMA Guides are the AMA's copyrighted work: keep them on the local machine and confirm the license before wider use.

**Finding a patient's file.** The case list shows the patient name found in each file, with date of injury, claim number and the original file name. Type in the search box to filter by name, claim number, date of birth, employer or file name. Click the ✎ pencil to change a label (the PDF itself is never renamed). Files processed before this version show a *Write summary* button.

**Big files are processed once.** OCR results are cached in `data/ocr_cache` and search vectors in `data/index/<case>`. If processing stops (Mac sleeps, Ollama closes), running it again continues where it stopped. Restarting the server or `ask` afterwards loads the case in seconds.

**Check accuracy on the test questions:**
```bash
python -m ime_copilot eval-qa     # right page found + CPT recall, per question
```

**How a question is answered:** (1) the model first turns the question into a few targeted searches (and resolves follow-ups like "what did it show?" using the last few questions); (2) the file is searched with those plus the CPT codes of the question's topic; (3) the model answers from the best passages (each labelled with its page type), a fact sheet from every bill in the file (date of injury, first and last dates of service, totals by treatment type) and the case summary for background, connecting facts across pages like an analyst.

**How answers stay trustworthy:** the model marks each fact with the passage or fact it came from, like [3] or [F2]; the system turns those into page numbers and takes the quote from the page itself, so pages and quotes can't be made up. Dates, dollar amounts and CPT codes in the answer are checked against the cited text and flagged if missing, and every CPT code must exist in the document. If nothing can be verified, you get the best-matching passages instead, clearly labelled.

**Privacy:** keep FileVault on, keep patient PDFs out of iCloud and Drive folders, and never use Ollama models ending in `-cloud`.

---

## Phase 1 pipeline details (Azure-ready)

Upload a claim-file PDF and get it back **page by page**: text, tables, layout, OCR confidence, page type and duplicates. You also get a first **treatment ledger** where every row cites its page, and the IME company's **questions** word for word.

```
PDF upload (API or CLI)
  -> Azure Blob Storage: cases/<id>/original.pdf (never modified)
  -> Azure AI Document Intelligence (prebuilt-layout): text, tables, word confidence, line polygons
       -> cases/<id>/layout.json  (for highlighting cited lines in the workbench later)
  -> low-confidence / blank pages flagged for human review
  -> near-duplicate detection, page classification
  -> case details: patient name, DOB, claim no., date of injury, employer, insurer
       (voted across pages; SSN kept as last 4 digits only)
  -> ledger: services (CPT, units, charges), ICD-10 diagnoses, case questions
       every row = page number + exact quote, verified (cite or omit)
  -> PostgreSQL (SQLite locally) + cases/<id>/ledger.json
```

Without any Azure resources, the same pipeline runs locally on synthetic data, using the PDF text layer and Tesseract instead of Document Intelligence. Switching to Azure is configuration only.

## Quick start (Windows PowerShell, inside `ime-copilot`)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env
python -m ime_copilot synth                  # fake 12-page case + ground truth
python -m ime_copilot eval data\synthetic\synthetic_001.pdf data\synthetic\synthetic_001.truth.json
pytest                                       # 26 tests, cloud services faked
```

### Large PDFs (1,000+ scanned pages)

```powershell
python -m ime_copilot run big.pdf --pages 1-50     # try a slice first
```

Set `IME_OCR_DPI=200` (about 2x faster), `IME_OCR_WORKERS` (default: one per CPU) and `IME_OCR_CACHE=data/ocr_cache` (resume after an interruption). Progress and time remaining are printed as it runs.

### Google Colab / Azure ML notebook

`IME_Copilot_Colab.ipynb` (next to this folder) runs everything on one PDF and shows tables of details, pages, ledger and questions. Free Colab is **not** HIPAA-covered: use it only for synthetic or de-identified files. For real records, run the same notebook in Azure Machine Learning notebooks under the Microsoft BAA.

### Phase 1 deliverable: upload a PDF, retrieve its text page by page

```powershell
python -m ime_copilot serve                  # then open http://127.0.0.1:8000/docs
```

| Endpoint | Returns |
| --- | --- |
| `POST /cases` (form: `file`, optional `case_id`) | Stores the original, processes it, returns a summary |
| `GET /cases` | All processed cases |
| `GET /cases/{id}/pages` | Every page: type, OCR source and confidence, review flag, duplicate-of |
| `GET /cases/{id}/pages/{n}` | One page's full text and tables |
| `GET /cases/{id}/ledger` | Services, diagnoses, case questions, each with page and quote |

Every call needs the header `X-API-Key: <IME_API_KEY from .env>`. This is a development guard; production puts Microsoft Entra ID sign-in in front of the API.

## Connect to Azure

```powershell
pip install -e ".[azure,dev]"
az login        # your Entra ID identity is used; no keys needed
```

Then set these in `.env`:

| Setting | Value |
| --- | --- |
| `IME_STORAGE=azure_blob` | plus `AZURE_STORAGE_ACCOUNT_URL` and `AZURE_STORAGE_CONTAINER` (private container) |
| `IME_OCR=azure_di` | plus `AZURE_DI_ENDPOINT` |
| `IME_DB_URL` | Azure Database for PostgreSQL connection string |
| `IME_PROVIDER=azure_openai`, `IME_EXTRACTOR=llm` | plus `AZURE_OPENAI_ENDPOINT` and deployment names in `IME_MODEL_*` (later phases) |

Roles your identity (and later the app's managed identity) needs:
- *Storage Blob Data Contributor* on the storage account
- *Cognitive Services User* on Document Intelligence
- *Cognitive Services OpenAI User* on Azure OpenAI

## HIPAA rules built into the code

- **Text only to models.** Document Intelligence turns pages into text; only text goes to Azure OpenAI. Sending PDFs or page images to the model is avoided, because their coverage under the Microsoft BAA isn't established.
- **US processing.** Use *Data Zone Standard (US)* deployments for Azure OpenAI, not *Global*, and US regions for every resource.
- **Cite or omit.** A ledger row survives only if its quote is on the cited page and contains the row's date, code and amount.
- **No PHI in logs.** `model_calls` stores token counts, cost, latency and a prompt hash, never text.
- **Entra ID over keys.** Keys are supported for local development only.
- **Synthetic or de-identified files only** until the HIPAA go-live gate.

## Layout

```
ime_copilot/
  api.py              FastAPI upload/read API
  storage.py          Azure Blob / local storage
  ingest/azure_di.py  Document Intelligence: text, tables, confidence, layout, 2,000-page chunks
  ingest/pdf.py       OCR router (azure_di | pdftext | tesseract | auto | textract)
  ingest/dedupe.py    near-duplicate pages (never merges notes with different dates)
  ingest/classify.py  page types: rules baseline + LLM
  extract/            services, diagnoses, case questions, citation verifier
  gateway.py          model gateway: Azure OpenAI (v1 API), Bedrock, retries, cost metering
  db.py               ledger tables (PostgreSQL / SQLite)
  pipeline.py, evaluate.py, synthetic.py, cli.py
tests/                pipeline tests + Azure tests with fakes
```

## Next steps (your phase plan)

- **Phase 1 finish:** run against real Azure resources with synthetic PDFs; move processing to a queue (Azure Service Bus + a worker) for large files.
- **Phase 2:** group pages into documents; tune classification on the de-identified golden set.
- **Phase 3:** facility, paid amounts, medications, pain scores and functional measures; more bill formats (UB-04, itemized statements).
- **Phase 5 (suggested early):** rules engine with the first public-source rules (LCD frequency limits, Medicare fee schedule).
