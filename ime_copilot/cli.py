"""Command line:
  ime ask <pdf or case id> ["question"]    ask questions about a document (interactive if no question)
  ime run <pdf> | ime eval <pdf> <truth.json> | ime eval-qa | ime synth | ime serve | ime check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Settings
from .evaluate import load_truth, score
from .pipeline import print_summary, process_case, write_outputs
from .ingest.pdf import parse_range
from .storage import get_storage
from .synthetic import build_case


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ime", description="IME Copilot - record pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="Generate a synthetic case PDF + ground truth")
    s.add_argument("--out", default="data/synthetic")
    s.add_argument("--case-id", default="synthetic_001")

    r = sub.add_parser("run", help="Process a case PDF into the ledger")
    r.add_argument("pdf")
    r.add_argument("--case-id")
    r.add_argument("--out", default="data/output")
    r.add_argument("--pages", help="Process only these pages, e.g. 1-50 (try a slice of a big file first)")

    e = sub.add_parser("eval", help="Process a case and score it against ground truth")
    e.add_argument("pdf")
    e.add_argument("truth")
    e.add_argument("--out", default="data/output")
    e.add_argument("--pages", help="Process only these pages, e.g. 1-50")

    v = sub.add_parser("serve", help="Run the upload API (docs at http://127.0.0.1:8000/docs)")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8000)

    sub.add_parser("check", help="Test the model connections with a tiny made-up request (no patient data)")

    a = sub.add_parser("ask", help="Ask questions about a document: answer + CPT codes + page numbers")
    a.add_argument("target", help="PDF file (processed first if needed) or an already processed case id")
    a.add_argument("question", nargs="?", help="Leave out to type questions one after another")
    a.add_argument("--pages", help="Only process these pages of a new PDF, e.g. 1-200")
    a.add_argument("--reprocess", action="store_true", help="Process the PDF again even if done before")
    a.add_argument("--json", action="store_true", help="Print the full result as JSON")

    sub.add_parser("eval-qa", help="Score question answering on the synthetic case (pages + CPT codes)")

    args = parser.parse_args(argv)
    settings = Settings.from_env()

    if args.cmd == "check":
        from .check import run_check
        return 0 if run_check(settings) else 1

    if args.cmd == "ask":
        return _ask(args, settings)

    if args.cmd == "eval-qa":
        from .rag.evalqa import run_eval_qa
        return run_eval_qa(settings)

    if args.cmd == "serve":
        import uvicorn
        uvicorn.run("ime_copilot.api:app", host=args.host, port=args.port)
        return 0

    if args.cmd == "synth":
        pdf, truth = build_case(args.out, args.case_id)
        print(f"Synthetic case: {pdf}\nGround truth:   {truth}")
        return 0

    pdf = Path(args.pdf)
    case_id = getattr(args, "case_id", None) or pdf.stem
    print(f"Processing {pdf.name}  (ocr={settings.ocr}, extractor={settings.extractor}, "
          f"provider={settings.provider})", file=sys.stderr)
    result = process_case(pdf, case_id, settings, storage=get_storage(settings),
                          page_range=parse_range(args.pages))
    paths = write_outputs(result, args.out)
    print_summary(result)
    print(f"\nLedger: {paths['json']}\nServices CSV: {paths['csv']}", file=sys.stderr)

    if args.cmd == "eval":
        report = score(result, load_truth(args.truth))
        print("\n=== Evaluation ===")
        print(json.dumps(report, indent=2))
        return 0 if all(report["targets_met"].values()) else 1
    return 0


def _ask(args, settings) -> int:
    import re
    from .db import get_case, get_engine
    from .rag.service import get_qa, result_json, result_text

    engine = get_engine(settings.db_url)
    target = Path(args.target)
    if target.suffix.lower() == ".pdf" or target.exists():
        if not target.exists():
            print(f"File not found: {target}", file=sys.stderr)
            return 1
        case_id = re.sub(r"[^A-Za-z0-9_-]", "_", target.stem)[:64]
        if args.reprocess or get_case(engine, case_id) is None:
            print(f"Reading {target.name} (first time only; OCR results are cached)...", file=sys.stderr)
            process_case(target, case_id, settings, storage=get_storage(settings),
                         page_range=parse_range(args.pages))
    else:
        case_id = args.target
    print(f"Loading case '{case_id}' (search index is saved after the first time)...", file=sys.stderr)
    import time
    t0, last = time.monotonic(), [-1]

    def show(done: int, total: int) -> None:
        if total and done < total and (done == 0 or done * 10 // total != last[0]):
            last[0] = done * 10 // total
            spent = time.monotonic() - t0
            left = f", ~{spent / done * (total - done) / 60:.0f} min left" if done else ""
            print(f"[index] {done}/{total} passages{left}", file=sys.stderr, flush=True)

    try:
        qa = get_qa(engine, case_id, settings, refresh=True, on_progress=show)
    except KeyError as err:
        print(err, file=sys.stderr)
        return 1
    model = settings.models.get("answer") if settings.provider != "none" else None
    print(f"Ready: {len(qa.pages)} pages | answering with: {model or 'no model (extractive search only)'}",
          file=sys.stderr)

    from .case_summary import load_overview
    overview = (load_overview(settings.index_dir, case_id) or {}).get("text") or None
    history: list[dict] = []

    def one(question: str) -> None:
        r = qa.ask(question, history=history, overview=overview)
        history.append({"question": question, "answer": r.answer})
        print(result_json(r) if args.json else result_text(r))

    if args.question:
        one(args.question)
        return 0
    print("Type a question and press Enter (empty line or 'exit' to quit).", file=sys.stderr)
    while True:
        try:
            q = input("\n? ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("exit", "quit"):
            break
        one(q)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
