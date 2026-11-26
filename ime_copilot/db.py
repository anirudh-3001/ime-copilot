"""Treatment ledger storage (SQLite locally, PostgreSQL in AWS).

One row per fact, and every row keeps page_no + quote + confidence so the
physician can click through to the source page.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, MetaData, String, Table, Text,
    create_engine, delete, inspect, select, text,
)
from sqlalchemy.engine import Engine

from .schemas import CaseResult

metadata = MetaData()

cases = Table(
    "cases", metadata,
    Column("case_id", String(64), primary_key=True),
    Column("source_file", Text, nullable=False),
    Column("page_count", Integer, nullable=False),
    Column("model_cost_usd", Float, default=0.0),
    Column("details_json", Text),                    # CaseDetails: name, DOB, claim no... with citations
    Column("processed_at", DateTime(timezone=True)),
)

pages = Table(
    "pages", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("case_id", String(64), ForeignKey("cases.case_id"), index=True),
    Column("page_no", Integer, nullable=False),
    Column("page_type", String(32)),
    Column("type_confidence", Float),
    Column("text_source", String(16)),
    Column("ocr_confidence", Float),
    Column("duplicate_of", Integer),
    Column("needs_review", Boolean, default=False),
    Column("low_confidence_words", Float),
    Column("angle", Float),
    Column("tables_json", Text),                     # JSON list of PageTable
    Column("text", Text),
)

# Columns added after the first release; created on old SQLite dev databases.
_ADDED_COLUMNS = {"cases": {"details_json": "TEXT"}, "pages": {"low_confidence_words": "FLOAT", "angle": "FLOAT", "tables_json": "TEXT"}}

services = Table(
    "services", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("case_id", String(64), ForeignKey("cases.case_id"), index=True),
    Column("date_of_service", String(10), index=True),
    Column("cpt", String(8), index=True),
    Column("modifier", String(8)),
    Column("units", Integer),
    Column("billed_amount", Float),
    Column("dx_pointer", String(8)),
    Column("provider", Text),
    Column("page_no", Integer, nullable=False),
    Column("quote", Text, nullable=False),
    Column("confidence", Float),
    Column("method", String(8)),
)

diagnoses = Table(
    "diagnoses", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("case_id", String(64), ForeignKey("cases.case_id"), index=True),
    Column("code", String(10), index=True),
    Column("description", Text),
    Column("first_seen_date", String(10)),
    Column("page_no", Integer, nullable=False),
    Column("quote", Text, nullable=False),
    Column("confidence", Float),
    Column("method", String(8)),
)

questions = Table(
    "questions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("case_id", String(64), ForeignKey("cases.case_id"), index=True),
    Column("seq", Integer, nullable=False),          # order in the letter
    Column("number", String(16), nullable=False),    # as numbered by the IME company
    Column("text", Text, nullable=False),
    Column("category", String(24)),
    Column("instructions", Text),                    # JSON list
    Column("page_no", Integer, nullable=False),
    Column("method", String(8)),
)

# Metering only: no prompt or output text is stored here (it may contain PHI).
model_calls = Table(
    "model_calls", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("case_id", String(64), index=True),
    Column("task", String(24)),
    Column("model_id", String(128)),
    Column("input_tokens", Integer),
    Column("output_tokens", Integer),
    Column("cost_usd", Float),
    Column("latency_ms", Integer),
    Column("prompt_sha256", String(64)),
    Column("ok", Boolean),
    Column("created_at", DateTime(timezone=True)),
)


def get_engine(db_url: str) -> Engine:
    if db_url.startswith("sqlite:///"):
        path = Path(db_url.replace("sqlite:///", "", 1))
        if path.parent and str(path.parent) not in ("", "."):
            path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(db_url, future=True)
    metadata.create_all(engine)
    _add_missing_columns(engine)
    return engine


def _add_missing_columns(engine: Engine) -> None:
    insp = inspect(engine)
    with engine.begin() as conn:
        for table, cols in _ADDED_COLUMNS.items():
            existing = {c["name"] for c in insp.get_columns(table)}
            for name, sql_type in cols.items():
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}"))


def save_case(engine: Engine, result: CaseResult) -> None:
    """Replace any earlier run of this case with the new result."""
    cid = result.case_id
    with engine.begin() as conn:
        for table in (pages, services, diagnoses, questions):
            conn.execute(delete(table).where(table.c.case_id == cid))
        conn.execute(delete(cases).where(cases.c.case_id == cid))

        conn.execute(cases.insert().values(
            case_id=cid, source_file=result.source_file, page_count=len(result.pages),
            model_cost_usd=result.model_cost_usd, details_json=result.details.model_dump_json(),
            processed_at=datetime.now(timezone.utc),
        ))
        if result.pages:
            conn.execute(pages.insert(), [
                dict(case_id=cid, page_no=p.page_no, page_type=p.page_type,
                     type_confidence=p.type_confidence, text_source=p.text_source,
                     ocr_confidence=p.ocr_confidence, duplicate_of=p.duplicate_of,
                     needs_review=p.needs_review, low_confidence_words=p.low_confidence_words,
                     angle=p.angle, tables_json=json.dumps([t.model_dump() for t in p.tables]),
                     text=p.text)
                for p in result.pages
            ])
        if result.services:
            conn.execute(services.insert(), [
                dict(case_id=cid, date_of_service=s.date_of_service, cpt=s.cpt,
                     modifier=s.modifier, units=s.units, billed_amount=s.billed_amount,
                     dx_pointer=s.dx_pointer, provider=s.provider,
                     page_no=s.citation.page_no, quote=s.citation.quote,
                     confidence=s.confidence, method=s.method)
                for s in result.services
            ])
        if result.diagnoses:
            conn.execute(diagnoses.insert(), [
                dict(case_id=cid, code=d.code, description=d.description,
                     first_seen_date=d.first_seen_date, page_no=d.citation.page_no,
                     quote=d.citation.quote, confidence=d.confidence, method=d.method)
                for d in result.diagnoses
            ])
        if result.questions:
            conn.execute(questions.insert(), [
                dict(case_id=cid, seq=i + 1, number=q.number, text=q.text,
                     category=q.category, instructions=json.dumps(q.instructions),
                     page_no=q.citation.page_no, method=q.method)
                for i, q in enumerate(result.questions)
            ])


def _rows(engine: Engine, stmt) -> list[dict]:
    with engine.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(stmt)]


def _with_details(row: dict) -> dict:
    row["details"] = json.loads(row.pop("details_json", None) or "{}")
    return row


def list_cases(engine: Engine) -> list[dict]:
    return [_with_details(r) for r in _rows(engine, select(cases).order_by(cases.c.processed_at.desc()))]


def get_case(engine: Engine, case_id: str) -> dict | None:
    rows = _rows(engine, select(cases).where(cases.c.case_id == case_id))
    return _with_details(rows[0]) if rows else None


def get_pages(engine: Engine, case_id: str, with_text: bool = False) -> list[dict]:
    cols = [c for c in pages.c if with_text or c.name not in ("text", "tables_json")]
    rows = _rows(engine, select(*cols).where(pages.c.case_id == case_id).order_by(pages.c.page_no))
    for r in rows:
        if "tables_json" in r:
            r["tables"] = json.loads(r.pop("tables_json") or "[]")
        r.pop("id", None)
        r.pop("case_id", None)
    return rows


def get_ledger(engine: Engine, case_id: str) -> dict:
    def clean(table, order):
        out = _rows(engine, select(table).where(table.c.case_id == case_id).order_by(*order))
        for r in out:
            r.pop("id", None)
            r.pop("case_id", None)
            if "instructions" in r:
                r["instructions"] = json.loads(r["instructions"] or "[]")
        return out
    return {
        "services": clean(services, [services.c.date_of_service, services.c.cpt]),
        "diagnoses": clean(diagnoses, [diagnoses.c.first_seen_date]),
        "questions": clean(questions, [questions.c.seq]),
    }


def log_model_call(engine: Engine | None, **row) -> None:
    if engine is None:
        return
    row.setdefault("created_at", datetime.now(timezone.utc))
    with engine.begin() as conn:
        conn.execute(model_calls.insert().values(**row))
