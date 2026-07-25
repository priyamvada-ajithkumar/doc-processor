"""Pipeline orchestrator + SQLite persistence + Celery task wrapper.

The orchestrator is intentionally a plain function (`process_document`)
with no framework coupling. Celery wraps it; tests call it directly;
Streamlit calls it directly. One code path, three entry points — the
same "framework-independent core" pattern as the Datum project.

Storage records every processed document AND every human correction.
Corrections are the feedback loop: they tell you which fields fail most
(prompt improvements), whether validation rules are too strict (false
positives), and how accurate auto-approval actually is (threshold
tuning). A pipeline without a correction log cannot improve.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import get_settings
from .extraction import ExtractionError, classify, extract
from .models import DocType, ProcessingResult
from .ocr import load_document
from .quality import (
    business_rules,
    detect_anomalies,
    overall_confidence,
    route,
    score_fields,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    source_file TEXT,
    processed_at TEXT,
    doc_type TEXT,
    route TEXT,
    overall_confidence REAL,
    ocr_backend TEXT,
    result_json TEXT
);
CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT REFERENCES documents(doc_id),
    field TEXT,
    original_value TEXT,
    corrected_value TEXT,
    reason TEXT,            -- extraction_error | validation_false_positive | other
    corrected_at TEXT
);
"""


def _conn() -> sqlite3.Connection:
    settings = get_settings()
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.db_path)
    conn.executescript(_SCHEMA)
    return conn


def save_result(result: ProcessingResult) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?,?,?)",
            (
                result.doc_id,
                result.source_file,
                datetime.now(timezone.utc).isoformat(),
                result.doc_type.value,
                result.route,
                result.overall_confidence,
                result.ocr_backend,
                result.model_dump_json(),
            ),
        )


def load_results(route_filter: str | None = None) -> list[ProcessingResult]:
    with _conn() as conn:
        q = "SELECT result_json FROM documents"
        params: tuple = ()
        if route_filter:
            q += " WHERE route = ?"
            params = (route_filter,)
        rows = conn.execute(q + " ORDER BY processed_at DESC", params).fetchall()
    return [ProcessingResult.model_validate_json(r[0]) for r in rows]


def historical_totals() -> list[float]:
    totals = []
    for r in load_results():
        if r.doc_type == DocType.invoice and r.extraction:
            totals.append(float(r.extraction.get("total_amount", 0)))
    return totals


def record_correction(doc_id: str, field: str, original: str, corrected: str, reason: str) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO corrections (doc_id, field, original_value, corrected_value, reason, corrected_at) "
            "VALUES (?,?,?,?,?,?)",
            (doc_id, field, original, corrected, reason, datetime.now(timezone.utc).isoformat()),
        )


def analytics() -> dict:
    with _conn() as conn:
        by_route = dict(
            conn.execute("SELECT route, COUNT(*) FROM documents GROUP BY route").fetchall()
        )
        total = sum(by_route.values())
        error_fields = conn.execute(
            "SELECT field, COUNT(*) c FROM corrections WHERE reason='extraction_error' "
            "GROUP BY field ORDER BY c DESC LIMIT 5"
        ).fetchall()
    return {
        "documents_processed": total,
        "by_route": by_route,
        "auto_approval_rate": round(by_route.get("auto_approve", 0) / total, 3) if total else 0.0,
        "top_correction_fields": [{"field": f, "corrections": c} for f, c in error_fields],
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def process_document(path: str | Path) -> ProcessingResult:
    """One document end to end. Never raises for data problems — errors
    become part of the result and route to detailed_review, because in a
    batch pipeline one bad document must not kill the batch."""
    path = Path(path)
    doc_id = uuid.uuid4().hex[:12]

    ocr = load_document(path)
    doc_type = classify(ocr.text)

    result = ProcessingResult(
        doc_id=doc_id,
        source_file=path.name,
        doc_type=doc_type,
        ocr_backend=ocr.backend,
        ocr_confidence=ocr.confidence,
        raw_text_preview=ocr.text[:800],
    )

    if doc_type == DocType.unknown:
        result.extraction_error = "Could not classify document type."
        result.route = "detailed_review"
        save_result(result)
        return result

    try:
        extraction = extract(ocr.text, doc_type)
        result.extraction = extraction.model_dump(mode="json")
        result.field_confidences = score_fields(extraction, ocr.text, ocr.confidence)
        result.validation_issues = business_rules(extraction)
        result.anomalies = detect_anomalies(extraction, historical_totals())
        result.overall_confidence = overall_confidence(result.field_confidences, ocr.confidence)
    except (ExtractionError, ValueError) as exc:
        # Includes pydantic ValidationError (subclass of ValueError):
        # the LLM produced something that failed schema/cross-field checks
        # even after retries.
        result.extraction_error = str(exc)[:500]

    result.route = route(result)
    save_result(result)
    log.info("processed %s -> %s (conf=%.2f)", path.name, result.route, result.overall_confidence)
    return result


# ---------------------------------------------------------------------------
# Celery integration (optional — used by docker-compose)
# ---------------------------------------------------------------------------

try:  # pragma: no cover
    from celery import Celery

    celery_app = Celery("docproc", broker="redis://redis:6379/0", backend="redis://redis:6379/1")

    @celery_app.task(name="docproc.process_document", max_retries=2, autoretry_for=(IOError,))
    def process_document_task(path: str) -> dict:
        """Async wrapper. Retries only on IO errors (transient); data
        errors are handled inside process_document and must NOT retry —
        the same bad PDF will fail identically every time."""
        return process_document(path).model_dump(mode="json")

except ImportError:  # celery not installed — sync mode only
    celery_app = None
