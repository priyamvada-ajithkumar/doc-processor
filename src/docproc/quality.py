"""Post-extraction intelligence: confidence, business rules, anomalies, routing.

This module is what separates a demo from a production pipeline. The LLM
gives you *an* answer; these stages decide whether to *trust* it.

Confidence per field combines independent signals:
  * source_match  — does the extracted value literally appear in the OCR
    text? (Exact 1.0 / fuzzy 0.7 / absent 0.2.) The single best
    hallucination detector: a value not present in the source was made up.
  * ocr_quality   — garbage in, garbage out; field confidence inherits
    document OCR confidence.
  * format_valid  — the value survived Pydantic parsing (dates parsed,
    amounts numeric), a weak but free signal.

Routing turns the overall score plus validation results into a business
decision. The thresholds are the pipeline's most important knobs: raise
auto_approve and you save review labor but let more errors through —
that trade-off is a business decision, which is why they live in config,
not code.
"""

from __future__ import annotations

import re
import statistics

from pydantic import BaseModel

from .config import get_settings
from .models import (
    FieldConfidence,
    Invoice,
    ProcessingResult,
    ValidationIssue,
)

# ---------------------------------------------------------------------------
# Field confidence
# ---------------------------------------------------------------------------


def _source_match(value: str, text: str) -> float:
    if not value or value in {"UNKNOWN", "UNPARSED"}:
        return 0.0
    norm_text = re.sub(r"\s+", " ", text.lower())
    norm_val = re.sub(r"\s+", " ", str(value).lower()).strip()
    if norm_val in norm_text:
        return 1.0
    # fuzzy: all tokens individually present (e.g. reformatted dates/amounts)
    tokens = [t for t in re.split(r"[^\w.]+", norm_val) if len(t) > 2]
    if tokens and all(t in norm_text for t in tokens):
        return 0.7
    return 0.2


def score_fields(extraction: BaseModel, ocr_text: str, ocr_conf: float) -> list[FieldConfidence]:
    scores: list[FieldConfidence] = []
    for field, value in extraction.model_dump(mode="json").items():
        if isinstance(value, list):
            value = f"{len(value)} items"
            src = 0.8  # lists are scored via their own cross-field validators
        else:
            src = _source_match(str(value) if value is not None else "", ocr_text)
        signals = {
            "source_match": src,
            "ocr_quality": ocr_conf,
            "format_valid": 1.0,  # it parsed, or we wouldn't be here
        }
        conf = round(0.55 * src + 0.30 * ocr_conf + 0.15 * 1.0, 4)
        scores.append(
            FieldConfidence(field=field, value=str(value), confidence=conf, signals=signals)
        )
    return scores


# ---------------------------------------------------------------------------
# Business rules (beyond Pydantic type validation)
# ---------------------------------------------------------------------------


def business_rules(extraction: BaseModel) -> list[ValidationIssue]:
    settings = get_settings()
    issues: list[ValidationIssue] = []
    if isinstance(extraction, Invoice):
        if extraction.vendor_name.lower() not in settings.vendor_set:
            issues.append(
                ValidationIssue(
                    field="vendor_name",
                    severity="warning",
                    message=f"Vendor '{extraction.vendor_name}' not in known vendor list.",
                )
            )
        if extraction.total_amount > settings.max_plausible_invoice_total:
            issues.append(
                ValidationIssue(
                    field="total_amount",
                    severity="error",
                    message=f"Total {extraction.total_amount} exceeds plausibility "
                    f"cap {settings.max_plausible_invoice_total}.",
                )
            )
        if extraction.due_date is None:
            issues.append(
                ValidationIssue(
                    field="due_date", severity="warning", message="No due date found."
                )
            )
    return issues


# ---------------------------------------------------------------------------
# Anomaly detection against history
# ---------------------------------------------------------------------------


def detect_anomalies(extraction: BaseModel, history_totals: list[float]) -> list[str]:
    """Flag statistical outliers vs previously processed documents.

    Robust z-score using median/MAD instead of mean/std — invoice
    amounts are heavy-tailed and a single big invoice would otherwise
    inflate the std and hide real outliers.
    """
    anomalies: list[str] = []
    if isinstance(extraction, Invoice) and len(history_totals) >= 5:
        med = statistics.median(history_totals)
        mad = statistics.median([abs(t - med) for t in history_totals]) or 1.0
        z = 0.6745 * (extraction.total_amount - med) / mad
        if abs(z) > 3.5:
            anomalies.append(
                f"total_amount {extraction.total_amount} is a statistical outlier "
                f"(robust z={z:.1f}, median={med:.2f})"
            )
    return anomalies


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route(result: ProcessingResult) -> str:
    settings = get_settings()
    has_error = any(i.severity == "error" for i in result.validation_issues)
    if result.extraction is None or has_error or result.anomalies:
        return "detailed_review"
    if result.overall_confidence >= settings.auto_approve_threshold and not result.validation_issues:
        return "auto_approve"
    if result.overall_confidence >= settings.fast_review_threshold:
        return "fast_review"
    return "detailed_review"


def overall_confidence(fields: list[FieldConfidence], ocr_conf: float) -> float:
    if not fields:
        return 0.0
    # Weakest-link weighting: mean pulled toward the minimum, because one
    # hallucinated critical field can invalidate an otherwise clean doc.
    mean = sum(f.confidence for f in fields) / len(fields)
    weakest = min(f.confidence for f in fields)
    return round(0.6 * mean + 0.25 * weakest + 0.15 * ocr_conf, 4)
