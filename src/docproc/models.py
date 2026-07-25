"""Extraction schemas — the contract between the LLM and the pipeline.

Design principle: the schema IS the product. Everything downstream
(extraction prompts, validation, the review UI, analytics) is driven by
these Pydantic models. Adding a new document type = adding a schema and
registering it; no pipeline changes.

Two kinds of validation live here:
* **Type-level** (field validators): dates parse, amounts are positive,
  enums match. These run automatically when the LLM output is parsed —
  with `instructor`, validation errors are fed BACK to the LLM so it can
  self-correct (retry-with-feedback loop).
* **Cross-field** (model validators): line items sum to the total, due
  date is after the invoice date. Catches hallucinated totals.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Document taxonomy
# ---------------------------------------------------------------------------


class DocType(str, Enum):
    invoice = "invoice"
    contract = "contract"
    unknown = "unknown"


# ---------------------------------------------------------------------------
# Invoice schema
# ---------------------------------------------------------------------------


class LineItem(BaseModel):
    description: str = Field(min_length=1, max_length=300)
    quantity: float = Field(gt=0)
    unit_price: float = Field(ge=0)
    total: float = Field(ge=0)

    @model_validator(mode="after")
    def total_matches(self) -> "LineItem":
        expected = round(self.quantity * self.unit_price, 2)
        # 2-cent tolerance for rounding differences on the source document
        if abs(expected - round(self.total, 2)) > 0.02:
            raise ValueError(
                f"line item total {self.total} != quantity*unit_price {expected}"
            )
        return self


class Invoice(BaseModel):
    """Target schema for invoice extraction.

    Every field is Optional-free on purpose: if a value is genuinely
    absent from the document the LLM must say so via `null`-able fields
    marked explicitly below — silence is not allowed.
    """

    vendor_name: str = Field(min_length=1, max_length=200)
    invoice_number: str = Field(min_length=1, max_length=64)
    invoice_date: date
    due_date: date | None = None
    currency: str = Field(default="EUR", pattern=r"^[A-Z]{3}$")
    line_items: list[LineItem] = Field(min_length=1)
    subtotal: float | None = Field(default=None, ge=0)
    tax_amount: float = Field(ge=0)
    total_amount: float = Field(gt=0)
    payment_terms: str | None = Field(default=None, max_length=300)

    @field_validator("invoice_date", "due_date", mode="before")
    @classmethod
    def parse_flexible_dates(cls, v):
        """Accept common European and ISO formats (OCR output varies)."""
        if v is None or isinstance(v, date):
            return v
        from datetime import datetime

        for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%d %B %Y", "%B %d, %Y"):
            try:
                return datetime.strptime(str(v).strip(), fmt).date()
            except ValueError:
                continue
        raise ValueError(f"unparseable date: {v!r}")

    @model_validator(mode="after")
    def totals_consistent(self) -> "Invoice":
        items_sum = round(sum(li.total for li in self.line_items), 2)
        if self.subtotal is not None and abs(items_sum - self.subtotal) > 0.05:
            raise ValueError(
                f"line items sum {items_sum} != subtotal {self.subtotal}"
            )
        base = self.subtotal if self.subtotal is not None else items_sum
        expected_total = round(base + self.tax_amount, 2)
        if abs(expected_total - round(self.total_amount, 2)) > 0.05:
            raise ValueError(
                f"subtotal+tax {expected_total} != total_amount {self.total_amount}"
            )
        if self.due_date and self.due_date < self.invoice_date:
            raise ValueError("due_date is before invoice_date")
        return self


# ---------------------------------------------------------------------------
# Contract schema
# ---------------------------------------------------------------------------


class Contract(BaseModel):
    parties: list[str] = Field(min_length=2)
    effective_date: date
    term_months: int | None = Field(default=None, gt=0, le=1200)
    governing_law: str | None = None
    key_obligations: list[str] = Field(default_factory=list, max_length=20)
    termination_notice_days: int | None = Field(default=None, ge=0, le=3650)

    @field_validator("effective_date", mode="before")
    @classmethod
    def parse_date(cls, v):
        return Invoice.parse_flexible_dates(v)


SCHEMA_REGISTRY: dict[DocType, type[BaseModel]] = {
    DocType.invoice: Invoice,
    DocType.contract: Contract,
}


# ---------------------------------------------------------------------------
# Pipeline result envelopes
# ---------------------------------------------------------------------------


class FieldConfidence(BaseModel):
    field: str
    value: str
    confidence: float = Field(ge=0, le=1)
    signals: dict[str, float] = Field(default_factory=dict)


class ValidationIssue(BaseModel):
    field: str
    severity: Literal["warning", "error"]
    message: str


class ProcessingResult(BaseModel):
    """Everything the pipeline knows about one processed document."""

    doc_id: str
    source_file: str
    doc_type: DocType
    ocr_backend: str
    ocr_confidence: float = Field(ge=0, le=1)
    extraction: dict | None = None          # validated schema dump, or None on failure
    extraction_error: str | None = None
    field_confidences: list[FieldConfidence] = Field(default_factory=list)
    validation_issues: list[ValidationIssue] = Field(default_factory=list)
    anomalies: list[str] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0, le=1, default=0.0)
    route: Literal["auto_approve", "fast_review", "detailed_review"] = "detailed_review"
    raw_text_preview: str = ""
