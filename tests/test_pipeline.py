from pathlib import Path

import pytest

from docproc import pipeline
from docproc.config import get_settings
from docproc.extraction import classify
from docproc.models import DocType
from docproc.ocr import load_document

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Point storage at a temp DB so tests never share state.

    Settings are lru_cached, so we set the env var and clear the cache —
    the same mechanism production uses to configure the pipeline."""
    monkeypatch.setenv("DOCPROC_DB_PATH", str(tmp_path / "test.db"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_native_pdf_uses_native_path():
    ocr = load_document(SAMPLES / "invoice_native.pdf")
    assert ocr.backend == "native"
    assert "INV-2026-0042" in ocr.text
    assert ocr.confidence > 0.95


def test_scanned_image_ocr_reads_invoice_number():
    ocr = load_document(SAMPLES / "invoice_scan.png")
    assert ocr.backend in ("tesseract", "tesseract+easyocr")
    assert "INV-2026-0042" in ocr.text.replace(" ", "")
    assert ocr.confidence > 0.5


def test_classifier():
    assert classify("Invoice Number: 1 Total Due 500 VAT") == DocType.invoice
    assert classify("This agreement is between the parties, governed by law") == DocType.contract
    assert classify("hello world") == DocType.unknown


def test_end_to_end_native_invoice():
    result = pipeline.process_document(SAMPLES / "invoice_native.pdf")
    assert result.doc_type == DocType.invoice
    assert result.extraction is not None
    assert result.extraction["invoice_number"] == "INV-2026-0042"
    assert result.extraction["total_amount"] == 1695.75
    assert len(result.extraction["line_items"]) == 3
    assert result.route in ("auto_approve", "fast_review")


def test_end_to_end_scanned_invoice_routes_by_confidence():
    result = pipeline.process_document(SAMPLES / "invoice_scan.png")
    assert result.doc_type == DocType.invoice
    assert result.ocr_confidence < 0.99


def test_contract_extraction():
    result = pipeline.process_document(SAMPLES / "contract.txt")
    assert result.doc_type == DocType.contract
    assert result.extraction["term_months"] == 24
    assert result.extraction["termination_notice_days"] == 60


def test_unknown_doc_goes_to_detailed_review(tmp_path):
    p = tmp_path / "noise.txt"
    p.write_text("lorem ipsum dolor sit amet")
    result = pipeline.process_document(p)
    assert result.doc_type == DocType.unknown
    assert result.route == "detailed_review"


def test_correction_feedback_loop():
    result = pipeline.process_document(SAMPLES / "invoice_native.pdf")
    pipeline.record_correction(result.doc_id, "vendor_name", "X", "Y", "extraction_error")
    stats = pipeline.analytics()
    assert stats["documents_processed"] >= 1
    assert stats["top_correction_fields"][0]["field"] == "vendor_name"
