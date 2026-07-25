from docproc.models import DocType, FieldConfidence, Invoice, LineItem, ProcessingResult
from docproc.quality import _source_match, detect_anomalies, overall_confidence, route


def make_invoice(total=1000.0):
    net = round(total / 1.19, 2)
    return Invoice(
        vendor_name="ACME Industrial AG",
        invoice_number="INV-9",
        invoice_date="2026-01-01",
        line_items=[LineItem(description="x", quantity=1, unit_price=net, total=net)],
        tax_amount=round(total - net, 2),
        total_amount=total,
    )


def test_source_match_detects_hallucination():
    text = "Invoice Number: INV-1 Total: 500.00"
    assert _source_match("INV-1", text) == 1.0
    assert _source_match("INV-9999", text) <= 0.2


def test_anomaly_detector_flags_outlier():
    history = [100.0, 110.0, 95.0, 105.0, 102.0, 98.0]
    assert detect_anomalies(make_invoice(total=50000.0), history)
    assert not detect_anomalies(make_invoice(total=104.0), history)


def test_anomaly_detector_needs_history():
    assert detect_anomalies(make_invoice(total=50000.0), [100.0]) == []


def test_weakest_link_confidence():
    strong = [FieldConfidence(field=f"f{i}", value="v", confidence=0.95) for i in range(5)]
    with_weak = strong + [FieldConfidence(field="bad", value="v", confidence=0.1)]
    assert overall_confidence(with_weak, 0.9) < overall_confidence(strong, 0.9)


def test_routing_thresholds():
    base = dict(doc_id="d", source_file="f", doc_type=DocType.invoice,
                ocr_backend="native", ocr_confidence=0.99, extraction={"a": 1})
    assert route(ProcessingResult(**base, overall_confidence=0.95)) == "auto_approve"
    assert route(ProcessingResult(**base, overall_confidence=0.70)) == "fast_review"
    assert route(ProcessingResult(**base, overall_confidence=0.30)) == "detailed_review"
