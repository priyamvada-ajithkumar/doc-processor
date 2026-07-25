import pytest
from pydantic import ValidationError

from docproc.models import Invoice, LineItem


def valid_invoice(**overrides):
    base = dict(
        vendor_name="Northwind Supplies GmbH",
        invoice_number="INV-1",
        invoice_date="2026-06-15",
        due_date="2026-07-15",
        line_items=[LineItem(description="Widget", quantity=2, unit_price=10.0, total=20.0)],
        subtotal=20.0,
        tax_amount=3.8,
        total_amount=23.8,
    )
    base.update(overrides)
    return Invoice(**base)


def test_valid_invoice_parses():
    inv = valid_invoice()
    assert inv.total_amount == 23.8


def test_flexible_date_formats():
    assert valid_invoice(invoice_date="15.06.2026").invoice_date.isoformat() == "2026-06-15"


def test_line_item_math_enforced():
    with pytest.raises(ValidationError, match="quantity"):
        LineItem(description="x", quantity=2, unit_price=10.0, total=99.0)


def test_hallucinated_total_caught():
    with pytest.raises(ValidationError, match="total_amount"):
        valid_invoice(total_amount=500.0)


def test_due_before_invoice_caught():
    with pytest.raises(ValidationError, match="due_date"):
        valid_invoice(due_date="2026-01-01")
