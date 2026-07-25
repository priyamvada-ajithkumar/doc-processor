"""Generate sample documents for demos and tests.

Creates three files in data/samples/:
  invoice_native.pdf   — born-digital PDF (native text path)
  invoice_scan.png     — rendered invoice image (OCR path)
  contract.txt         — plain-text contract

The scan image is rendered with PIL at decent quality so Tesseract can
read it; add noise/rotation yourself to stress-test preprocessing.
"""

from __future__ import annotations

from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"

INVOICE_TEXT = """Northwind Supplies GmbH
Industriestrasse 12, 90402 Nuremberg

INVOICE

Invoice Number: INV-2026-0042
Invoice Date: 2026-06-15
Due Date: 2026-07-15
Currency: EUR

Description                        Quantity    Unit Price    Total
Industrial sensor unit             4           250.00        1000.00
Calibration service                2           150.00        300.00
Mounting hardware kit              10          12.50         125.00

Subtotal: 1425.00
VAT (19%): 270.75
Total Due: 1695.75 EUR

Payment Terms: Net 30 days
"""

CONTRACT_TEXT = """SERVICE AGREEMENT

This agreement is made between Northwind Supplies GmbH (the Provider) and
ACME Industrial AG (the Client).

Effective Date: 2026-05-01
The agreement has a term of 24 months from the effective date.

The Provider shall deliver quarterly maintenance of all sensor equipment.
The Client shall provide site access during business hours.

Either party may terminate with 60 days' written notice.
This agreement is governed by the laws of Germany.
"""


def make_native_pdf() -> Path:
    path = SAMPLES / "invoice_native.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 60), INVOICE_TEXT, fontname="cour", fontsize=11)
    doc.save(path)
    doc.close()
    return path


def make_scan_image() -> Path:
    path = SAMPLES / "invoice_scan.png"
    img = Image.new("RGB", (1400, 1500), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 28
        )
    except OSError:
        font = ImageFont.load_default()
    y = 60
    for line in INVOICE_TEXT.splitlines():
        draw.text((60, y), line, fill="black", font=font)
        y += 44
    img.save(path)
    return path


def make_contract() -> Path:
    path = SAMPLES / "contract.txt"
    path.write_text(CONTRACT_TEXT, encoding="utf-8")
    return path


def main() -> None:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    for maker in (make_native_pdf, make_scan_image, make_contract):
        print("created", maker())


if __name__ == "__main__":
    main()
