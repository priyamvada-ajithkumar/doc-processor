"""CLI batch processor.  Usage: python scripts/process.py file1.pdf [file2.png ...]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docproc.pipeline import analytics, process_document

def main() -> None:
    files = sys.argv[1:] or ["data/samples/invoice_native.pdf",
                             "data/samples/invoice_scan.png",
                             "data/samples/contract.txt"]
    for f in files:
        r = process_document(f)
        print(f"{r.source_file:30s} {r.doc_type.value:9s} "
              f"ocr={r.ocr_backend}({r.ocr_confidence:.2f}) "
              f"conf={r.overall_confidence:.2f} -> {r.route}")
        if r.extraction_error:
            print(f"   [error] {r.extraction_error}")
        for issue in r.validation_issues:
            print(f"   [{issue.severity}] {issue.field}: {issue.message}")
        for a in r.anomalies:
            print(f"   [anomaly] {a}")
    print("\nAnalytics:", analytics())

if __name__ == "__main__":
    main()
