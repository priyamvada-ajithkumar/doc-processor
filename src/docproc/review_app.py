"""Streamlit review UI — the human-in-the-loop interface.

Run:  PYTHONPATH=src streamlit run src/docproc/review_app.py

Three tabs mirror the reviewer workflow:
  1. Upload & Process — drop a file, watch it flow through the pipeline.
  2. Review Queue — documents routed to review, color-coded fields,
     inline correction with reason capture (the feedback loop!).
  3. Analytics — auto-approval rate, route distribution, top failing
     fields. The auto-approval rate is THE efficiency metric: it is the
     percentage of documents that cost zero human minutes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from docproc.config import get_settings
from docproc.pipeline import (
    analytics,
    load_results,
    process_document,
    record_correction,
)

st.set_page_config(page_title="Document Processor", layout="wide")
st.title("📄 Multi-Modal Document Processor")

settings = get_settings()
tab_upload, tab_review, tab_analytics = st.tabs(["Upload & Process", "Review Queue", "Analytics"])

# ---------------------------------------------------------------- Upload tab
with tab_upload:
    uploaded = st.file_uploader(
        "Upload a document", type=["pdf", "png", "jpg", "jpeg", "tiff", "txt"]
    )
    if uploaded is not None:
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        dest = settings.upload_dir / uploaded.name
        dest.write_bytes(uploaded.getvalue())
        with st.spinner("Processing (OCR → classify → extract → validate → route)…"):
            result = process_document(dest)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Document type", result.doc_type.value)
        c2.metric("OCR backend", result.ocr_backend)
        c3.metric("Overall confidence", f"{result.overall_confidence:.0%}")
        route_icon = {"auto_approve": "✅", "fast_review": "🟡", "detailed_review": "🔴"}
        c4.metric("Route", f"{route_icon.get(result.route,'')} {result.route}")

        if result.extraction_error:
            st.error(f"Extraction failed: {result.extraction_error}")
        if result.validation_issues:
            for issue in result.validation_issues:
                (st.error if issue.severity == "error" else st.warning)(
                    f"{issue.field}: {issue.message}"
                )
        for a in result.anomalies:
            st.error(f"Anomaly: {a}")
        if result.extraction:
            st.subheader("Extracted data")
            st.json(result.extraction)
        with st.expander("OCR text preview"):
            st.text(result.raw_text_preview)

# ---------------------------------------------------------------- Review tab
with tab_review:
    queue = [r for r in load_results() if r.route in ("fast_review", "detailed_review")]
    if not queue:
        st.success("Review queue is empty 🎉")
    for result in queue:
        with st.expander(
            f"{'🔴' if result.route == 'detailed_review' else '🟡'} "
            f"{result.source_file} — {result.doc_type.value} — conf {result.overall_confidence:.0%}"
        ):
            left, right = st.columns(2)
            with left:
                st.caption("OCR text")
                st.text(result.raw_text_preview)
                for issue in result.validation_issues:
                    st.warning(f"{issue.field}: {issue.message}")
            with right:
                st.caption("Extracted fields (edit to correct)")
                for fc in result.field_confidences:
                    color = "🟢" if fc.confidence >= 0.8 else ("🟡" if fc.confidence >= 0.5 else "🔴")
                    new_val = st.text_input(
                        f"{color} {fc.field} ({fc.confidence:.0%})",
                        value=fc.value,
                        key=f"{result.doc_id}-{fc.field}",
                    )
                    if new_val != fc.value:
                        reason = st.selectbox(
                            "Why?",
                            ["extraction_error", "validation_false_positive", "other"],
                            key=f"{result.doc_id}-{fc.field}-reason",
                        )
                        if st.button("Save correction", key=f"{result.doc_id}-{fc.field}-save"):
                            record_correction(result.doc_id, fc.field, fc.value, new_val, reason)
                            st.success("Correction recorded — this feeds the improvement loop.")

# ------------------------------------------------------------- Analytics tab
with tab_analytics:
    stats = analytics()
    c1, c2 = st.columns(2)
    c1.metric("Documents processed", stats["documents_processed"])
    c2.metric("Auto-approval rate", f"{stats['auto_approval_rate']:.0%}")
    st.bar_chart(stats["by_route"])
    if stats["top_correction_fields"]:
        st.subheader("Fields needing the most corrections")
        st.table(stats["top_correction_fields"])
        st.caption(
            "High-correction fields are where to improve prompts, few-shot "
            "examples, or validation rules next."
        )
