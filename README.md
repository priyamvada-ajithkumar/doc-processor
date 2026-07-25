# Multi-Modal Document Processor

An end-to-end document processing pipeline: any document (PDF, scan, image) in → OCR → LLM structured extraction → validation → confidence-based routing → human review — the pattern behind one of the largest categories of enterprise AI deployment.

```
document ─► ingestion ─► OCR (dual-engine) ─► classify ─► LLM extraction ─► validation
                │             │                              (Pydantic +      (types +
         native-text     tesseract (+easyocr)                 instructor)     business rules)
         fast path       + vision fallback                        │                │
                                                                  ▼                ▼
                                              confidence scoring ─► anomaly check ─► ROUTING
                                                                                      │
                                          ┌───────────────────┬───────────────────────┤
                                          ▼                   ▼                       ▼
                                    auto_approve         fast_review           detailed_review
                                    (zero human cost)          Streamlit review UI
                                                               (corrections feed back)
```

## Highlights

- **Automatic strategy detection** — born-digital PDFs use the native-text fast path; scanned PDFs/images go through preprocessing (grayscale → upscale → autocontrast → denoise → binarize) and OCR.
- **Dual-engine OCR with agreement scoring** — Tesseract always, EasyOCR optionally; where engines agree, confidence rises. Table-aware PSM configuration prevents the classic column-by-column reading failure.
- **Vision-model fallback** — pages where OCR confidence is poor are sent to GPT-4o/Claude vision (when a real provider is configured).
- **Schema-driven extraction** — Pydantic models are the single source of truth. With `instructor`, validation errors are fed back to the LLM for self-correction. Cross-field validators (line items must sum to totals) catch hallucinated numbers.
- **Hallucination detection via source matching** — every extracted value is checked against the OCR text; values not present in the source get low confidence.
- **Business rules + anomaly detection** — vendor allowlist, plausibility caps, and robust-z-score outlier detection against historical documents.
- **Confidence-based routing** — auto-approve / fast review / detailed review; thresholds are config, because the accuracy-vs-labor trade-off is a business decision.
- **Human-in-the-loop review UI** (Streamlit) with color-coded field confidence, inline corrections, and a correction log that powers the improvement loop.
- **Three run modes**: pure-Python sync, Celery + Redis async workers (docker-compose), and `mock` LLM provider so everything runs offline and deterministically.

## Quickstart (offline — no API keys needed)

```bash
pip install -r requirements.txt          # needs tesseract-ocr installed (apt/brew)
python scripts/make_samples.py           # generate demo documents
python scripts/process.py               # batch-process the samples
PYTHONPATH=src streamlit run src/docproc/review_app.py   # review UI
PYTHONPATH=src pytest tests/ -v          # 18 tests
```

## Real LLM extraction

```bash
pip install openai instructor            # or: anthropic instructor
export DOCPROC_LLM_PROVIDER=openai OPENAI_API_KEY=sk-...
python scripts/process.py my_invoice.pdf
```

## Async at scale

```bash
docker compose up --build                # redis + celery workers + review UI on :8501
```

See **KNOWLEDGE.md** for design rationale, concept explanations, and interview prep.
