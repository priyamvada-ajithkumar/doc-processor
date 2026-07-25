# KNOWLEDGE.md — Understanding the Document Processor

Read this to *own* the project, not just run it. The interview section at the end assumes you've read the rest.

---

## 1. The problem this solves

Enterprises receive enormous volumes of semi-structured documents — invoices, contracts, delivery notes, forms. Today humans read them and type values into systems (ERP, accounting, CRM). It's slow, expensive, and error-prone. The pipeline automates the read-and-type step while keeping humans in the loop exactly where the machine is uncertain.

The core economic insight: **the value isn't extraction accuracy, it's the auto-approval rate** — the fraction of documents that cost *zero human minutes*. A pipeline that's 95% accurate but can't tell you *which* 5% is wrong forces humans to review everything (worthless). A pipeline that's 90% accurate but *knows* when it's uncertain lets humans review only the uncertain 20% (transformative). This is why half the codebase is confidence estimation, not extraction.

---

## 2. Stage-by-stage: what and why

### 2.1 Ingestion & strategy detection (`ocr.py::load_document`)

A PDF is not one thing. Born-digital PDFs contain real text (extract it directly — perfect fidelity, free). Scanned PDFs contain *pictures* of text (OCR needed — lossy, costly). We detect which by measuring text density per page: below a threshold, we rasterize at 300 DPI and OCR.

Why 300 DPI? Tesseract's accuracy degrades sharply when glyphs are small; 300 DPI is the standard archival scan resolution where character shapes are fully resolved.

### 2.2 Preprocessing (`ocr.py::preprocess`)

Each step targets a specific failure mode:

| Step | Failure mode it fixes |
|---|---|
| Grayscale | Color noise confuses thresholding |
| Upscale (min 1000px) | Small glyphs → misread characters |
| Autocontrast | Washed-out/faded scans |
| Median filter | Salt-and-pepper scanner noise |
| Binarization (threshold 160) | Soft glyph edges → segmentation errors |

### 2.3 Dual-engine OCR + agreement

Tesseract (LSTM-based, 30 years of engineering) and EasyOCR (deep-learning detector+recognizer) fail *differently*. Where two independent systems agree, the probability both made the same mistake is low — so **agreement is evidence of correctness**. Our ensemble confidence = mean engine confidence × (0.5 + 0.5 × agreement). This is the same intuition as ensemble methods in ML generally.

**The PSM bug — remember this story.** During this build, Tesseract initially read the invoice table *column-by-column*: all descriptions, then all quantities, then all prices. Line items became unreconstructable and the cross-field validator (correctly!) rejected the extraction. Fix: `--psm 6` (assume a uniform text block) with `preserve_interword_spaces=1`, which keeps table rows intact. Two lessons: (1) OCR configuration matters as much as the engine; (2) the validation layer caught a silent data-corruption bug — that's exactly what it's for.

### 2.4 Vision-model fallback

When classic OCR confidence falls below a threshold (default 0.55), the page image goes directly to a vision LLM (GPT-4o / Claude). Vision models handle handwriting, weird layouts, and terrible scans far better — at maybe 100× the cost per page. The threshold implements a **cost-quality cascade**: cheap method first, expensive method only when the cheap one demonstrably failed. This cascade pattern shows up everywhere in production AI (small model → big model, cache → compute, heuristic → LLM).

### 2.5 Classification

Invoice vs contract, by keyword heuristic. Deliberately *not* an LLM: classification is cheap, high-volume, and keyword signals are strong. The LLM budget is spent where it earns its cost — extraction. Knowing *where not to use an LLM* is a senior-engineer signal.

### 2.6 LLM extraction (`extraction.py`)

The pattern: **schema-first extraction with retry-on-validation-failure.**

1. Pydantic model defines exactly what to extract (types, bounds, enums).
2. `instructor` patches the OpenAI/Anthropic client so `response_model=Invoice` returns a parsed, validated object.
3. If validation fails (bad date, negative amount, totals don't sum), instructor sends the *validation error back to the LLM* and retries. The model self-corrects. This loop is the single highest-leverage trick in structured extraction — hallucinated totals rarely survive "your line items sum to 1425 but you said total is 1444, fix it."

The system prompt enforces the extraction policy: only what's in the text, `null` for absent values, no inference. Hallucinated defaults ("payment terms: Net 30" when the document says nothing) are the #1 failure mode; we attack it in the prompt *and* in confidence scoring.

**Why a mock provider?** Three reasons production teams do this: CI must run offline and deterministically (LLM outputs vary run-to-run); development iteration shouldn't cost API money; and the mock proves the *pipeline* works independent of the *model* — the schemas are the contract, providers are swappable. When interviewers ask "how do you test LLM systems?", this is the answer.

### 2.7 Confidence scoring (`quality.py`)

Per-field confidence blends three signals:

* **source_match (weight 0.55)** — does the value literally appear in the OCR text? Exact → 1.0, tokens-present → 0.7, absent → 0.2. *A value not present in the source was invented.* This is the cheapest, most effective hallucination detector that exists.
* **ocr_quality (0.30)** — garbage in, garbage out.
* **format_valid (0.15)** — it survived Pydantic parsing.

Overall document confidence uses **weakest-link weighting** (0.6·mean + 0.25·min + 0.15·OCR): one hallucinated critical field should tank the document even when nine fields are perfect, because for an invoice, a wrong total is not "90% correct" — it's wrong.

### 2.8 Validation layers

Three tiers, in order of generality:

1. **Type validation** (Pydantic fields) — dates parse, amounts positive.
2. **Cross-field validation** (model validators) — line items sum to subtotal, subtotal + tax = total, due date after invoice date. This is where hallucinated arithmetic dies.
3. **Business rules** (config-driven) — vendor allowlist, plausibility caps. These encode *organizational* knowledge and change without touching the schema.

### 2.9 Anomaly detection

Robust z-score (median/MAD instead of mean/std) against historical totals. Why robust? Invoice amounts are heavy-tailed: one €80k invoice inflates the standard deviation so much that real outliers hide inside "normal". Median absolute deviation is immune to this. Interviewers love this detail because it shows statistics applied thoughtfully rather than by reflex.

### 2.10 Routing — the business decision

```
error / anomaly / no extraction  → detailed_review
confidence ≥ 0.85, no issues     → auto_approve
confidence ≥ 0.60                → fast_review
else                             → detailed_review
```

The thresholds live in config because they encode a *business* trade-off: raise auto-approve → save labor, admit more errors. Where the line sits depends on the cost of an error (a wrong €50 invoice ≠ a wrong €50k invoice), which is not an engineering decision.

### 2.11 The correction feedback loop

Every human correction records: field, original, corrected, and *why* (extraction error vs validation false-positive). This log is the pipeline's improvement engine:

* Fields with many corrections → improve prompts / few-shot examples for those fields.
* Many "validation_false_positive" → rules are too strict, loosen them.
* Auto-approved documents later corrected → thresholds are too lax, tighten them.

A pipeline without a correction log can't improve. This closed loop is what makes it a *system* rather than a script.

### 2.12 Celery + Redis (async)

`process_document` is a plain function; Celery merely wraps it. Key retry design point: **retry on IOError only** (transient — network, disk), never on data errors — the same malformed PDF fails identically every time, so retrying it just burns compute. Distinguishing transient from permanent failures is the heart of retry policy design.

---

## 3. Glossary

- **OCR (Optical Character Recognition)** — turning pictures of text into text. Tesseract = classic open-source engine; EasyOCR = deep-learning-based.
- **PSM (Page Segmentation Mode)** — Tesseract's assumption about page layout; wrong PSM = scrambled reading order.
- **Binarization** — converting to pure black/white; makes glyph edges crisp for recognition.
- **instructor** — library that binds LLM outputs to Pydantic schemas with automatic validation-error retry.
- **Structured output** — forcing an LLM to produce data matching a schema instead of free text.
- **MAD (Median Absolute Deviation)** — robust spread measure; unlike standard deviation, not distorted by outliers.
- **Human-in-the-loop (HITL)** — system design where humans handle exactly the cases the machine flags as uncertain.
- **Cost-quality cascade** — cheap method first, escalate to the expensive one only on failure.

---

## 4. Interview Q&A

**Q: Walk me through processing one scanned invoice.**
A: The loader detects the PDF has no native text, rasterizes at 300 DPI, and preprocesses each page (grayscale, upscale, contrast, denoise, binarize). Tesseract with table-aware PSM reads it, optionally cross-checked by EasyOCR, giving text plus an agreement-weighted confidence. A keyword classifier labels it an invoice, so the Invoice Pydantic schema drives LLM extraction via instructor — validation errors are fed back for self-correction. Each field is scored: does the value appear in the OCR text, how good was OCR, did it parse? Business rules check the vendor allowlist and plausibility caps; a robust z-score compares the total against history. The blended confidence routes it: auto-approve, fast review, or detailed review, where a reviewer sees color-coded fields side-by-side with the document and every correction is logged to improve prompts and thresholds.

**Q: How do you handle hallucination?**
A: Four layers. The prompt forbids inference and demands null for absent values. Cross-field validators make invented numbers self-inconsistent — a hallucinated total won't match the line-item sum, and instructor feeds that error back for correction. Source matching flags any value not literally present in the OCR text. And whatever survives with low confidence routes to a human. I don't claim zero hallucination; I claim hallucinations don't get *auto-approved*.

**Q: Why not just send the document image straight to GPT-4o vision for everything?**
A: Cost and control. Vision calls cost ~100× OCR per page; at enterprise volume that's the whole budget. The cascade uses the free path when it works and escalates only on measured failure. Also, classic OCR gives per-word confidence data that feeds the routing logic; a vision model gives you text with no calibrated uncertainty.

**Q: What breaks at 100k documents/day?**
A: The design anticipates it: Celery workers scale horizontally, Redis is the broker, and SQLite becomes the bottleneck — swap for PostgreSQL (the storage layer is isolated, so it's one module). The review UI would need auth and pagination. And the correction log becomes big enough to fine-tune a small extraction model on — which is exactly the bridge to my LoRA project.

**Q: Hardest bug?**
A: Tesseract reading a table column-by-column, so line items were unreconstructable. What I love about it: the *validation layer caught it* — line items didn't sum to the subtotal, extraction was rejected, and the document routed to review instead of bad data being auto-approved. The fix was PSM configuration, but the lesson is that defense-in-depth validation converts silent corruption into visible, routable failure.

---

## 5. Demo script (~3.5 min)

1. `python scripts/process.py` — three documents, three paths: native PDF fast lane, scanned image through OCR, contract through a different schema. Point at the confidence and route columns. (45s)
2. Open the Streamlit UI, upload the scanned invoice, walk through metrics: doc type, OCR backend, confidence, route. (45s)
3. Show the extracted JSON next to the OCR text preview; point out the line items summing to the total. (30s)
4. Upload a document from an unknown vendor → warning fires → fast_review. Edit a field in the review queue, save a correction, explain the feedback loop. (60s)
5. Analytics tab: auto-approval rate as the headline metric. (30s)

## 6. Portfolio headline

> *"I built a multi-modal document processing pipeline: automatic native-vs-scan detection, dual-engine OCR with agreement scoring and a vision-model fallback, schema-driven LLM extraction with validation-retry, hallucination detection via source matching, business-rule and anomaly checks, and confidence-based routing to a human review UI with a correction feedback loop. 18 automated tests, offline mock mode, Celery/Redis scaling via docker-compose."*
