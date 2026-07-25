"""Ingestion and OCR: any document in, (text, confidence, backend) out.

Strategy selection (automatic):

    PDF ──► native text extraction (PyMuPDF)
       │        │ enough text? ──► done (confidence 0.99, backend "native")
       │        ▼ too little (scanned/image PDF)
       │      rasterize pages ──► OCR path
    image ─────────────────────► OCR path

OCR path = preprocessing (grayscale, upscale, binarize, denoise)
         + Tesseract (always) + EasyOCR (if installed)
         + agreement scoring between engines.

Why dual-engine? OCR engines fail differently: Tesseract struggles with
low contrast, EasyOCR with dense tables. Where they AGREE, we can trust
the text; where they diverge, confidence drops — and low confidence is a
*signal* the pipeline uses (vision-model fallback, review routing), not
just a number in a log.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageFilter, ImageOps

from .config import get_settings

log = logging.getLogger(__name__)

try:  # optional second engine (heavy: pulls torch)
    import easyocr  # type: ignore

    _EASYOCR_READER = None  # lazy init — model download is expensive

    def _easyocr():
        global _EASYOCR_READER
        if _EASYOCR_READER is None:
            _EASYOCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
        return _EASYOCR_READER

    _HAS_EASYOCR = True
except ImportError:  # pragma: no cover
    _HAS_EASYOCR = False


@dataclass
class OCRResult:
    text: str
    confidence: float  # 0..1
    backend: str       # "native" | "tesseract" | "tesseract+easyocr" | "vision"
    pages: int


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def preprocess(img: Image.Image) -> Image.Image:
    """Classic OCR preprocessing chain. Each step targets a failure mode:

    grayscale  -> color noise confuses binarization
    upscale    -> Tesseract wants >=300 DPI-equivalent glyph sizes
    autocontrast -> washed-out scans
    median filter -> salt-and-pepper scanner noise
    binarize   -> crisp black/white glyph edges
    """
    img = ImageOps.grayscale(img)
    if min(img.size) < 1000:
        scale = 1000 / min(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = img.point(lambda p: 255 if p > 160 else 0)
    return img


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------


def _tesseract_ocr(img: Image.Image, lang: str) -> tuple[str, float]:
    """Text + mean word confidence from Tesseract's TSV output.

    --psm 6 ("assume a single uniform block of text") with preserved
    inter-word spaces keeps TABLE ROWS together. The default PSM
    segments whitespace-separated columns as separate blocks and reads
    them column-by-column, which destroys line items — one of the most
    common invoice-OCR failure modes.
    """
    config = "--psm 6 -c preserve_interword_spaces=1"
    data = pytesseract.image_to_data(
        img, lang=lang, config=config, output_type=pytesseract.Output.DICT
    )
    confs = [float(c) for w, c in zip(data["text"], data["conf"]) if w.strip() and float(c) >= 0]
    text = pytesseract.image_to_string(img, lang=lang, config=config)
    mean_conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0
    return text, round(mean_conf, 4)


def _easyocr_ocr(img: Image.Image) -> tuple[str, float]:
    import numpy as np

    result = _easyocr().readtext(np.array(img))
    words = [r[1] for r in result]
    confs = [r[2] for r in result]
    return " ".join(words), round(sum(confs) / len(confs), 4) if confs else ("", 0.0)


def _agreement(a: str, b: str) -> float:
    """Normalized similarity between two engines' outputs (0..1).

    We compare on normalized tokens so whitespace/case differences don't
    count as disagreement — only actual character-level reading errors.
    """
    norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip()
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def run_ocr(img: Image.Image) -> tuple[str, float, str]:
    settings = get_settings()
    img = preprocess(img)
    t_text, t_conf = _tesseract_ocr(img, settings.ocr_lang)

    if _HAS_EASYOCR and settings.use_easyocr:
        e_text, e_conf = _easyocr_ocr(img)
        agree = _agreement(t_text, e_text)
        # Ensemble confidence: engine confidences weighted by agreement.
        combined = round(((t_conf + e_conf) / 2) * (0.5 + 0.5 * agree), 4)
        # Prefer the higher-confidence engine's text
        text = t_text if t_conf >= e_conf else e_text
        return text, combined, "tesseract+easyocr"
    return t_text, t_conf, "tesseract"


# ---------------------------------------------------------------------------
# Vision-model fallback (for handwriting / terrible scans)
# ---------------------------------------------------------------------------


def vision_fallback(img: Image.Image) -> tuple[str, float] | None:
    """Send the page image to a vision LLM when classic OCR fails.

    Only wired for real providers; in mock mode we return None and the
    pipeline proceeds with the low-confidence OCR text (and routes the
    document to detailed review — degraded but honest).
    """
    settings = get_settings()
    if settings.llm_provider == "anthropic":
        try:
            import anthropic, base64

            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="PNG")
            b64 = base64.standard_b64encode(buf.getvalue()).decode()
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model=settings.anthropic_model,
                max_tokens=4000,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                        {"type": "text", "text": "Transcribe all text in this document image. Preserve structure (tables as aligned text). Output only the transcription."},
                    ],
                }],
            )
            return msg.content[0].text, 0.90
        except Exception as exc:  # pragma: no cover
            log.warning("vision fallback failed: %s", exc)
    elif settings.llm_provider == "openai":
        try:
            import base64
            from openai import OpenAI

            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="PNG")
            b64 = base64.standard_b64encode(buf.getvalue()).decode()
            client = OpenAI()
            resp = client.chat.completions.create(
                model=settings.openai_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe all text in this document image. Preserve structure. Output only the transcription."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }],
            )
            return resp.choices[0].message.content, 0.90
        except Exception as exc:  # pragma: no cover
            log.warning("vision fallback failed: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_document(path: str | Path) -> OCRResult:
    """Any supported file -> text with confidence and provenance."""
    settings = get_settings()
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        doc = fitz.open(path)
        native_text = "\n".join(page.get_text() for page in doc)
        # Strategy detection: enough native text per page => born-digital PDF
        if len(native_text.strip()) >= settings.min_native_text_chars * max(len(doc), 1) * 0.3 \
                and len(native_text.strip()) >= settings.min_native_text_chars:
            return OCRResult(native_text, 0.99, "native", len(doc))
        # Scanned PDF: rasterize at 300 DPI and OCR each page
        texts, confs, backend = [], [], "tesseract"
        for page in doc:
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text, conf, backend = run_ocr(img)
            if conf < settings.vision_fallback_threshold:
                vf = vision_fallback(img)
                if vf:
                    text, conf, backend = vf[0], vf[1], "vision"
            texts.append(text)
            confs.append(conf)
        return OCRResult("\n".join(texts), round(sum(confs) / len(confs), 4), backend, len(doc))

    if suffix in {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}:
        img = Image.open(path)
        text, conf, backend = run_ocr(img)
        if conf < settings.vision_fallback_threshold:
            vf = vision_fallback(img)
            if vf:
                text, conf, backend = vf[0], vf[1], "vision"
        return OCRResult(text, conf, backend, 1)

    if suffix in {".txt", ".md"}:
        return OCRResult(path.read_text(encoding="utf-8", errors="replace"), 1.0, "native", 1)

    raise ValueError(f"Unsupported file type: {suffix}")
