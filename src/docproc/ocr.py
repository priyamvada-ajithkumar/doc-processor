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
import shutil
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageOps

from .config import get_settings

log = logging.getLogger(__name__)

# The Windows (UB Mannheim) installer does not reliably add itself to PATH,
# so pytesseract's bare `tesseract` subprocess call fails with WinError 2 even
# on a correct install. Fall back to the default install location.
if sys.platform == "win32" and not shutil.which("tesseract"):
    _WIN_TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if _WIN_TESSERACT.exists():
        pytesseract.pytesseract.tesseract_cmd = str(_WIN_TESSERACT)
    else:
        log.warning("tesseract not found on PATH or at %s", _WIN_TESSERACT)

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


#: Target short-edge size for the upscale step. A page scanned at 300 DPI is
#: ~2480px on its short edge; anything materially below that has glyphs too
#: small for Tesseract regardless of how large the canvas is.
UPSCALE_TARGET_PX = 2000


def _otsu_threshold(img: Image.Image) -> int:
    """Otsu's method: the grey level that best separates ink from paper.

    A fixed threshold guesses at the scan's exposure and is wrong for any
    scan that isn't the one it was tuned on. Otsu derives the split from
    the image's own histogram by maximizing between-class variance.
    """
    hist = img.histogram()
    total = sum(hist)
    sum_all = sum(i * h for i, h in enumerate(hist))
    sum_bg = weight_bg = 0
    best_variance, best_threshold = -1.0, 128
    for level in range(256):
        weight_bg += hist[level]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += level * hist[level]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance > best_variance:
            best_variance, best_threshold = variance, level
    return best_threshold


def preprocess(img: Image.Image) -> Image.Image:
    """Classic OCR preprocessing chain. Each step targets a failure mode:

    grayscale  -> color noise confuses binarization
    upscale    -> Tesseract wants >=300 DPI-equivalent glyph sizes
    autocontrast -> washed-out scans
    binarize   -> crisp black/white glyph edges

    No median filter: measured against both clean and artificially noisy
    samples it lost accuracy in each case (it erodes thin glyph strokes,
    which costs more than the speckle it removes). Denoising belongs after
    upscaling and at a radius tied to glyph size, not as a fixed size-3
    pass — until that exists, not denoising reads better.
    """
    img = ImageOps.grayscale(img)
    # Scale on the SHORT edge against a DPI-equivalent target. Gating on
    # `min(size) < 1000` instead let a 1400x1500 page (~120 DPI) through
    # untouched, and Tesseract returned noise for it at 0.16 confidence.
    if min(img.size) < UPSCALE_TARGET_PX:
        scale = UPSCALE_TARGET_PX / min(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    img = ImageOps.autocontrast(img)
    threshold = _otsu_threshold(img)
    img = img.point(lambda p: 255 if p > threshold else 0)
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
