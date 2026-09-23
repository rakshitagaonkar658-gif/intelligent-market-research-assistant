"""
rag/ingestion.py
────────────────
Document parsing layer for the RAG pipeline.

Supports:
  - PDF    via pypdf
  - DOCX   via python-docx
  - TXT    plain text
  - Images via pytesseract OCR (text-only — NOT vision/image understanding)

All parsers return plain Python strings.  The image parser always prefixes
its output with an OCR disclaimer label so callers can surface it in the UI.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# OCR disclaimer — always prepended to image-extracted text
_OCR_LABEL = (
    "[OCR-extracted text — image analysis is text-only, not vision-based. "
    "The system cannot interpret charts, diagrams, or visual layouts.]\n\n"
)


# ── PDF ───────────────────────────────────────────────────────────────────────

def parse_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file using pypdf."""
    try:
        from pypdf import PdfReader  # noqa: PLC0415
    except ImportError:
        raise ImportError(
            "pypdf is required for PDF parsing. Run: pip install pypdf"
        )

    reader = PdfReader(io.BytesIO(file_bytes))
    pages: list[str] = []
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"[Page {page_num}]\n{text}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not extract text from PDF page %d: %s", page_num, exc)

    full_text = "\n\n".join(pages)
    if not full_text.strip():
        logger.warning("PDF extraction produced no text — the file may be image-based. Consider uploading as an image for OCR.")
        return "[PDF contained no extractable text. If this is a scanned document, try uploading it as a PNG/JPG image.]"
    return full_text


# ── DOCX ──────────────────────────────────────────────────────────────────────

def parse_docx(file_bytes: bytes) -> str:
    """Extract text from a DOCX file using python-docx."""
    try:
        from docx import Document  # noqa: PLC0415
    except ImportError:
        raise ImportError(
            "python-docx is required for DOCX parsing. Run: pip install python-docx"
        )

    doc = Document(io.BytesIO(file_bytes))
    paragraphs: list[str] = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            paragraphs.append(text)

    # Also extract text from tables
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                paragraphs.append(row_text)

    full_text = "\n\n".join(paragraphs)
    if not full_text.strip():
        return "[DOCX contained no extractable text paragraphs.]"
    return full_text


# ── TXT ───────────────────────────────────────────────────────────────────────

def parse_txt(file_bytes: bytes) -> str:
    """Decode a plain-text file, trying UTF-8 then latin-1 as fallback."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return file_bytes.decode(encoding)
        except (UnicodeDecodeError, ValueError):
            continue
    return file_bytes.decode("utf-8", errors="replace")


# ── Image / OCR ───────────────────────────────────────────────────────────────

def parse_image(file_bytes: bytes) -> str:
    """
    Extract text from an image using Tesseract OCR.

    IMPORTANT: This is TEXT EXTRACTION ONLY.  The system cannot interpret
    charts, graphs, visual layouts, or any non-textual image content.
    The returned string always begins with an OCR disclaimer label.

    Returns the OCR disclaimer even if Tesseract is not installed, so the
    caller always receives a valid (non-empty) string.
    """
    try:
        from PIL import Image  # noqa: PLC0415
        import pytesseract  # noqa: PLC0415
    except ImportError:
        logger.warning(
            "pytesseract or Pillow not installed — OCR unavailable. "
            "Run: pip install pytesseract Pillow  (and install Tesseract binary)"
        )
        return (
            _OCR_LABEL
            + "[OCR unavailable: pytesseract or Pillow is not installed. "
            "Install them and the Tesseract binary to enable image text extraction.]"
        )

    try:
        image = Image.open(io.BytesIO(file_bytes))
        # Convert to RGB if needed (handles PNG with alpha channel)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        ocr_text = pytesseract.image_to_string(image).strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("OCR failed: %s", exc)
        return _OCR_LABEL + f"[OCR failed: {exc}]"

    if not ocr_text:
        return (
            _OCR_LABEL
            + "[OCR produced no text. The image may not contain readable text, "
            "or the image quality may be too low for OCR.]"
        )

    return _OCR_LABEL + ocr_text


# ── Dispatcher ────────────────────────────────────────────────────────────────

def ingest_file(file_name: str, file_bytes: bytes) -> str:
    """
    Dispatch to the appropriate parser based on the file extension.

    Parameters
    ----------
    file_name : str
        Original filename (used to determine format from extension).
    file_bytes : bytes
        Raw file content.

    Returns
    -------
    str
        Extracted text ready for chunking and embedding.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    suffix = Path(file_name).suffix.lower()

    if suffix == ".pdf":
        logger.info("Ingesting PDF: %s", file_name)
        return parse_pdf(file_bytes)

    if suffix in (".docx", ".doc"):
        logger.info("Ingesting DOCX: %s", file_name)
        return parse_docx(file_bytes)

    if suffix == ".txt":
        logger.info("Ingesting TXT: %s", file_name)
        return parse_txt(file_bytes)

    if suffix in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"):
        logger.info("Ingesting image via OCR: %s", file_name)
        return parse_image(file_bytes)

    raise ValueError(
        f"Unsupported file type: '{suffix}'. "
        "Supported formats: PDF, DOCX, TXT, PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP."
    )


def is_ocr_result(text: str) -> bool:
    """Return True if the text was produced by OCR (starts with the OCR label)."""
    return text.startswith(_OCR_LABEL[:30])
