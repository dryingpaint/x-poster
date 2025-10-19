"""PDF text extraction using PyMuPDF."""

from pathlib import Path
from typing import Any

import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path: str | Path) -> tuple[str, dict[str, Any]]:
    """
    Extract text from a PDF file using PyMuPDF.

    Returns:
        Tuple of (full_text, metadata)
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)

    # Extract metadata
    metadata = {
        "pages": doc.page_count,
        "author": doc.metadata.get("author"),
        "title": doc.metadata.get("title"),
        "subject": doc.metadata.get("subject"),
        "keywords": doc.metadata.get("keywords"),
        "creator": doc.metadata.get("creator"),
        "producer": doc.metadata.get("producer"),
        "creation_date": doc.metadata.get("creationDate"),
        "mod_date": doc.metadata.get("modDate"),
    }

    # Extract text from all pages
    full_text = []
    page_texts = []

    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text()
        page_texts.append({"page": page_num + 1, "text": text})
        full_text.append(text)

    doc.close()

    # Store page-level info in metadata
    metadata["page_texts"] = page_texts

    return "\n\n".join(full_text), metadata


def is_scanned_pdf(pdf_path: str | Path, sample_pages: int = 3) -> bool:
    """
    Heuristic to detect if PDF is scanned (image-based).
    Returns True if appears to be scanned.
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)

    text_chars = 0
    pages_checked = min(sample_pages, doc.page_count)

    for page_num in range(pages_checked):
        page = doc[page_num]
        text = page.get_text()
        text_chars += len(text.strip())

    doc.close()

    # If very little text extracted, likely scanned
    avg_chars_per_page = text_chars / pages_checked
    return avg_chars_per_page < 100


async def process_pdf_file(
    pdf_path: str | Path, apply_ocr: bool = True
) -> tuple[str, dict[str, Any]]:
    """
    Process a PDF file, applying OCR if needed.

    Args:
        pdf_path: Path to PDF file
        apply_ocr: Whether to apply OCR for scanned PDFs

    Returns:
        Tuple of (full_text, metadata)
    """
    from src.ingestion.ocr import apply_ocr_if_needed

    pdf_path = Path(pdf_path)

    # First try direct extraction
    text, metadata = extract_text_from_pdf(pdf_path)

    # Check if we need OCR
    if apply_ocr and is_scanned_pdf(pdf_path):
        text = await apply_ocr_if_needed(pdf_path)
        metadata["ocr_applied"] = True
    else:
        metadata["ocr_applied"] = False

    return text, metadata

