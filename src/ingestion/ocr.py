"""OCR processing using OCRmyPDF."""

import asyncio
import tempfile
from pathlib import Path


async def apply_ocr_if_needed(pdf_path: str | Path) -> str:
    """
    Apply OCR to a PDF using ocrmypdf.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text from OCR'd PDF
    """
    pdf_path = Path(pdf_path)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        tmp_output = Path(tmp_file.name)

    try:
        # Run ocrmypdf
        process = await asyncio.create_subprocess_exec(
            "ocrmypdf",
            "--skip-text",  # Only OCR pages without text
            "--optimize", "0",  # Don't optimize, we just want text
            "--output-type", "pdf",
            str(pdf_path),
            str(tmp_output),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            # OCR failed, fall back to original
            from src.ingestion.pdf_processor import extract_text_from_pdf

            text, _ = extract_text_from_pdf(pdf_path)
            return text

        # Extract text from OCR'd PDF
        from src.ingestion.pdf_processor import extract_text_from_pdf

        text, _ = extract_text_from_pdf(tmp_output)
        return text

    finally:
        # Clean up temp file
        if tmp_output.exists():
            tmp_output.unlink()

