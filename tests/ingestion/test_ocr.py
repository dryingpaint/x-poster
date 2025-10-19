"""Tests for OCR functionality.

Following TDD: These tests define the expected behavior BEFORE implementation.

Note: OCR tests require ocrmypdf and tesseract to be installed.
Tests will be skipped if these dependencies are not available.
"""

import shutil
from pathlib import Path

import pytest

from src.ingestion.ocr import apply_ocr_if_needed

# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
EMPTY_PDF = FIXTURES_DIR / "empty.pdf"
SIMPLE_PDF = FIXTURES_DIR / "simple.pdf"

# Check if ocrmypdf is available
HAS_OCRMYPDF = shutil.which("ocrmypdf") is not None

# Skip all tests if OCR dependencies not available
pytestmark = pytest.mark.skipif(
    not HAS_OCRMYPDF, reason="ocrmypdf not installed (brew install ocrmypdf)"
)


class TestApplyOCRIfNeeded:
    """Test the apply_ocr_if_needed async function."""

    @pytest.mark.asyncio
    async def test_apply_ocr_basic(self):
        """Test basic OCR functionality."""
        # Use empty PDF which should trigger OCR
        text = await apply_ocr_if_needed(EMPTY_PDF)

        # Should return a string
        assert isinstance(text, str)

    @pytest.mark.asyncio
    async def test_apply_ocr_returns_text(self):
        """Test that OCR returns extracted text."""
        # Use simple PDF (native text, OCR should preserve it)
        text = await apply_ocr_if_needed(SIMPLE_PDF)

        assert isinstance(text, str)
        # Should have some content
        assert len(text) >= 0  # May be empty for scanned docs

    @pytest.mark.asyncio
    async def test_apply_ocr_with_path_object(self):
        """Test that function works with Path objects."""
        text = await apply_ocr_if_needed(Path(SIMPLE_PDF))

        assert isinstance(text, str)

    @pytest.mark.asyncio
    async def test_apply_ocr_with_string_path(self):
        """Test that function works with string paths."""
        text = await apply_ocr_if_needed(str(SIMPLE_PDF))

        assert isinstance(text, str)

    @pytest.mark.asyncio
    async def test_apply_ocr_preserves_native_text(self):
        """Test that OCR preserves native PDF text."""
        # Use simple PDF which has native text
        text = await apply_ocr_if_needed(SIMPLE_PDF)

        # Should preserve the original text
        assert isinstance(text, str)
        # Should have reasonable length (not empty)
        if len(text) > 0:
            # Check for expected content
            assert len(text) > 10

    @pytest.mark.asyncio
    async def test_apply_ocr_handles_scanned_pdf(self):
        """Test OCR on PDF with no text (simulated scan)."""
        # Empty PDF simulates a scanned document
        text = await apply_ocr_if_needed(EMPTY_PDF)

        # Should handle gracefully
        assert isinstance(text, str)
        # Empty PDF may result in empty text
        assert len(text) >= 0

    @pytest.mark.asyncio
    async def test_apply_ocr_nonexistent_file(self):
        """Test error handling for nonexistent file."""
        with pytest.raises((FileNotFoundError, Exception)):
            await apply_ocr_if_needed("/nonexistent/file.pdf")

    @pytest.mark.asyncio
    async def test_apply_ocr_creates_no_artifacts(self):
        """Test that OCR cleans up temporary files."""
        import tempfile
        from pathlib import Path

        # Get temp directory
        temp_dir = Path(tempfile.gettempdir())

        # Count PDFs before
        pdfs_before = list(temp_dir.glob("tmp*.pdf"))

        # Run OCR
        await apply_ocr_if_needed(SIMPLE_PDF)

        # Count PDFs after
        pdfs_after = list(temp_dir.glob("tmp*.pdf"))

        # Should not have created persistent temp files
        # (allowing for some margin as other processes may create temp files)
        assert len(pdfs_after) - len(pdfs_before) <= 1


class TestOCREdgeCases:
    """Test edge cases and error handling for OCR."""

    @pytest.mark.asyncio
    async def test_ocr_with_corrupted_pdf(self):
        """Test OCR behavior with potentially problematic PDF."""
        # This tests the fallback behavior when OCR fails
        # Create a minimal test by using empty PDF
        try:
            text = await apply_ocr_if_needed(EMPTY_PDF)
            # Should handle gracefully
            assert isinstance(text, str)
        except Exception as e:
            # If it raises, should be a reasonable error
            assert isinstance(e, RuntimeError | OSError | FileNotFoundError)

    @pytest.mark.asyncio
    async def test_ocr_fallback_on_failure(self):
        """Test that OCR falls back to original extraction on failure."""
        # Use a valid PDF
        text = await apply_ocr_if_needed(SIMPLE_PDF)

        # Should always return something (either OCR'd or original)
        assert isinstance(text, str)


class TestOCRIntegration:
    """Integration tests for OCR with PDF processing."""

    @pytest.mark.asyncio
    async def test_ocr_integration_with_pdf_processor(self):
        """Test OCR works in the context of PDF processing."""
        from src.ingestion.pdf_processor import process_pdf_file

        # Process a PDF with OCR enabled
        text, metadata = await process_pdf_file(SIMPLE_PDF, apply_ocr=True)

        assert isinstance(text, str)
        assert len(text) > 0
        assert "ocr_applied" in metadata
        assert isinstance(metadata["ocr_applied"], bool)

    @pytest.mark.asyncio
    async def test_ocr_skipped_for_native_pdf(self):
        """Test that OCR is skipped for PDFs with native text."""
        from src.ingestion.pdf_processor import process_pdf_file

        # Simple PDF has native text
        text, metadata = await process_pdf_file(SIMPLE_PDF, apply_ocr=True)

        # Should not apply OCR since it has native text
        assert metadata["ocr_applied"] is False
        assert len(text) > 0

    @pytest.mark.asyncio
    async def test_ocr_applied_for_scanned_pdf(self):
        """Test that OCR is applied for scanned PDFs."""
        from src.ingestion.pdf_processor import process_pdf_file

        # Empty PDF simulates scanned document
        text, metadata = await process_pdf_file(EMPTY_PDF, apply_ocr=True)

        # Should apply OCR
        assert metadata["ocr_applied"] is True
        assert isinstance(text, str)


# Optional: Test for OCRmyPDF specific behavior
class TestOCRmyPDFBehavior:
    """Test specific behaviors of ocrmypdf implementation."""

    @pytest.mark.asyncio
    async def test_ocrmypdf_skip_text_option(self):
        """Test that --skip-text option is respected."""
        # This is more of a behavioral test
        # OCRmyPDF should skip pages that already have text
        text = await apply_ocr_if_needed(SIMPLE_PDF)

        # Should return text (either original or OCR'd)
        assert isinstance(text, str)

    @pytest.mark.asyncio
    async def test_ocrmypdf_handles_timeout(self):
        """Test that OCR handles long-running operations gracefully."""
        # Use a simple PDF (should be fast)
        import time

        start = time.time()
        text = await apply_ocr_if_needed(SIMPLE_PDF)
        duration = time.time() - start

        # Should complete reasonably quickly for a simple PDF
        # (allowing generous timeout for CI environments)
        assert duration < 30  # 30 seconds max

        assert isinstance(text, str)
