"""Tests for PDF processing functionality.

Following TDD: These tests define the expected behavior BEFORE implementation.
"""

from pathlib import Path

import pytest

from src.ingestion.pdf_processor import (
    extract_text_from_pdf,
    is_scanned_pdf,
    process_pdf_file,
)

# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
SIMPLE_PDF = FIXTURES_DIR / "simple.pdf"
MULTIPAGE_PDF = FIXTURES_DIR / "multipage.pdf"
EMPTY_PDF = FIXTURES_DIR / "empty.pdf"


class TestExtractTextFromPDF:
    """Test the extract_text_from_pdf function."""

    def test_extract_text_simple_pdf(self):
        """Test extracting text from a simple PDF."""
        text, metadata = extract_text_from_pdf(SIMPLE_PDF)

        # Should extract text
        assert isinstance(text, str)
        assert len(text) > 0

        # Should extract metadata
        assert isinstance(metadata, dict)
        assert "pages" in metadata
        assert metadata["pages"] > 0

    def test_extract_text_returns_metadata(self):
        """Test that metadata is properly extracted."""
        text, metadata = extract_text_from_pdf(SIMPLE_PDF)

        # Should have standard metadata fields
        assert "pages" in metadata
        assert "author" in metadata
        assert "title" in metadata
        assert "subject" in metadata
        assert "keywords" in metadata
        assert "creator" in metadata
        assert "producer" in metadata
        assert "creation_date" in metadata
        assert "mod_date" in metadata

    def test_extract_text_multipage(self):
        """Test extracting text from multi-page PDF."""
        text, metadata = extract_text_from_pdf(MULTIPAGE_PDF)

        # Should extract text from all pages
        assert len(text) > 100
        assert metadata["pages"] == 3

        # Should have page-level metadata
        assert "page_texts" in metadata
        assert len(metadata["page_texts"]) == 3

        # Each page should have text
        for page_info in metadata["page_texts"]:
            assert "page" in page_info
            assert "text" in page_info

    def test_extract_text_preserves_content(self):
        """Test that expected content is present."""
        text, metadata = extract_text_from_pdf(SIMPLE_PDF)

        # Should contain expected content from the fixture
        assert "Test Document" in text or "test" in text.lower()

    def test_extract_text_with_path_object(self):
        """Test that function works with Path objects."""
        text, metadata = extract_text_from_pdf(Path(SIMPLE_PDF))

        assert isinstance(text, str)
        assert len(text) > 0

    def test_extract_text_with_string_path(self):
        """Test that function works with string paths."""
        text, metadata = extract_text_from_pdf(str(SIMPLE_PDF))

        assert isinstance(text, str)
        assert len(text) > 0

    def test_extract_text_empty_pdf(self):
        """Test handling of PDF with no text."""
        text, metadata = extract_text_from_pdf(EMPTY_PDF)

        # Should return empty or minimal text
        assert isinstance(text, str)
        assert metadata["pages"] >= 0

    def test_extract_text_nonexistent_file(self):
        """Test error handling for nonexistent file."""
        with pytest.raises((FileNotFoundError, Exception)):
            extract_text_from_pdf("/nonexistent/file.pdf")

    def test_metadata_author_from_pdf(self):
        """Test that author metadata is extracted."""
        text, metadata = extract_text_from_pdf(SIMPLE_PDF)

        # Our test fixture should have author metadata
        if metadata.get("author"):
            assert isinstance(metadata["author"], str)
            assert len(metadata["author"]) > 0

    def test_metadata_title_from_pdf(self):
        """Test that title metadata is extracted."""
        text, metadata = extract_text_from_pdf(SIMPLE_PDF)

        # Our test fixture should have title metadata
        if metadata.get("title"):
            assert isinstance(metadata["title"], str)
            assert "Test Document" in metadata["title"]


class TestIsScannedPDF:
    """Test the is_scanned_pdf function."""

    def test_is_scanned_native_pdf(self):
        """Test that native PDFs are not detected as scanned."""
        is_scanned = is_scanned_pdf(SIMPLE_PDF)

        # Simple PDF has native text, should not be scanned
        assert isinstance(is_scanned, bool)
        assert is_scanned is False

    def test_is_scanned_empty_pdf(self):
        """Test that empty PDFs are detected as potentially scanned."""
        is_scanned = is_scanned_pdf(EMPTY_PDF)

        # Empty PDF has no text, should be detected as scanned
        assert isinstance(is_scanned, bool)
        assert is_scanned is True

    def test_is_scanned_multipage(self):
        """Test scanned detection on multi-page PDF."""
        is_scanned = is_scanned_pdf(MULTIPAGE_PDF)

        # Multipage PDF has text, should not be scanned
        assert isinstance(is_scanned, bool)
        assert is_scanned is False

    def test_is_scanned_with_sample_pages(self):
        """Test that sample_pages parameter works."""
        # Should work with different sample sizes
        is_scanned_1 = is_scanned_pdf(MULTIPAGE_PDF, sample_pages=1)
        is_scanned_2 = is_scanned_pdf(MULTIPAGE_PDF, sample_pages=2)

        assert isinstance(is_scanned_1, bool)
        assert isinstance(is_scanned_2, bool)

    def test_is_scanned_with_path_object(self):
        """Test that function works with Path objects."""
        is_scanned = is_scanned_pdf(Path(SIMPLE_PDF))

        assert isinstance(is_scanned, bool)

    def test_is_scanned_with_string_path(self):
        """Test that function works with string paths."""
        is_scanned = is_scanned_pdf(str(SIMPLE_PDF))

        assert isinstance(is_scanned, bool)

    def test_is_scanned_nonexistent_file(self):
        """Test error handling for nonexistent file."""
        with pytest.raises((FileNotFoundError, Exception)):
            is_scanned_pdf("/nonexistent/file.pdf")


class TestProcessPDFFile:
    """Test the process_pdf_file async function."""

    @pytest.mark.asyncio
    async def test_process_pdf_basic(self):
        """Test basic PDF processing."""
        text, metadata = await process_pdf_file(SIMPLE_PDF, apply_ocr=False)

        # Should extract text
        assert isinstance(text, str)
        assert len(text) > 0

        # Should have metadata
        assert isinstance(metadata, dict)
        assert "pages" in metadata
        assert "ocr_applied" in metadata

    @pytest.mark.asyncio
    async def test_process_pdf_without_ocr(self):
        """Test processing with OCR disabled."""
        text, metadata = await process_pdf_file(SIMPLE_PDF, apply_ocr=False)

        # Should not apply OCR
        assert metadata["ocr_applied"] is False

    @pytest.mark.asyncio
    async def test_process_pdf_native_text(self):
        """Test that native PDFs don't trigger OCR."""
        text, metadata = await process_pdf_file(SIMPLE_PDF, apply_ocr=True)

        # Should detect native text and skip OCR
        assert metadata["ocr_applied"] is False
        assert len(text) > 0

    @pytest.mark.asyncio
    async def test_process_pdf_multipage(self):
        """Test processing multi-page PDF."""
        text, metadata = await process_pdf_file(MULTIPAGE_PDF, apply_ocr=False)

        # Should process all pages
        assert len(text) > 100
        assert metadata["pages"] == 3

    @pytest.mark.asyncio
    async def test_process_pdf_with_path_object(self):
        """Test that function works with Path objects."""
        text, metadata = await process_pdf_file(Path(SIMPLE_PDF), apply_ocr=False)

        assert isinstance(text, str)
        assert len(text) > 0

    @pytest.mark.asyncio
    async def test_process_pdf_with_string_path(self):
        """Test that function works with string paths."""
        text, metadata = await process_pdf_file(str(SIMPLE_PDF), apply_ocr=False)

        assert isinstance(text, str)
        assert len(text) > 0

    @pytest.mark.asyncio
    async def test_process_pdf_returns_metadata(self):
        """Test that all expected metadata is returned."""
        text, metadata = await process_pdf_file(SIMPLE_PDF, apply_ocr=False)

        # Should have OCR flag
        assert "ocr_applied" in metadata
        assert isinstance(metadata["ocr_applied"], bool)

        # Should have page count
        assert "pages" in metadata
        assert metadata["pages"] > 0

    @pytest.mark.asyncio
    async def test_process_pdf_preserves_original_metadata(self):
        """Test that original PDF metadata is preserved."""
        text, metadata = await process_pdf_file(SIMPLE_PDF, apply_ocr=False)

        # Should still have original metadata
        assert "author" in metadata
        assert "title" in metadata
        assert "page_texts" in metadata

    @pytest.mark.asyncio
    async def test_process_pdf_nonexistent_file(self):
        """Test error handling for nonexistent file."""
        with pytest.raises((FileNotFoundError, Exception)):
            await process_pdf_file("/nonexistent/file.pdf")


class TestPDFProcessingEdgeCases:
    """Test edge cases in PDF processing."""

    def test_extract_text_page_separation(self):
        """Test that pages are properly separated in output."""
        text, metadata = extract_text_from_pdf(MULTIPAGE_PDF)

        # Text should be separated (likely by newlines)
        # Multi-page should have multiple sections
        lines = text.split("\n")
        assert len(lines) > metadata["pages"]  # More lines than pages

    def test_metadata_handles_missing_fields(self):
        """Test that missing metadata fields are handled gracefully."""
        text, metadata = extract_text_from_pdf(EMPTY_PDF)

        # Should have all fields, but they may be None
        assert "author" in metadata
        assert "title" in metadata
        assert "subject" in metadata

        # None values are acceptable for missing metadata
        if metadata["author"] is not None:
            assert isinstance(metadata["author"], str)

    @pytest.mark.asyncio
    async def test_process_pdf_empty_file(self):
        """Test processing of empty/minimal PDF."""
        text, metadata = await process_pdf_file(EMPTY_PDF, apply_ocr=False)

        # Should not crash
        assert isinstance(text, str)
        assert isinstance(metadata, dict)

    def test_extract_text_unicode(self):
        """Test that unicode characters are handled correctly."""
        # This test assumes simple.pdf might have some special chars
        text, metadata = extract_text_from_pdf(SIMPLE_PDF)

        # Should return valid string (may contain unicode)
        assert isinstance(text, str)
        # Should be encodable to UTF-8
        text.encode("utf-8")
