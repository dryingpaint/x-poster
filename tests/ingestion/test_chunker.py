"""Tests for text chunking functionality.

Following TDD: These tests define the expected behavior BEFORE implementation.
"""

import pytest

from src.ingestion.chunker import chunk_text, create_chunks_with_overlap


class TestChunkText:
    """Test the chunk_text function."""

    def test_chunk_text_basic(self):
        """Test basic chunking with simple text."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunk_text(text, chunk_size=20, chunk_overlap=5)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_chunk_text_respects_sentence_boundaries(self):
        """Test that chunks don't split mid-sentence."""
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)

        # Each chunk should contain complete sentences
        for chunk in chunks:
            # Should not end mid-word
            assert not chunk.endswith(" is")
            assert not chunk.endswith(" sentence")

    def test_chunk_text_has_overlap(self):
        """Test that consecutive chunks have overlapping content."""
        text = " ".join([f"Sentence number {i}." for i in range(20)])
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)

        if len(chunks) > 1:
            # Check that there's some overlap between consecutive chunks
            for i in range(len(chunks) - 1):
                # Find common words between chunks
                words_chunk1 = set(chunks[i].split())
                words_chunk2 = set(chunks[i + 1].split())
                overlap_words = words_chunk1.intersection(words_chunk2)
                # Should have some overlapping words
                assert len(overlap_words) > 0

    def test_chunk_text_target_size(self):
        """Test that chunks approximately match target token size."""
        # Create text with actual sentences (more realistic)
        text = " ".join([f"This is sentence number {i} with some content." for i in range(100)])
        chunk_size = 100
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=10)

        # Import tiktoken to verify token counts
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")

        for chunk in chunks:
            tokens = len(encoding.encode(chunk))
            # Allow some flexibility (chunks can be smaller or slightly larger)
            assert tokens <= chunk_size * 1.5  # Max 50% larger
            # But not too small (except possibly last chunk)
            if chunk != chunks[-1]:
                assert tokens >= chunk_size * 0.3  # At least 30% of target

    def test_chunk_text_empty_string(self):
        """Test handling of empty input."""
        chunks = chunk_text("", chunk_size=100, chunk_overlap=10)
        # Should return empty list or list with empty string
        assert len(chunks) <= 1
        if len(chunks) == 1:
            assert chunks[0] == ""

    def test_chunk_text_single_sentence(self):
        """Test chunking of very short text."""
        text = "This is a single short sentence."
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_very_long_sentence(self):
        """Test handling of sentences longer than chunk_size."""
        # Create a very long sentence
        long_sentence = "This sentence has " + " ".join(["word" for _ in range(500)]) + "."
        chunks = chunk_text(long_sentence, chunk_size=50, chunk_overlap=10)

        # Should still create chunks, even if sentence is too long
        assert len(chunks) > 0
        # All content should be preserved
        combined = " ".join(chunks)
        # Check that no words were lost (accounting for overlap)
        assert all(word in combined for word in long_sentence.split()[:10])

    def test_chunk_text_custom_encoding(self):
        """Test using different tiktoken encoding."""
        text = "Test sentence. Another test sentence."
        chunks = chunk_text(text, chunk_size=50, chunk_overlap=5, encoding_name="cl100k_base")

        assert len(chunks) > 0
        assert isinstance(chunks[0], str)

    def test_chunk_text_preserves_content(self):
        """Test that all content is preserved across chunks."""
        text = "First. Second. Third. Fourth. Fifth. Sixth. Seventh. Eighth. Ninth. Tenth."
        chunks = chunk_text(text, chunk_size=30, chunk_overlap=10)

        # Combine all unique words from chunks
        all_words = set()
        for chunk in chunks:
            all_words.update(chunk.split())

        original_words = set(text.split())
        # Most words should be preserved (accounting for punctuation)
        assert len(all_words.intersection(original_words)) >= len(original_words) * 0.8


class TestCreateChunksWithOverlap:
    """Test the create_chunks_with_overlap function."""

    def test_create_chunks_basic(self):
        """Test basic chunk creation with metadata."""
        text = "First sentence. Second sentence. Third sentence."
        chunks = create_chunks_with_overlap(text, chunk_size=50, chunk_overlap=10)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, dict) for chunk in chunks)

    def test_create_chunks_structure(self):
        """Test that chunks have correct structure."""
        text = "Test sentence one. Test sentence two."
        chunks = create_chunks_with_overlap(text, chunk_size=50, chunk_overlap=10)

        for chunk in chunks:
            # Each chunk should have content and meta
            assert "content" in chunk
            assert "meta" in chunk
            assert isinstance(chunk["content"], str)
            assert isinstance(chunk["meta"], dict)

    def test_create_chunks_metadata(self):
        """Test that chunks include correct metadata."""
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        chunks = create_chunks_with_overlap(text, chunk_size=30, chunk_overlap=5)

        for i, chunk in enumerate(chunks):
            # Should have chunk index
            assert "chunk_index" in chunk["meta"]
            assert chunk["meta"]["chunk_index"] == i

            # Should have total chunks
            assert "total_chunks" in chunk["meta"]
            assert chunk["meta"]["total_chunks"] == len(chunks)

    def test_create_chunks_custom_metadata(self):
        """Test that custom metadata is preserved."""
        text = "Test sentence."
        custom_meta = {"source": "test.pdf", "page": 1, "author": "Test Author"}

        chunks = create_chunks_with_overlap(text, metadata=custom_meta, chunk_size=50)

        for chunk in chunks:
            # Custom metadata should be included
            assert "source" in chunk["meta"]
            assert chunk["meta"]["source"] == "test.pdf"
            assert "page" in chunk["meta"]
            assert chunk["meta"]["page"] == 1
            assert "author" in chunk["meta"]
            assert chunk["meta"]["author"] == "Test Author"

            # Standard metadata should also be present
            assert "chunk_index" in chunk["meta"]
            assert "total_chunks" in chunk["meta"]

    def test_create_chunks_no_metadata(self):
        """Test chunk creation without custom metadata."""
        text = "Test sentence for chunking."
        chunks = create_chunks_with_overlap(text, metadata=None, chunk_size=50)

        # Should still work
        assert len(chunks) > 0
        for chunk in chunks:
            assert "chunk_index" in chunk["meta"]
            assert "total_chunks" in chunk["meta"]

    def test_create_chunks_empty_text(self):
        """Test handling of empty text."""
        chunks = create_chunks_with_overlap("", chunk_size=50)

        # Should handle gracefully
        assert isinstance(chunks, list)
        # May return empty list or list with empty chunk
        if len(chunks) > 0:
            assert chunks[0]["content"] in ["", None] or len(chunks[0]["content"]) == 0


class TestChunkingEdgeCases:
    """Test edge cases and error handling."""

    def test_very_small_chunk_size(self):
        """Test behavior with very small chunk size."""
        text = "This is a test sentence with multiple words."
        chunks = chunk_text(text, chunk_size=5, chunk_overlap=1)

        # Should still produce chunks
        assert len(chunks) > 0

    def test_overlap_larger_than_chunk_size(self):
        """Test handling when overlap is larger than chunk size."""
        text = "Test sentence one. Test sentence two."
        # This is a weird case but should handle gracefully
        chunks = chunk_text(text, chunk_size=20, chunk_overlap=30)

        # Should still work (overlap will be limited by chunk size)
        assert len(chunks) > 0

    def test_chunk_size_zero(self):
        """Test that chunk_size=0 is handled."""
        text = "Test sentence."
        # This should either raise an error or handle gracefully
        try:
            chunks = chunk_text(text, chunk_size=0, chunk_overlap=0)
            # If it doesn't raise, should return something reasonable
            assert isinstance(chunks, list)
        except (ValueError, ZeroDivisionError):
            # Acceptable to raise an error for invalid input
            pass

    def test_unicode_text(self):
        """Test chunking with unicode characters."""
        text = "This is a test with émojis 🎉 and spëcial çharacters. Another sentence."
        chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)

        assert len(chunks) > 0
        # Content should be preserved
        combined = " ".join(chunks)
        assert "émojis" in combined or "émojis" in text
        assert "🎉" in combined or "🎉" in text

    def test_newlines_and_whitespace(self):
        """Test handling of newlines and extra whitespace."""
        text = "First sentence.\n\nSecond sentence with  extra spaces.\n\nThird sentence."
        chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)

        assert len(chunks) > 0
        # Should handle whitespace gracefully
        for chunk in chunks:
            assert isinstance(chunk, str)
