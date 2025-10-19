"""Tests for embedding generation module.

Following TDD principles from PARALLEL_DEVELOPMENT.md
"""

import pytest

from src.generation.embeddings import embed_batch, embed_text, get_embedder


class TestEmbedder:
    """Test embedder model loading and caching."""

    def test_get_embedder_returns_model(self):
        """Test that get_embedder returns a SentenceTransformer model."""
        embedder = get_embedder()
        assert embedder is not None
        assert hasattr(embedder, "encode")

    def test_get_embedder_caches_instance(self):
        """Test that get_embedder returns the same instance (singleton)."""
        embedder1 = get_embedder()
        embedder2 = get_embedder()
        assert embedder1 is embedder2  # Should be same object


class TestEmbedText:
    """Test single text embedding generation."""

    def test_embed_text_returns_vector(self):
        """Test that embed_text returns a vector of correct dimension."""
        text = "AI safety research is important"
        embedding = embed_text(text)

        assert isinstance(embedding, list)
        assert len(embedding) == 1024  # BGE-M3 dimension
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_text_is_normalized(self):
        """Test that embeddings are normalized (L2 norm ~= 1.0)."""
        text = "Machine learning model evaluation"
        embedding = embed_text(text)

        # Calculate L2 norm
        import math

        norm = math.sqrt(sum(x**2 for x in embedding))
        assert abs(norm - 1.0) < 0.01  # Should be very close to 1.0

    def test_embed_text_different_texts_different_embeddings(self):
        """Test that different texts produce different embeddings."""
        text1 = "Artificial intelligence"
        text2 = "Banana recipes"

        emb1 = embed_text(text1)
        emb2 = embed_text(text2)

        assert emb1 != emb2  # Should be different vectors

    def test_embed_text_same_text_same_embedding(self):
        """Test that same text produces same embedding (deterministic)."""
        text = "Reproducible embeddings"

        emb1 = embed_text(text)
        emb2 = embed_text(text)

        assert emb1 == emb2  # Should be identical

    def test_embed_text_handles_empty_string(self):
        """Test that embed_text handles empty strings gracefully."""
        embedding = embed_text("")

        assert isinstance(embedding, list)
        assert len(embedding) == 1024

    def test_embed_text_handles_long_text(self):
        """Test that embed_text handles long text (>512 tokens)."""
        # Create a long text
        long_text = "AI safety research " * 200  # ~400 words

        embedding = embed_text(long_text)

        assert isinstance(embedding, list)
        assert len(embedding) == 1024

    def test_embed_text_handles_unicode(self):
        """Test that embed_text handles Unicode characters."""
        text = "机器学习 🤖 深度学习"

        embedding = embed_text(text)

        assert isinstance(embedding, list)
        assert len(embedding) == 1024


class TestEmbedBatch:
    """Test batch embedding generation."""

    def test_embed_batch_returns_vectors(self):
        """Test that embed_batch returns list of vectors."""
        texts = [
            "First text about AI",
            "Second text about ML",
            "Third text about DL",
        ]

        embeddings = embed_batch(texts)

        assert isinstance(embeddings, list)
        assert len(embeddings) == 3
        assert all(len(emb) == 1024 for emb in embeddings)

    def test_embed_batch_empty_list(self):
        """Test that embed_batch handles empty list."""
        embeddings = embed_batch([])

        assert isinstance(embeddings, list)
        assert len(embeddings) == 0

    def test_embed_batch_single_text(self):
        """Test that embed_batch works with single text."""
        texts = ["Single text"]

        embeddings = embed_batch(texts)

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 1024

    def test_embed_batch_matches_individual_embeds(self):
        """Test that batch embeddings match individual embeddings."""
        texts = ["Text one", "Text two"]

        # Batch embeddings
        batch_embeddings = embed_batch(texts)

        # Individual embeddings
        individual_embeddings = [embed_text(t) for t in texts]

        # Should be identical (or very close due to floating point)
        for batch_emb, indiv_emb in zip(batch_embeddings, individual_embeddings):
            # Check if embeddings are very similar (allow tiny numerical differences)
            differences = [abs(b - i) for b, i in zip(batch_emb, indiv_emb)]
            max_diff = max(differences)
            assert max_diff < 1e-6  # Very small tolerance

    def test_embed_batch_custom_batch_size(self):
        """Test that embed_batch respects custom batch size."""
        texts = [f"Text number {i}" for i in range(50)]

        # Should work with different batch sizes
        embeddings = embed_batch(texts, batch_size=10)

        assert len(embeddings) == 50
        assert all(len(emb) == 1024 for emb in embeddings)

    def test_embed_batch_large_batch(self):
        """Test that embed_batch handles large batches efficiently."""
        # Create 100 texts
        texts = [f"Document about topic {i}" for i in range(100)]

        embeddings = embed_batch(texts, batch_size=16)

        assert len(embeddings) == 100
        assert all(len(emb) == 1024 for emb in embeddings)

    def test_embed_batch_all_normalized(self):
        """Test that all batch embeddings are normalized."""
        import math

        texts = ["Text A", "Text B", "Text C"]
        embeddings = embed_batch(texts)

        for embedding in embeddings:
            norm = math.sqrt(sum(x**2 for x in embedding))
            assert abs(norm - 1.0) < 0.01


class TestEmbeddingSimilarity:
    """Test semantic similarity properties of embeddings."""

    def test_similar_texts_high_similarity(self):
        """Test that semantically similar texts have high cosine similarity."""
        text1 = "machine learning and artificial intelligence"
        text2 = "AI and ML technologies"

        emb1 = embed_text(text1)
        emb2 = embed_text(text2)

        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(emb1, emb2))

        # Should be high similarity (>0.5)
        assert dot_product > 0.5, f"Expected high similarity, got {dot_product}"

    def test_dissimilar_texts_low_similarity(self):
        """Test that semantically different texts have lower similarity."""
        text1 = "quantum physics and particle acceleration"
        text2 = "cooking pasta recipes"

        emb1 = embed_text(text1)
        emb2 = embed_text(text2)

        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(emb1, emb2))

        # Should be lower similarity than related texts
        # (not necessarily negative, but definitely <0.7)
        assert dot_product < 0.7, f"Expected low similarity, got {dot_product}"


class TestPerformance:
    """Test performance characteristics of embeddings."""

    def test_batch_faster_than_individual(self, benchmark=None):
        """Test that batch embedding is more efficient than individual calls.

        Note: This is a qualitative test - batch should complete reasonably fast.
        """
        import time

        texts = [f"Document {i}" for i in range(20)]

        # Measure batch embedding time
        start = time.time()
        embed_batch(texts, batch_size=16)
        batch_time = time.time() - start

        # Should complete in reasonable time
        assert batch_time < 10.0, f"Batch embedding too slow: {batch_time}s"

    def test_embedder_model_cached(self):
        """Test that embedder model is cached and not reloaded."""
        import time

        # First call (might load model)
        start = time.time()
        embed_text("First call")
        first_call_time = time.time() - start

        # Second call (should use cached model)
        start = time.time()
        embed_text("Second call")
        second_call_time = time.time() - start

        # Second call should be much faster (no model loading)
        # Allow first call up to 30s for model download, second should be <1s
        assert second_call_time < 1.0, f"Second call not using cache: {second_call_time}s"
