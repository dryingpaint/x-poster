"""Embedding generation using BGE-M3."""

from sentence_transformers import SentenceTransformer

from src.core.config import get_config


# Global embedder instance (lazy loaded)
_embedder = None


def get_embedder() -> SentenceTransformer:
    """Get or create embedder instance."""
    global _embedder

    if _embedder is None:
        config = get_config()
        _embedder = SentenceTransformer(config.embedding_model)

    return _embedder


def embed_text(text: str) -> list[float]:
    """
    Generate embedding for a single text.

    Args:
        text: Text to embed

    Returns:
        Embedding vector as list of floats
    """
    embedder = get_embedder()
    embedding = embedder.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def embed_batch(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """
    Generate embeddings for a batch of texts.

    Args:
        texts: List of texts to embed
        batch_size: Batch size for encoding

    Returns:
        List of embedding vectors
    """
    embedder = get_embedder()
    embeddings = embedder.encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )
    return [emb.tolist() for emb in embeddings]

