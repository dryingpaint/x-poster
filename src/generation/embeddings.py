"""Embedding generation using OpenAI (default) or local SentenceTransformers fallback.

Note: No local caching; embeddings for internal chunks are stored in Postgres
and fetched during retrieval to avoid recomputation.
"""

from typing import Optional

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from src.core.config import get_config

# Global embedder instances (lazy loaded)
_embedder: Optional[SentenceTransformer] = None
_openai_client: Optional[OpenAI] = None
_provider_logged: bool = False


def get_openai_client() -> OpenAI:
    """Get or create OpenAI client for embeddings."""
    global _openai_client
    if _openai_client is None:
        config = get_config()
        _openai_client = OpenAI(api_key=config.openai_api_key)
    return _openai_client


def get_embedder() -> SentenceTransformer:
    """Get or create local embedder instance (fallback)."""
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
    config = get_config()
    global _provider_logged
    # Try OpenAI first (no local caching)
    try:
        client = get_openai_client()
        resp = client.embeddings.create(
            model=config.openai_embedding_model,
            input=text,
            dimensions=config.embedding_dim,
        )
        vector = resp.data[0].embedding
        if not _provider_logged:
            print(f"Embeddings provider: OpenAI {config.openai_embedding_model} (dim={config.embedding_dim})")
            _provider_logged = True
        return vector
    except Exception as e:
        print(f"OpenAI embeddings failed (single): {e}")
        # Fallback to local model
        embedder = get_embedder()
        embedding = embedder.encode(text, normalize_embeddings=True)
        if not _provider_logged:
            print(f"Embeddings provider: Local SentenceTransformer {get_config().embedding_model}")
            _provider_logged = True
        return embedding.tolist()


def embed_batch(texts: list[str], batch_size: int = 4) -> list[list[float]]:
    """
    Generate embeddings for a batch of texts.

    Args:
        texts: List of texts to embed
        batch_size: Batch size for encoding

    Returns:
        List of embedding vectors
    """
    config = get_config()

    # Try OpenAI first in batches
    global _provider_logged
    # Try OpenAI first (no local caching)
    try:
        client = get_openai_client()
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            print(f"   Processing embeddings {i+1}-{min(i+batch_size, len(texts))} of {len(texts)} (OpenAI)")
            resp = client.embeddings.create(
                model=config.openai_embedding_model,
                input=batch_texts,
                dimensions=config.embedding_dim,
            )
            all_embeddings.extend([item.embedding for item in resp.data])

        if not _provider_logged:
            print(f"Embeddings provider: OpenAI {config.openai_embedding_model} (dim={config.embedding_dim})")
            _provider_logged = True
        return all_embeddings
    except Exception as e:
        print(f"OpenAI embeddings failed (batch): {e}")
        # Fallback to local model
        embedder = get_embedder()
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            print(f"   Processing embeddings {i+1}-{min(i+batch_size, len(texts))} of {len(texts)} (Local)")

            # Truncate very long texts to avoid memory issues
            batch_texts = [text[:1500] if len(text) > 1500 else text for text in batch_texts]

            batch_embeddings = embedder.encode(
                batch_texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
            )
            all_embeddings.extend([emb.tolist() for emb in batch_embeddings])

        if not _provider_logged:
            print(f"Embeddings provider: Local SentenceTransformer {get_config().embedding_model}")
            _provider_logged = True
        return all_embeddings
