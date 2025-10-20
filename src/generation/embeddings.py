"""Embedding generation using OpenAI (default) or local SentenceTransformers fallback.

Note: No local caching; embeddings for internal chunks are stored in Postgres
and fetched during retrieval to avoid recomputation.
"""

from typing import Optional

from openai import OpenAI
import tiktoken
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
        # Pre-truncate by tokens to avoid OpenAI context errors
        enc = tiktoken.get_encoding("cl100k_base")
        max_tokens = config.openai_embedding_max_tokens
        toks = enc.encode(text)
        if len(toks) > max_tokens:
            toks = toks[:max_tokens]
            text = enc.decode(toks)
        resp = client.embeddings.create(
            model=config.openai_embedding_model,
            input=text,
            dimensions=config.embedding_dim,
        )
        vector = resp.data[0].embedding
        return vector
    except Exception:
        # Fallback to local model
        embedder = get_embedder()
        embedding = embedder.encode(text, normalize_embeddings=True)
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
    client = get_openai_client()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        try:
            # Pre-truncate each item by tokens
            enc = tiktoken.get_encoding("cl100k_base")
            max_tokens = config.openai_embedding_max_tokens
            truncated = []
            for t in batch_texts:
                toks = enc.encode(t)
                if len(toks) > max_tokens:
                    toks = toks[:max_tokens]
                    t = enc.decode(toks)
                truncated.append(t)
            resp = client.embeddings.create(
                model=config.openai_embedding_model,
                input=truncated,
                dimensions=config.embedding_dim,
            )
            all_embeddings.extend([item.embedding for item in resp.data])
        except Exception:
            # Retry per-item: prefer OpenAI single-call, then fallback to local for that item only
            for t in batch_texts:
                try:
                    vec = embed_text(t)  # tries OpenAI single, falls back to local if needed
                    all_embeddings.append(vec)
                except Exception:
                    # Final fallback: local embedding directly
                    embedder = get_embedder()
                    t_trunc = t[:1500] if len(t) > 1500 else t
                    vec = embedder.encode([t_trunc], batch_size=1, normalize_embeddings=True, show_progress_bar=False)[0]
                    all_embeddings.append(vec.tolist())
    return all_embeddings
