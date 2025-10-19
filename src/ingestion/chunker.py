"""Text chunking utilities (paragraph-based with token fallback)."""

import re
from typing import Any

import tiktoken

def _normalize_text_preserving_paragraphs(text: str) -> str:
    """
    Normalize text to preserve paragraph boundaries while removing hard wraps.

    - Collapses single newlines (line wraps) into spaces
    - Preserves blank lines (\n\n) as paragraph separators
    - Removes hyphenation at line breaks (e.g., "exam-\nple" -> "example")
    """
    if not text:
        return ""

    # Remove hyphenation across line breaks
    text = re.sub(r"-\n\s*", "", text)

    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse single newlines to spaces, keep blank lines
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Normalize multiple blank lines to exactly two newlines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


def _split_into_paragraphs(text: str) -> list[str]:
    """Split normalized text into paragraphs on blank lines."""
    if not text:
        return []
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(
    text: str,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
    encoding_name: str = "cl100k_base",
) -> list[str]:
    """
    Create chunks per paragraph. If a paragraph is extremely long (beyond a high
    token threshold), fall back to token-based windows with overlap.

    Args:
        text: Text to chunk
        chunk_size: Token window size for fallback splitting of long paragraphs
        chunk_overlap: Token overlap for fallback splitting
        encoding_name: Tiktoken encoding to use

    Returns:
        List of paragraph or token-window chunks
    """
    normalized = _normalize_text_preserving_paragraphs(text)
    paragraphs = _split_into_paragraphs(normalized)

    encoding = tiktoken.get_encoding(encoding_name)

    # Reasonable high threshold to consider a paragraph too long
    MAX_TOKENS_PER_PARAGRAPH = 2000

    if chunk_size <= 0:
        # Safe default for fallback windowing
        chunk_size = 1200

    step = max(1, chunk_size - max(0, chunk_overlap))

    chunks: list[str] = []
    for paragraph in paragraphs:
        tokens = encoding.encode(paragraph)
        if len(tokens) <= MAX_TOKENS_PER_PARAGRAPH:
            chunks.append(paragraph)
            continue

        # Fallback: token-based windowing with overlap
        for start in range(0, len(tokens), step):
            end = start + chunk_size
            window_tokens = tokens[start:end]
            window_text = encoding.decode(window_tokens).strip()
            if window_text:
                chunks.append(window_text)

    return chunks


def create_chunks_with_overlap(
    text: str,
    metadata: dict[str, Any] | None = None,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> list[dict[str, Any]]:
    """
    Create paragraph-based chunks with metadata for database insertion.

    Args:
        text: Text to chunk into paragraphs
        metadata: Optional metadata to include
        chunk_size: Unused (kept for compatibility)
        chunk_overlap: Unused (kept for compatibility)

    Returns:
        List of chunk dictionaries ready for embedding
    """
    chunks_text = chunk_text(text, chunk_size, chunk_overlap)

    chunks = []
    for i, chunk_content in enumerate(chunks_text):
        chunk_data = {
            "content": chunk_content,
            "meta": {
                "chunk_index": i,
                "total_chunks": len(chunks_text),
                **(metadata or {}),
            },
        }
        chunks.append(chunk_data)

    return chunks

