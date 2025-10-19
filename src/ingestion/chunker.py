"""Text chunking utilities."""

import re
from typing import Any

import tiktoken


def chunk_text(
    text: str,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
    encoding_name: str = "cl100k_base",
) -> list[str]:
    """
    Split text into chunks based on token count.

    Args:
        text: Text to chunk
        chunk_size: Target size in tokens
        chunk_overlap: Overlap between chunks in tokens
        encoding_name: Tiktoken encoding to use

    Returns:
        List of text chunks
    """
    encoding = tiktoken.get_encoding(encoding_name)

    # Split into sentences first for better boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(encoding.encode(sentence))

        if current_tokens + sentence_tokens > chunk_size and current_chunk:
            # Save current chunk
            chunks.append(" ".join(current_chunk))

            # Start new chunk with overlap
            # Keep last few sentences for overlap
            overlap_sentences = []
            overlap_tokens = 0

            for sent in reversed(current_chunk):
                sent_tokens = len(encoding.encode(sent))
                if overlap_tokens + sent_tokens <= chunk_overlap:
                    overlap_sentences.insert(0, sent)
                    overlap_tokens += sent_tokens
                else:
                    break

            current_chunk = overlap_sentences
            current_tokens = overlap_tokens

        current_chunk.append(sentence)
        current_tokens += sentence_tokens

    # Add remaining chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def create_chunks_with_overlap(
    text: str,
    metadata: dict[str, Any] | None = None,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> list[dict[str, Any]]:
    """
    Create chunks with metadata for database insertion.

    Args:
        text: Text to chunk
        metadata: Optional metadata to include
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap in tokens

    Returns:
        List of chunk dictionaries ready for embedding
    """
    chunks_text = chunk_text(text, chunk_size, chunk_overlap)

    chunks = []
    for i, chunk_text in enumerate(chunks_text):
        chunk_data = {
            "content": chunk_text,
            "meta": {
                "chunk_index": i,
                "total_chunks": len(chunks_text),
                **(metadata or {}),
            },
        }
        chunks.append(chunk_data)

    return chunks

