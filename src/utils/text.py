"""Text processing utilities."""

import re

import tiktoken


def truncate_text(text: str, max_length: int = 1000) -> str:
    """Truncate text to max length, trying to break at sentence boundaries."""
    if len(text) <= max_length:
        return text

    # Try to truncate at sentence boundary
    truncated = text[:max_length]
    last_period = truncated.rfind(".")
    last_question = truncated.rfind("?")
    last_exclaim = truncated.rfind("!")

    boundary = max(last_period, last_question, last_exclaim)

    if boundary > max_length * 0.8:  # If we found a good boundary
        return text[: boundary + 1]
    else:
        return truncated + "..."


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove special characters that might cause issues
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    return text.strip()


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in text."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))

