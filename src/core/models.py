"""Data models for the agent tweeter system."""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ItemKind(str, Enum):
    """Types of items that can be stored."""

    PDF = "pdf"
    DOC = "doc"
    NOTE = "note"
    SLIDE = "slide"
    IMAGE = "image"
    AUDIO = "audio"
    CODE = "code"
    OTHER = "other"


class Item(BaseModel):
    """Internal item (multimodal document)."""

    item_id: UUID
    kind: ItemKind
    title: str | None = None
    source_uri: str  # storage://bucket/key or external ref
    content_text: str  # canonical text (OCR or caption if needed)
    meta: dict[str, Any] = Field(default_factory=dict)  # {pages, author, tags, ...}
    created_at: datetime
    updated_at: datetime


class ItemChunk(BaseModel):
    """Chunked text representation for retrieval."""

    chunk_id: UUID
    item_id: UUID
    content: str
    embedding: list[float] | None = None
    # Note: tsv (tsvector) is handled at DB level


class WebCache(BaseModel):
    """Cached web page."""

    url_hash: str
    url: str
    domain: str
    title: str | None = None
    published_at: datetime | None = None
    content: str
    embedding: list[float] | None = None
    fetched_at: datetime


class MediaFile(BaseModel):
    """Reference to a media file (image, video, etc.) from web content."""

    media_id: str
    media_type: str  # 'image', 'video', 'audio', etc.
    description: str
    context: str  # Why this media is relevant
    source_url: str | None = None
    local_path: str | None = None  # Path to downloaded file


class SearchResult(BaseModel):
    """A search result from internal or web retrieval.

    For LLM-filtered web results, additional structured fields are populated.
    """

    source_id: str  # Unique identifier for this passage
    content: str
    title: str | None = None
    url: str | None = None  # For web results
    source_uri: str | None = None  # For internal results
    author: str | None = None
    published_at: datetime | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0
    source_type: str = "unknown"  # 'internal' or 'web'

    # Optional fields for LLM-filtered web results
    key_points: list[str] = Field(default_factory=list)
    media_files: list[MediaFile] = Field(default_factory=list)
    credibility_score: float | None = None
    relevance_score: float | None = None

    def to_context_text(self, include_metadata: bool = True) -> str:
        """Convert to formatted text for LLM context.

        Creates a comprehensive text representation including structured fields,
        metadata, and local file references.

        Args:
            include_metadata: Whether to include scores and metadata

        Returns:
            Formatted text suitable for LLM context
        """
        lines = []

        # Header with title and source
        if self.title:
            lines.append(f"## {self.title}")
        else:
            lines.append(f"## Source {self.source_id}")

        # Metadata
        meta_parts = []
        if self.url:
            meta_parts.append(f"URL: {self.url}")
        elif self.source_uri:
            meta_parts.append(f"Source: {self.source_uri}")

        if self.author:
            meta_parts.append(f"Author: {self.author}")

        if self.published_at:
            meta_parts.append(f"Published: {self.published_at.strftime('%Y-%m-%d')}")

        if include_metadata and self.credibility_score is not None:
            meta_parts.append(f"Credibility: {self.credibility_score:.2f}")

        if include_metadata and self.relevance_score is not None:
            meta_parts.append(f"Relevance: {self.relevance_score:.2f}")

        if meta_parts:
            lines.append(" | ".join(meta_parts))

        lines.append("")  # Blank line

        # Main content
        lines.append(self.content)

        # Key points (if available)
        if self.key_points:
            lines.append("")
            lines.append("### Key Points")
            for point in self.key_points:
                lines.append(f"• {point}")

        # Media files (if available)
        if self.media_files:
            lines.append("")
            lines.append("### Referenced Media")
            for media in self.media_files:
                lines.append(f"• **{media.media_type.capitalize()}**: {media.description}")
                if media.local_path:
                    lines.append(f"  File: {media.local_path}")
                if media.context:
                    lines.append(f"  Context: {media.context}")

        return "\n".join(lines)


class EvidenceFact(BaseModel):
    """A single fact extracted from evidence."""

    fact: str  # Paraphrased fact
    quote: str  # Direct quote (<=20 words)
    source_id: str
    url: str | None = None
    title: str | None = None
    author: str | None = None
    published_at: datetime | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class EvidencePack(BaseModel):
    """Collection of evidence facts assembled from retrieval results."""

    facts: list[EvidenceFact]
    sources: dict[str, SearchResult]  # source_id -> SearchResult


class Citation(BaseModel):
    """Citation linking a number to a source."""

    n: int  # Citation number [1], [2], etc.
    source_id: str


class Tweet(BaseModel):
    """A single tweet variant."""

    text: str
    citations: list[Citation] = Field(default_factory=list)


class TweetThread(BaseModel):
    """A thread of tweets."""

    tweets: list[Tweet]


class Source(BaseModel):
    """Source information for response."""

    source_id: str
    title: str | None = None
    url: str | None = None
    source_uri: str | None = None
    author: str | None = None
    published_at: datetime | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class GenerateRequest(BaseModel):
    """Request to generate tweets."""

    prompt: str
    max_variants: int = 3
    max_thread_tweets: int = 6


class GenerateResponse(BaseModel):
    """Response with generated tweets."""

    variants: list[Tweet] = Field(default_factory=list)
    thread: list[Tweet] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
