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


class SearchResult(BaseModel):
    """A search result from internal or web retrieval."""

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


class MediaFile(BaseModel):
    """Reference to a media file (image, video, etc.) from web content."""

    media_id: str
    media_type: str  # 'image', 'video', 'audio', etc.
    description: str
    context: str  # Why this media is relevant
    source_url: str | None = None
    local_path: str | None = None  # Path to downloaded file


class FilteredWebResult(BaseModel):
    """LLM-filtered web search result with extracted relevant content."""

    source_id: str
    original_url: str | None = None
    title: str | None = None
    author: str | None = None
    published_at: datetime | None = None
    relevant_text: str  # LLM-extracted relevant content
    key_points: list[str] = Field(default_factory=list)
    media_files: list[MediaFile] = Field(default_factory=list)
    credibility_score: float = Field(ge=0.0, le=1.0)
    relevance_score: float = Field(ge=0.0, le=1.0)
    meta: dict[str, Any] = Field(default_factory=dict)
    extracted_at: datetime


class GenerateResponse(BaseModel):
    """Response with generated tweets."""

    variants: list[Tweet] = Field(default_factory=list)
    thread: list[Tweet] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)

