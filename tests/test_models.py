"""Test data models."""

from datetime import datetime
from uuid import uuid4

from src.core.models import (
    Citation,
    EvidenceFact,
    ItemKind,
    SearchResult,
    Tweet,
)


def test_citation_model():
    """Test Citation model."""
    citation = Citation(n=1, source_id="test_source")
    assert citation.n == 1
    assert citation.source_id == "test_source"


def test_tweet_model():
    """Test Tweet model."""
    tweet = Tweet(
        text="Test tweet [1]",
        citations=[Citation(n=1, source_id="src_1")],
    )
    assert "[1]" in tweet.text
    assert len(tweet.citations) == 1


def test_search_result_model():
    """Test SearchResult model."""
    result = SearchResult(
        source_id="test_1",
        content="Test content",
        title="Test Title",
        url="https://example.com",
        score=0.95,
        source_type="web",
    )
    assert result.source_id == "test_1"
    assert result.score == 0.95


def test_evidence_fact_model():
    """Test EvidenceFact model."""
    fact = EvidenceFact(
        fact="Test fact",
        quote="Test quote",
        source_id="src_1",
        url="https://example.com",
        confidence=0.9,
    )
    assert fact.confidence == 0.9
    assert 0.0 <= fact.confidence <= 1.0


def test_item_kind_enum():
    """Test ItemKind enum."""
    assert ItemKind.PDF == "pdf"
    assert ItemKind.IMAGE == "image"

