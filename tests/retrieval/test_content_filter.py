"""Tests for LLM-based content filtering."""

import asyncio
from datetime import datetime
from pathlib import Path

import pytest

from src.core.models import FilteredWebResult, MediaFile, SearchResult
from src.retrieval.content_filter import (
    extract_relevant_content,
    filter_and_download,
    format_for_agent_history,
)


@pytest.fixture
def sample_search_result():
    """Create a sample search result for testing."""
    return SearchResult(
        source_id="web_0_123456",
        content="""
        Climate change is causing unprecedented global warming. According to the
        IPCC 2023 report, global temperatures have risen 1.1°C since pre-industrial
        times. Scientists warn that we need to limit warming to 1.5°C to avoid
        catastrophic impacts. Renewable energy adoption has increased by 45% in the
        last decade, but fossil fuel subsidies remain a major barrier. The report
        includes several charts showing temperature trends and emission pathways.
        """,
        title="IPCC Climate Report 2023: Key Findings",
        url="https://example.com/ipcc-2023",
        author="Dr. Jane Smith",
        published_at=datetime(2023, 3, 15),
        meta={"domain": "example.com"},
        score=0.95,
        source_type="web",
    )


@pytest.mark.asyncio
async def test_extract_relevant_content(sample_search_result):
    """Test LLM extraction of relevant content."""
    query = "What does the IPCC say about climate change?"

    filtered = await extract_relevant_content(
        sample_search_result,
        query,
        user_context="User wants specific data and statistics",
    )

    # Check basic fields are preserved
    assert filtered.source_id == sample_search_result.source_id
    assert filtered.original_url == sample_search_result.url
    assert filtered.title == sample_search_result.title
    assert filtered.author == sample_search_result.author

    # Check extracted content exists
    assert len(filtered.relevant_text) > 0
    assert 0.0 <= filtered.relevance_score <= 1.0
    assert 0.0 <= filtered.credibility_score <= 1.0

    # Check key points are extracted
    assert isinstance(filtered.key_points, list)

    # Check media files structure
    assert isinstance(filtered.media_files, list)
    for media in filtered.media_files:
        assert isinstance(media, MediaFile)
        assert media.media_type in ["image", "video", "audio", "chart", "unknown"]


@pytest.mark.asyncio
async def test_filter_and_download():
    """Test filtering multiple results with media download."""
    results = [
        SearchResult(
            source_id=f"web_{i}_test",
            content=f"Sample content about topic {i}. " * 20,
            title=f"Article {i}",
            url=f"https://example.com/article-{i}",
            meta={"domain": "example.com"},
            score=0.8,
            source_type="web",
        )
        for i in range(3)
    ]

    query = "What are the main points?"
    output_dir = Path("data/test_media")

    filtered_results = await filter_and_download(
        results,
        query,
        output_dir=output_dir,
        max_concurrent=2,
    )

    # Check all results were processed
    assert len(filtered_results) <= len(results)

    # Check each result
    for filtered in filtered_results:
        assert isinstance(filtered, FilteredWebResult)
        assert filtered.relevant_text
        assert 0.0 <= filtered.relevance_score <= 1.0

    # Cleanup test directory
    if output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)


def test_format_for_agent_history(sample_search_result):
    """Test formatting filtered results for agent message history."""
    # Create a mock filtered result
    filtered = FilteredWebResult(
        source_id=sample_search_result.source_id,
        original_url=sample_search_result.url,
        title=sample_search_result.title,
        author=sample_search_result.author,
        published_at=sample_search_result.published_at,
        relevant_text="Key climate findings from IPCC report...",
        key_points=[
            "Global temperatures up 1.1°C",
            "Need to limit to 1.5°C",
            "Renewable energy up 45%",
        ],
        media_files=[
            MediaFile(
                media_id="test_media_1",
                media_type="image",
                description="Temperature trend chart",
                context="Shows historical warming",
                source_url="https://example.com/chart.png",
                local_path="/path/to/chart.png",
            )
        ],
        credibility_score=0.9,
        relevance_score=0.95,
        meta=sample_search_result.meta,
        extracted_at=datetime.utcnow(),
    )

    formatted = format_for_agent_history(filtered)

    # Check formatting includes all key sections
    assert "## IPCC Climate Report 2023" in formatted
    assert "**URL**" in formatted
    assert "**Author**: Dr. Jane Smith" in formatted
    assert "**Relevance**: 0.95" in formatted
    assert "**Credibility**: 0.90" in formatted
    assert "### Extracted Content" in formatted
    assert "### Key Points" in formatted
    assert "- Global temperatures up 1.1°C" in formatted
    assert "### Media Files" in formatted
    assert "Temperature trend chart" in formatted
    assert "`/path/to/chart.png`" in formatted


@pytest.mark.asyncio
async def test_extract_handles_errors():
    """Test that extraction handles errors gracefully."""
    # Create result with minimal content
    result = SearchResult(
        source_id="web_error_test",
        content="",  # Empty content should still work
        title="Test",
        url="https://example.com/test",
        meta={},
        score=0.5,
        source_type="web",
    )

    filtered = await extract_relevant_content(result, "test query")

    # Should return a result even with error/empty content
    assert isinstance(filtered, FilteredWebResult)
    assert filtered.source_id == result.source_id


def test_media_file_model():
    """Test MediaFile model validation."""
    media = MediaFile(
        media_id="test_123",
        media_type="image",
        description="Test image",
        context="For testing",
        source_url="https://example.com/image.jpg",
        local_path=None,
    )

    assert media.media_id == "test_123"
    assert media.media_type == "image"
    assert media.local_path is None

    # Test with local path
    media.local_path = "/tmp/image.jpg"
    assert media.local_path == "/tmp/image.jpg"


def test_filtered_web_result_model():
    """Test FilteredWebResult model validation."""
    result = FilteredWebResult(
        source_id="test_id",
        original_url="https://example.com",
        title="Test Article",
        relevant_text="Extracted content here",
        key_points=["Point 1", "Point 2"],
        media_files=[],
        credibility_score=0.8,
        relevance_score=0.9,
        meta={"domain": "example.com"},
        extracted_at=datetime.utcnow(),
    )

    assert result.source_id == "test_id"
    assert len(result.key_points) == 2
    assert 0.0 <= result.credibility_score <= 1.0
    assert 0.0 <= result.relevance_score <= 1.0

    # Test validation - scores should be between 0 and 1
    with pytest.raises(Exception):  # Pydantic validation error
        FilteredWebResult(
            source_id="test",
            relevant_text="text",
            credibility_score=1.5,  # Invalid
            relevance_score=0.5,
            extracted_at=datetime.utcnow(),
        )
