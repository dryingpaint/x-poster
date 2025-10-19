"""Tests for gap analysis module.

Following TDD principles from PARALLEL_DEVELOPMENT.md
Tests LLM-based gap identification for targeted web search.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.models import SearchResult
from src.generation.gap_analysis import analyze_gaps


@pytest.fixture
def mock_internal_results():
    """Create mock internal search results for testing."""
    return [
        SearchResult(
            source_id="doc1",
            content="AI safety research has been growing. Early work focused on alignment.",
            title="Introduction to AI Safety",
            url=None,
            score=0.9,
            source_type="internal",
        ),
        SearchResult(
            source_id="doc2",
            content="Machine learning models can exhibit unexpected behaviors. Testing is crucial.",
            title="ML Model Testing",
            url=None,
            score=0.85,
            source_type="internal",
        ),
        SearchResult(
            source_id="doc3",
            content="Deep learning architectures have evolved significantly over the past decade.",
            title="Deep Learning Evolution",
            url=None,
            score=0.8,
            source_type="internal",
        ),
    ]


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI API response."""
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content=json.dumps(
                    {
                        "gaps": [
                            {
                                "type": "statistics",
                                "query": "AI safety research funding 2024 statistics",
                            },
                            {
                                "type": "expert_opinion",
                                "query": "AI safety experts opinions 2024",
                            },
                            {"type": "visual", "query": "AI alignment progress chart 2024"},
                        ]
                    }
                )
            )
        )
    ]
    return mock_response


class TestAnalyzeGapsFewInternalResults:
    """Test gap analysis when internal results are insufficient."""

    @pytest.mark.asyncio
    async def test_few_results_returns_original_query(self):
        """Test that with <3 internal results, returns original query."""
        query = "AI safety in 2025"
        internal_results = [
            SearchResult(
                source_id="doc1",
                content="Brief mention of AI safety",
                title="AI Overview",
                url=None,
                score=0.7,
                source_type="internal",
            )
        ]

        gaps = await analyze_gaps(query, internal_results)

        assert len(gaps) == 1
        assert gaps[0] == query

    @pytest.mark.asyncio
    async def test_empty_results_returns_original_query(self):
        """Test that with no internal results, returns original query."""
        query = "Machine learning trends"
        internal_results = []

        gaps = await analyze_gaps(query, internal_results)

        assert len(gaps) == 1
        assert gaps[0] == query


class TestAnalyzeGapsWithSufficientResults:
    """Test gap analysis with sufficient internal results."""

    @pytest.mark.asyncio
    async def test_analyze_gaps_returns_queries(self, mock_internal_results, mock_openai_response):
        """Test that analyze_gaps returns list of gap-filling queries."""
        query = "AI safety research progress"

        with patch("src.generation.gap_analysis.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
            mock_openai.return_value = mock_client

            gaps = await analyze_gaps(query, mock_internal_results)

            assert isinstance(gaps, list)
            assert len(gaps) > 0
            assert len(gaps) <= 5  # Should limit to max 5 queries
            assert all(isinstance(q, str) for q in gaps)

    @pytest.mark.asyncio
    async def test_analyze_gaps_calls_openai_correctly(
        self, mock_internal_results, mock_openai_response
    ):
        """Test that analyze_gaps calls OpenAI API with correct parameters."""
        query = "AI safety research progress"

        with patch("src.generation.gap_analysis.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
            mock_openai.return_value = mock_client

            await analyze_gaps(query, mock_internal_results)

            # Verify API was called
            mock_client.chat.completions.create.assert_called_once()

            # Verify call parameters
            call_args = mock_client.chat.completions.create.call_args
            assert call_args.kwargs["model"] == "gpt-3.5-turbo"
            assert call_args.kwargs["temperature"] == 0.3
            assert call_args.kwargs["response_format"] == {"type": "json_object"}

            # Verify messages structure
            messages = call_args.kwargs["messages"]
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            assert query in messages[1]["content"]

    @pytest.mark.asyncio
    async def test_analyze_gaps_includes_internal_content_in_prompt(
        self, mock_internal_results, mock_openai_response
    ):
        """Test that analyze_gaps includes internal content in the prompt."""
        query = "AI safety research"

        with patch("src.generation.gap_analysis.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
            mock_openai.return_value = mock_client

            await analyze_gaps(query, mock_internal_results)

            # Get the user prompt
            call_args = mock_client.chat.completions.create.call_args
            user_content = call_args.kwargs["messages"][1]["content"]

            # Should include content from internal results
            assert mock_internal_results[0].title in user_content
            assert mock_internal_results[0].content[:100] in user_content

    @pytest.mark.asyncio
    async def test_analyze_gaps_limits_internal_results_to_ten(self, mock_openai_response):
        """Test that analyze_gaps only uses top 10 internal results."""
        query = "AI safety"

        # Create 15 internal results
        many_results = [
            SearchResult(
                source_id=f"doc{i}",
                content=f"Content about AI topic {i}",
                title=f"Document {i}",
                url=None,
                score=0.9 - i * 0.01,
                source_type="internal",
            )
            for i in range(15)
        ]

        with patch("src.generation.gap_analysis.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
            mock_openai.return_value = mock_client

            await analyze_gaps(query, many_results)

            # Check that only first 10 were included
            call_args = mock_client.chat.completions.create.call_args
            user_content = call_args.kwargs["messages"][1]["content"]

            # First 10 should be in prompt
            assert "Document 0" in user_content
            assert "Document 9" in user_content

            # 11th and beyond should NOT be in prompt
            assert "Document 10" not in user_content
            assert "Document 14" not in user_content


class TestAnalyzeGapsGapTypes:
    """Test that gap analysis identifies different types of gaps."""

    @pytest.mark.asyncio
    async def test_identifies_statistics_gaps(self, mock_internal_results):
        """Test that gap analysis can identify missing statistics."""
        query = "AI safety research impact"

        # Mock response with statistics gap
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        {
                            "gaps": [
                                {
                                    "type": "statistics",
                                    "query": "AI safety funding statistics 2024",
                                }
                            ]
                        }
                    )
                )
            )
        ]

        with patch("src.generation.gap_analysis.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            gaps = await analyze_gaps(query, mock_internal_results)

            assert len(gaps) > 0
            # Should include statistics-related query
            assert any("statistic" in gap.lower() or "funding" in gap.lower() for gap in gaps)

    @pytest.mark.asyncio
    async def test_identifies_expert_opinion_gaps(self, mock_internal_results):
        """Test that gap analysis can identify missing expert opinions."""
        query = "Future of AI alignment"

        # Mock response with expert opinion gap
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        {
                            "gaps": [
                                {
                                    "type": "expert_opinion",
                                    "query": "AI alignment experts opinions 2024",
                                }
                            ]
                        }
                    )
                )
            )
        ]

        with patch("src.generation.gap_analysis.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            gaps = await analyze_gaps(query, mock_internal_results)

            assert len(gaps) > 0
            assert any("expert" in gap.lower() or "opinion" in gap.lower() for gap in gaps)

    @pytest.mark.asyncio
    async def test_identifies_visual_evidence_gaps(self, mock_internal_results):
        """Test that gap analysis can identify missing visual evidence."""
        query = "AI model performance trends"

        # Mock response with visual gap
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        {
                            "gaps": [
                                {
                                    "type": "visual",
                                    "query": "AI performance chart OR graph 2024",
                                }
                            ]
                        }
                    )
                )
            )
        ]

        with patch("src.generation.gap_analysis.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            gaps = await analyze_gaps(query, mock_internal_results)

            assert len(gaps) > 0
            assert any(
                any(term in gap.lower() for term in ["chart", "graph", "visual", "infographic"])
                for gap in gaps
            )


class TestAnalyzeGapsLimits:
    """Test gap analysis limits and constraints."""

    @pytest.mark.asyncio
    async def test_limits_to_five_queries(self, mock_internal_results):
        """Test that analyze_gaps returns max 5 queries."""
        query = "AI safety comprehensive overview"

        # Mock response with 10 gaps
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        {"gaps": [{"type": "statistics", "query": f"Query {i}"} for i in range(10)]}
                    )
                )
            )
        ]

        with patch("src.generation.gap_analysis.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            gaps = await analyze_gaps(query, mock_internal_results)

            # Should be limited to 5
            assert len(gaps) <= 5

    @pytest.mark.asyncio
    async def test_filters_empty_queries(self, mock_internal_results):
        """Test that empty query strings are filtered out."""
        query = "AI safety"

        # Mock response with some empty queries
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        {
                            "gaps": [
                                {"type": "statistics", "query": "Valid query 1"},
                                {"type": "expert_opinion", "query": ""},  # Empty
                                {"type": "visual", "query": "Valid query 2"},
                                {"type": "context", "query": ""},  # Empty
                            ]
                        }
                    )
                )
            )
        ]

        with patch("src.generation.gap_analysis.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            gaps = await analyze_gaps(query, mock_internal_results)

            # Should only include non-empty queries
            assert len(gaps) == 2
            assert all(len(q) > 0 for q in gaps)


class TestAnalyzeGapsErrorHandling:
    """Test error handling in gap analysis."""

    @pytest.mark.asyncio
    async def test_api_error_returns_original_query(self, mock_internal_results):
        """Test that API errors fall back to original query."""
        query = "AI safety research"

        with patch("src.generation.gap_analysis.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
            mock_openai.return_value = mock_client

            gaps = await analyze_gaps(query, mock_internal_results)

            # Should fall back to original query
            assert len(gaps) == 1
            assert gaps[0] == query

    @pytest.mark.asyncio
    async def test_json_parse_error_returns_original_query(self, mock_internal_results):
        """Test that JSON parsing errors fall back to original query."""
        query = "AI safety research"

        # Mock response with invalid JSON
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Invalid JSON {{{"))]

        with patch("src.generation.gap_analysis.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            gaps = await analyze_gaps(query, mock_internal_results)

            # Should fall back to original query
            assert len(gaps) == 1
            assert gaps[0] == query

    @pytest.mark.asyncio
    async def test_no_gaps_returns_original_query(self, mock_internal_results):
        """Test that when no gaps identified, returns original query."""
        query = "AI safety research"

        # Mock response with empty gaps
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=json.dumps({"gaps": []})))]

        with patch("src.generation.gap_analysis.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            gaps = await analyze_gaps(query, mock_internal_results)

            # Should fall back to original query
            assert len(gaps) == 1
            assert gaps[0] == query

    @pytest.mark.asyncio
    async def test_malformed_response_returns_original_query(self, mock_internal_results):
        """Test that malformed API responses fall back to original query."""
        query = "AI safety"

        # Mock response with missing 'gaps' key
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=json.dumps({"results": []})))]

        with patch("src.generation.gap_analysis.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            gaps = await analyze_gaps(query, mock_internal_results)

            # Should fall back to original query
            assert len(gaps) == 1
            assert gaps[0] == query


class TestAnalyzeGapsQueryQuality:
    """Test quality of generated gap queries."""

    @pytest.mark.asyncio
    async def test_queries_are_specific(self, mock_internal_results):
        """Test that generated queries are specific and actionable."""
        query = "AI alignment"

        # Mock response with specific queries
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        {
                            "gaps": [
                                {
                                    "type": "statistics",
                                    "query": "AI alignment research papers published 2024",
                                },
                                {
                                    "type": "expert_opinion",
                                    "query": "Paul Christiano AI alignment predictions 2024",
                                },
                            ]
                        }
                    )
                )
            )
        ]

        with patch("src.generation.gap_analysis.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            gaps = await analyze_gaps(query, mock_internal_results)

            # Queries should be more specific than original
            for gap in gaps:
                assert len(gap.split()) >= 3  # At least 3 words
                assert gap.lower() != query.lower()  # Different from original

    @pytest.mark.asyncio
    async def test_queries_are_diverse(self, mock_internal_results):
        """Test that generated queries cover different gap types."""
        query = "AI safety progress"

        # Mock response with diverse gap types
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        {
                            "gaps": [
                                {
                                    "type": "statistics",
                                    "query": "AI safety funding 2024 numbers",
                                },
                                {
                                    "type": "expert_opinion",
                                    "query": "AI safety expert quotes 2024",
                                },
                                {"type": "visual", "query": "AI safety progress chart 2024"},
                            ]
                        }
                    )
                )
            )
        ]

        with patch("src.generation.gap_analysis.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai.return_value = mock_client

            gaps = await analyze_gaps(query, mock_internal_results)

            # Should have multiple different queries
            assert len(set(gaps)) == len(gaps)  # All unique
