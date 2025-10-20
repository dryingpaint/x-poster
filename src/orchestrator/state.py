"""LangGraph state schema for the tweet generation pipeline."""

from typing import Optional, TypedDict

from src.core.models import (
    EvidencePack,
    GenerateResponse,
    SearchResult,
    Tweet,
)


class AgentState(TypedDict):
    """State that flows through the LangGraph pipeline.

    This represents all data passed between nodes in the state graph.
    Each step adds or modifies fields as the pipeline progresses.
    """

    # Input parameters
    query: str
    max_variants: int
    max_thread_tweets: int

    # Step 1: Embed query
    query_embedding: Optional[list[float]]

    # Step 2: Internal retrieval
    internal_results: Optional[list[SearchResult]]

    # Step 3: Gap analysis
    gap_queries: Optional[list[str]]

    # Step 4: Web search
    web_results: Optional[list[SearchResult]]

    # Step 5: Embedding (handled implicitly in merge step)

    # Step 6: Merge & dedupe
    merged_results: Optional[list[SearchResult]]

    # Step 7: Rerank
    final_results: Optional[list[SearchResult]]

    # Step 8: Evidence pack
    evidence: Optional[EvidencePack]

    # Step 9: Tweet generation
    variants: Optional[list[Tweet]]
    thread: Optional[list[Tweet]]

    # Step 10: Fact checking (modifies variants and thread)

    # Step 11: Final response
    response: Optional[GenerateResponse]

    # Error handling
    error: Optional[str]
