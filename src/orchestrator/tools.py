"""LangGraph node functions (tools) for the tweet generation pipeline.

Each function represents a node in the state graph and operates on AgentState.
"""

import asyncio
from pathlib import Path
from typing import Any

from src.core.config import get_config
from src.core.models import Source
from src.db.operations import search_internal
from src.generation.embeddings import embed_batch, embed_text
from src.generation.gap_analysis import analyze_gaps
from src.generation.writer import generate_tweets_from_results
from src.orchestrator.state import AgentState
from src.retrieval.content_filter import filter_and_download
from src.retrieval.merger import merge_and_dedupe_results
from src.retrieval.reranker import rerank_results
from src.retrieval.web_search import fetch_and_extract, search_web


# Node 1: Embed query
async def embed_query_node(state: AgentState) -> dict[str, Any]:
    """Step 1: Embed the user query using BGE-M3."""
    query = state["query"]
    query_embedding = embed_text(query)
    return {"query_embedding": query_embedding}


# Node 2: Internal search
async def internal_search_node(state: AgentState) -> dict[str, Any]:
    """Step 2: Search internal documents using hybrid search."""
    config = get_config()
    query = state["query"]
    query_embedding = state["query_embedding"]

    internal_results = await search_internal(
        query=query, query_embedding=query_embedding, top_k=config.internal_top_k
    )

    return {"internal_results": internal_results}


# Node 3: Gap analysis
async def gap_analysis_node(state: AgentState) -> dict[str, Any]:
    """Step 3: Analyze knowledge gaps and generate targeted search queries."""
    query = state["query"]
    internal_results = state["internal_results"] or []

    gap_queries = await analyze_gaps(query, internal_results)

    return {"gap_queries": gap_queries}


# Node 4: Web search (parallel)
async def web_search_node(state: AgentState) -> dict[str, Any]:
    """Step 4: Execute parallel web searches for each gap query."""
    config = get_config()
    gap_queries = state["gap_queries"] or []
    query = state["query"]  # Original query for context

    if not gap_queries:
        return {"web_results": []}

    # Parallel web searches for each gap query
    search_tasks = [search_web(q, config.web_top_k) for q in gap_queries]
    search_results_list = await asyncio.gather(*search_tasks)

    # Flatten results from all gap queries
    web_results = []
    for results in search_results_list:
        web_results.extend(results)

    # Fetch full content for results with short snippets
    web_results = await fetch_and_extract(web_results)

    # Apply LLM content filtering if enabled
    if config.enable_content_filtering:
        user_context = f"Looking for evidence to write tweets about: {query}"

        filtered_results = await filter_and_download(
            web_results,
            query=query,
            output_dir=Path(config.media_output_dir),
            user_context=user_context,
            max_concurrent=config.max_filter_concurrent,
        )

        # Use the filtered results directly (already SearchResult objects)
        web_results = filtered_results

    return {"web_results": web_results}


# Node 5: Merge & dedupe
async def merge_dedupe_node(state: AgentState) -> dict[str, Any]:
    """Step 6: Merge internal and web results, dedupe, and apply diversity constraints."""
    config = get_config()
    internal_results = state["internal_results"] or []
    web_results = state["web_results"] or []

    # Embed all results for deduplication
    internal_texts = [r.content for r in internal_results]
    web_texts = [r.content for r in web_results]

    internal_embeddings = embed_batch(internal_texts) if internal_texts else []
    web_embeddings = embed_batch(web_texts) if web_texts else []

    # Merge, dedupe, and apply diversity constraints
    merged_results = merge_and_dedupe_results(
        internal_results=internal_results,
        web_results=web_results,
        internal_embeddings=internal_embeddings,
        web_embeddings=web_embeddings,
        final_k=config.rerank_k,
    )

    return {"merged_results": merged_results}


# Node 6: Rerank
async def rerank_node(state: AgentState) -> dict[str, Any]:
    """Step 7: Rerank merged results using cross-encoder."""
    config = get_config()
    query = state["query"]
    merged_results = state["merged_results"] or []

    final_results = await rerank_results(query=query, results=merged_results, top_k=config.final_top_k)

    return {"final_results": final_results}


# Node 7: Tweet generation directly from results
async def tweet_generation_node(state: AgentState) -> dict[str, Any]:
    """Step 8: Generate tweets directly from final results using LLM (verbatim quotes only)."""
    query = state["query"]
    final_results = state["final_results"] or []
    max_variants = state["max_variants"]
    max_thread_tweets = state["max_thread_tweets"]

    variants, thread = await generate_tweets_from_results(
        query=query,
        results=final_results,
        max_variants=max_variants,
        max_thread_tweets=max_thread_tweets,
    )

    return {"variants": variants, "thread": thread}


# Node 10: Prepare response
async def prepare_response_node(state: AgentState) -> dict[str, Any]:
    """Step 11: Assemble final response with sources."""
    from src.core.models import GenerateResponse

    variants = state["variants"] or []
    thread = state["thread"] or []

    # Build source list from final_results (match Source model fields)
    sources = []
    for r in state.get("final_results", []) or []:
        sources.append(
            Source(
                source_id=r.source_id,
                title=r.title,
                url=r.url,
                source_uri=r.source_uri,
                author=r.author,
                published_at=r.published_at,
                meta=r.meta or {},
            )
        )

    response = GenerateResponse(variants=variants, thread=thread, sources=sources)

    return {"response": response}
