"""Web search and content extraction."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import dateparser
import httpx
from exa_py import Exa
from trafilatura import extract

from src.core.config import get_config
from src.core.models import FilteredWebResult, SearchResult
from src.retrieval.content_filter import filter_and_download


async def search_exa(query: str, num_results: int = 10) -> list[dict[str, Any]]:
    """Search using EXA API."""
    config = get_config()

    if not config.exa_api_key:
        return []

    exa = Exa(api_key=config.exa_api_key)

    try:
        results = exa.search_and_contents(
            query,
            num_results=num_results,
            text=True,
            highlights=True,
        )

        search_results = []
        for result in results.results:
            # EXA provides real relevance scores
            has_real_score = hasattr(result, "score") and result.score is not None
            search_results.append(
                {
                    "url": result.url,
                    "title": result.title,
                    "content": result.text or "",
                    "published_at": result.published_date,
                    "author": result.author,
                    "score": result.score if has_real_score else None,
                    "has_relevance_score": has_real_score,  # Flag for real scores
                }
            )

        return search_results

    except Exception as e:
        print(f"EXA search failed: {e}")
        return []


async def fetch_url_content(url: str, timeout: int = 6) -> str | None:
    """Fetch and extract clean text from a URL."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()

            # Extract clean text using trafilatura
            content = extract(response.text, include_comments=False, include_tables=True)
            return content

    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return None


async def search_web(query: str, top_k: int = 50) -> list[SearchResult]:
    """
    Search the web using primary provider (EXA) with fallback.

    Returns list of SearchResult objects.
    """
    # Try primary provider
    results = await search_exa(query, num_results=top_k)

    # Convert to SearchResult objects
    search_results = []
    for i, result in enumerate(results):
        domain = urlparse(result["url"]).netloc

        # For EXA, content might already be included
        content = result.get("content") or result.get("snippet", "")

        # Normalize published_at to datetime | None
        raw_published = result.get("published_at")
        published_at: datetime | None = None
        if raw_published:
            try:
                if isinstance(raw_published, datetime):
                    published_at = raw_published
                else:
                    parsed = dateparser.parse(str(raw_published))
                    published_at = parsed if isinstance(parsed, datetime) else None
            except Exception:
                published_at = None

        # Ensure published_at is None if empty string
        if published_at == "":
            published_at = None

        search_results.append(
            SearchResult(
                source_id=f"web_{i}_{hash(result['url'])}",
                content=content,
                title=result.get("title"),
                url=result["url"],
                author=result.get("author"),
                published_at=published_at,
                meta={
                    "domain": domain,
                },
                source_type="web",
            )
        )

    return search_results


async def fetch_and_extract(search_results: list[SearchResult]) -> list[SearchResult]:
    """
    Fetch full content for web search results that don't have it yet.

    For EXA results, content is often already included.
    For Serper results, we need to fetch and extract.
    """
    config = get_config()

    tasks = []
    results_to_fetch = []

    for result in search_results:
        # Only fetch if content is too short (likely just a snippet)
        if result.content and len(result.content) > 200:
            continue

        if result.url:
            results_to_fetch.append(result)
            tasks.append(fetch_url_content(result.url, timeout=config.web_fetch_timeout))

    # Fetch in parallel
    if tasks:
        contents = await asyncio.gather(*tasks, return_exceptions=True)

        for result, content in zip(results_to_fetch, contents, strict=True):
            if isinstance(content, str) and content:
                result.content = content

    # Filter out results with no content
    return [r for r in search_results if r.content and len(r.content) > 50]


async def search_and_filter(
    query: str,
    top_k: int = 50,
    user_context: str | None = None,
) -> tuple[list[FilteredWebResult], list[SearchResult]]:
    """
    Search web and filter results through LLM.

    Args:
        query: Search query
        top_k: Number of results to retrieve
        user_context: Optional context for filtering

    Returns:
        Tuple of (filtered_results, raw_results)
        - filtered_results: LLM-filtered results with extracted content
        - raw_results: Original SearchResult objects (for backward compatibility)
    """
    config = get_config()

    # Step 1: Get raw search results
    search_results = await search_web(query, top_k)

    # Step 2: Fetch full content
    search_results = await fetch_and_extract(search_results)

    # Step 3: Apply LLM filtering
    filtered_results = await filter_and_download(
        search_results,
        query,
        output_dir=Path(config.media_output_dir),
        user_context=user_context,
        max_concurrent=config.max_filter_concurrent,
    )

    return filtered_results, search_results
