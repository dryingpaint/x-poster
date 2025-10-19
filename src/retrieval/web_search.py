"""Web search and content extraction."""

import asyncio
from typing import Any
from datetime import datetime
from urllib.parse import urlparse

import httpx
from exa_py import Exa
from serpapi import GoogleSearch
from trafilatura import extract
import dateparser

from src.core.config import get_config
from src.core.models import SearchResult


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
            search_results.append(
                {
                    "url": result.url,
                    "title": result.title,
                    "content": result.text or "",
                    "published_at": result.published_date,
                    "author": result.author,
                    "score": result.score if hasattr(result, "score") else 1.0,
                }
            )

        return search_results

    except Exception as e:
        print(f"EXA search failed: {e}")
        return []


async def search_serper(query: str, num_results: int = 10) -> list[dict[str, Any]]:
    """Search using Serper (Google Search API)."""
    config = get_config()

    if not config.serper_api_key:
        return []

    try:
        search = GoogleSearch(
            {
                "q": query,
                "num": num_results,
                "api_key": config.serper_api_key,
            }
        )

        results = search.get_dict()

        search_results = []
        for item in results.get("organic_results", [])[:num_results]:
            search_results.append(
                {
                    "url": item.get("link"),
                    "title": item.get("title"),
                    "snippet": item.get("snippet", ""),
                    "published_at": None,
                    "author": None,
                    "score": 1.0 / (item.get("position", 1)),  # Higher rank = higher score
                }
            )

        return search_results

    except Exception as e:
        print(f"Serper search failed: {e}")
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
    Search the web using primary provider (EXA or Serper) with fallback.

    Returns list of SearchResult objects.
    """
    config = get_config()

    # Try primary provider
    if config.primary_search_provider == "exa":
        results = await search_exa(query, num_results=top_k)
        if not results:  # Fallback to Serper
            results = await search_serper(query, num_results=top_k)
    else:
        results = await search_serper(query, num_results=top_k)
        if not results:  # Fallback to EXA
            results = await search_exa(query, num_results=top_k)

    # Convert to SearchResult objects
    search_results = []
    for i, result in enumerate(results):
        domain = urlparse(result["url"]).netloc

        # For EXA, content might already be included
        content = result.get("content") or result.get("snippet", "")

        # Ensure score is a valid float
        score = result.get("score")
        if score is None or not isinstance(score, (int, float)):
            score = 0.0
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
        
        search_results.append(
            SearchResult(
                source_id=f"web_{i}_{hash(result['url'])}",
                content=content,
                title=result.get("title"),
                url=result["url"],
                author=result.get("author"),
                published_at=published_at,
                meta={"domain": domain},
                score=float(score),
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

