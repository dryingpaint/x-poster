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
from src.core.models import SearchResult
from src.retrieval.content_filter import filter_and_download


async def search_exa(
    query: str, num_results: int = 10, num_images: int = 5
) -> list[dict[str, Any]]:
    """Search using EXA API - images will be extracted from HTML."""
    config = get_config()

    if not config.exa_api_key:
        return []

    exa = Exa(api_key=config.exa_api_key)

    try:
        # Request text content from EXA
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
                    "has_relevance_score": has_real_score,
                }
            )

        return search_results

    except Exception as e:
        print(f"EXA search failed: {e}")
        return []


async def fetch_url_content(url: str, timeout: int = 6) -> tuple[str | None, list[str]]:
    """Fetch and extract clean text + image URLs from a URL.

    Returns:
        Tuple of (clean_text, image_urls)
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, headers=headers, follow_redirects=True)
            response.raise_for_status()

            # Extract clean text using trafilatura
            content = extract(response.text, include_comments=False, include_tables=True)

            # Extract image URLs from HTML with intelligent filtering
            from urllib.parse import urljoin

            from bs4 import BeautifulSoup

            soup = BeautifulSoup(response.text, "html.parser")
            image_candidates = []

            for img in soup.find_all("img", limit=20):
                src = img.get("src") or img.get("data-src")
                if not src:
                    continue

                absolute_url = urljoin(url, src)

                # Skip obvious non-content images by URL pattern
                skip_patterns = [
                    "/logo",
                    "/icon",
                    "avatar",
                    "emoji",
                    "badge",
                    "button",
                    "social-",
                    "facebook",
                    "twitter",
                    "linkedin",
                    "/share",
                    "1x1.",
                    "pixel.",
                    "tracking",
                    "/banner",
                    "/ad-",
                    "/ads/",
                ]
                if any(skip in absolute_url.lower() for skip in skip_patterns):
                    continue

                # Get dimensions from HTML attributes
                width = img.get("width")
                height = img.get("height")
                try:
                    w = int(width) if width else 0
                    h = int(height) if height else 0
                except (ValueError, TypeError):
                    w = h = 0

                # Check alt text and classes for chart/graph indicators
                alt_text = (img.get("alt") or "").lower()
                class_names = " ".join(img.get("class", [])).lower()

                is_likely_chart = any(
                    keyword in alt_text or keyword in class_names
                    for keyword in [
                        "chart",
                        "graph",
                        "diagram",
                        "infographic",
                        "visualization",
                        "data",
                        "plot",
                        "figure",
                    ]
                )

                # Score images (higher = better)
                score = 0
                if is_likely_chart:
                    score += 100  # Strong signal of data visualization
                if w > 600 and h > 400:  # Large images likely to be charts
                    score += 50
                elif w > 300 and h > 200:  # Medium images
                    score += 20
                elif w > 0 and (w < 100 or h < 100):  # Too small = icon/thumbnail
                    score -= 50

                if score > 0 or (w == 0 and h == 0):  # Include if good score or unknown size
                    image_candidates.append((score, absolute_url))

            # Sort by score and take top candidates only
            image_candidates.sort(reverse=True, key=lambda x: x[0])
            image_urls = [url for score, url in image_candidates[:5] if score > 0]

            return content, image_urls

    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return None, []


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
    Fetch full content + images for web search results.

    For EXA results, content is often already included, but we still fetch
    the HTML to extract image URLs.
    """
    config = get_config()

    tasks = []
    results_to_fetch = []

    for result in search_results:
        if result.url:
            results_to_fetch.append(result)
            # Always fetch to get images, even if we have content from EXA
            tasks.append(fetch_url_content(result.url, timeout=config.web_fetch_timeout))

    # Fetch in parallel
    if tasks:
        fetch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result, fetch_result in zip(results_to_fetch, fetch_results, strict=True):
            if isinstance(fetch_result, tuple):
                content, image_urls = fetch_result
                # Only replace content if we don't have it yet
                if content and (not result.content or len(result.content) < 200):
                    result.content = content
                # Always add image URLs if found
                if image_urls:
                    result.meta["image_urls"] = image_urls

    # Filter out results with no content
    return [r for r in search_results if r.content and len(r.content) > 50]


async def search_and_filter(
    query: str,
    top_k: int = 50,
    user_context: str | None = None,
) -> tuple[list[SearchResult], list[SearchResult]]:
    """
    Search web and filter results through LLM.

    Args:
        query: Search query
        top_k: Number of results to retrieve
        user_context: Optional context for filtering

    Returns:
        Tuple of (filtered_results, raw_results)
        - filtered_results: SearchResults with populated structured fields
        - raw_results: Original SearchResult objects (unfiltered)
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
