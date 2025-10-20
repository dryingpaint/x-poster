"""LLM-based content filtering and extraction for web search results."""

import asyncio
import hashlib
import json
from datetime import datetime
from pathlib import Path

import httpx
from openai import AsyncOpenAI

from src.core.config import get_config
from src.core.models import MediaFile, SearchResult


async def extract_relevant_content(
    search_result: SearchResult,
    query: str,
    user_context: str | None = None,
) -> SearchResult:
    """
    Use LLM to extract relevant information from web search result.

    Args:
        search_result: Raw search result with full content
        query: Original search query for relevance filtering
        user_context: Optional context about what the user is looking for

    Returns:
        SearchResult with populated structured fields (key_points, media_files, scores)
    """
    config = get_config()
    client = AsyncOpenAI(api_key=config.openai_api_key)

    # Build filtering prompt
    context_note = f"\n\nUser context: {user_context}" if user_context else ""

    prompt = f"""You are analyzing a web page to extract information relevant to this query:
Query: {query}{context_note}

Page Title: {search_result.title or "Unknown"}
Page URL: {search_result.url}
Page Content:
{search_result.content[:8000]}

Extract ALL relevant information that helps answer the query. Include:
1. Key facts, claims, and data points
2. Quotes from credible sources
3. Statistical evidence
4. Historical context or background
5. References to images, charts, or media (note what they show)

Format your response as JSON:
{{
    "relevant_text": "The extracted relevant information as flowing prose...",
    "key_points": ["Point 1", "Point 2", ...],
    "media_descriptions": [
        {{"type": "image", "description": "What the image shows", "context": "Why it's relevant"}},
        ...
    ],
    "credibility_score": 0.0-1.0,
    "relevance_score": 0.0-1.0
}}

Be comprehensive but focused. Extract everything useful for answering the query."""

    try:
        response = await client.chat.completions.create(
            model=config.filter_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.filter_temperature,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from LLM")

        extracted = json.loads(content)

        # Create MediaFile objects for media descriptions
        # Try to match with actual image URLs from EXA if available
        image_urls = search_result.meta.get("image_urls", [])
        media_files = []

        for i, media_desc in enumerate(extracted.get("media_descriptions", [])):
            # Use actual image URL if available (from EXA), otherwise None
            source_url = image_urls[i] if i < len(image_urls) else None

            media_files.append(
                MediaFile(
                    media_id=f"{search_result.source_id}_media_{i}",
                    media_type=media_desc.get("type", "unknown"),
                    description=media_desc.get("description", ""),
                    context=media_desc.get("context", ""),
                    source_url=source_url,  # Real image URL from EXA!
                    local_path=None,  # Will be populated during download
                )
            )

        # Return SearchResult with populated structured fields
        return SearchResult(
            source_id=search_result.source_id,
            content=extracted.get("relevant_text", ""),  # LLM-extracted relevant text
            title=search_result.title,
            url=search_result.url,
            author=search_result.author,
            published_at=search_result.published_at,
            meta={**search_result.meta, "extracted_at": datetime.utcnow().isoformat()},
            score=float(extracted.get("relevance_score", 0.5)),
            source_type="web",
            key_points=extracted.get("key_points", []),
            media_files=media_files,
            credibility_score=float(extracted.get("credibility_score", 0.5)),
            relevance_score=float(extracted.get("relevance_score", 0.5)),
        )

    except Exception as e:
        print(f"Content extraction failed for {search_result.url}: {e}")
        # Fallback: return original with minimal changes (just truncate content)
        return SearchResult(
            source_id=search_result.source_id,
            content=search_result.content[:2000],  # Truncate
            title=search_result.title,
            url=search_result.url,
            author=search_result.author,
            published_at=search_result.published_at,
            meta={**search_result.meta, "filter_failed": True},
            score=0.5,
            source_type="web",
            credibility_score=0.5,
            relevance_score=0.5,
        )


async def download_media_file(
    media: MediaFile,
    output_dir: Path,
    timeout: int = 10,
) -> MediaFile:
    """
    Download a media file from source URL and save locally.

    Args:
        media: MediaFile object with source URL
        output_dir: Directory to save downloaded files
        timeout: Download timeout in seconds

    Returns:
        Updated MediaFile with local_path populated
    """
    if not media.source_url:
        return media

    try:
        # Create output directory if needed
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate safe filename from URL hash
        url_hash = hashlib.sha256(media.source_url.encode()).hexdigest()[:16]
        extension = _guess_extension(media.media_type, media.source_url)
        filename = f"{media.media_id}_{url_hash}{extension}"
        filepath = output_dir / filename

        # Download file
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(media.source_url, follow_redirects=True)
            response.raise_for_status()

            # Save to disk
            filepath.write_bytes(response.content)

            # Update MediaFile with local path
            media.local_path = str(filepath.absolute())

        return media

    except Exception as e:
        print(f"Failed to download media {media.media_id} from {media.source_url}: {e}")
        return media


def _guess_extension(media_type: str, url: str) -> str:
    """Guess file extension from media type and URL."""
    # Try URL first
    if "." in url:
        potential_ext = Path(url.split("?")[0]).suffix
        if potential_ext:
            return potential_ext

    # Fallback to media type
    type_map = {
        "image": ".jpg",
        "video": ".mp4",
        "audio": ".mp3",
        "pdf": ".pdf",
    }
    return type_map.get(media_type, ".bin")


async def filter_and_download(
    search_results: list[SearchResult],
    query: str,
    output_dir: Path | str = "data/media",
    user_context: str | None = None,
    max_concurrent: int = 5,
) -> list[SearchResult]:
    """
    Filter multiple search results through LLM and download media files.

    Args:
        search_results: List of raw search results
        query: Original search query
        output_dir: Directory for downloaded media files
        user_context: Optional user context for filtering
        max_concurrent: Max concurrent operations

    Returns:
        List of SearchResults with populated structured fields and downloaded media
    """
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    # Extract relevant content from all results
    semaphore = asyncio.Semaphore(max_concurrent)

    async def extract_with_limit(result: SearchResult) -> SearchResult:
        async with semaphore:
            return await extract_relevant_content(result, query, user_context)

    filtered_results = await asyncio.gather(
        *[extract_with_limit(r) for r in search_results],
        return_exceptions=True,
    )

    # Filter out exceptions
    valid_results = [r for r in filtered_results if isinstance(r, SearchResult)]

    # Download all media files (only if they have source URLs)
    download_tasks = []
    for result in valid_results:
        for media in result.media_files:
            if media.source_url:  # Only download if we have an actual media URL
                download_tasks.append(download_media_file(media, output_dir))

    if download_tasks:
        await asyncio.gather(*download_tasks, return_exceptions=True)

    return valid_results


def format_for_agent_history(result: SearchResult) -> str:
    """
    Format search result as text for agent message history.

    Args:
        result: Search result (potentially with LLM-filtered fields)

    Returns:
        Formatted string for message history
    """
    # Use the built-in to_context_text method
    return result.to_context_text(include_metadata=True)
