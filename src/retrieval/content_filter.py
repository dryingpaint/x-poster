"""LLM-based content filtering and extraction for web search results."""

import asyncio
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI

from src.core.config import get_config
from src.core.models import FilteredWebResult, MediaFile, SearchResult


async def extract_relevant_content(
    search_result: SearchResult,
    query: str,
    user_context: str | None = None,
) -> FilteredWebResult:
    """
    Use LLM to extract relevant information from web search result.

    Args:
        search_result: Raw search result with full content
        query: Original search query for relevance filtering
        user_context: Optional context about what the user is looking for

    Returns:
        FilteredWebResult with extracted text and media references
    """
    config = get_config()
    client = AsyncOpenAI(api_key=config.openai_api_key)

    # Build filtering prompt
    context_note = f"\n\nUser context: {user_context}" if user_context else ""

    prompt = f"""You are analyzing a web page to extract information relevant to this query:
Query: {query}{context_note}

Page Title: {search_result.title or 'Unknown'}
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
        media_files = []
        for i, media_desc in enumerate(extracted.get("media_descriptions", [])):
            media_files.append(
                MediaFile(
                    media_id=f"{search_result.source_id}_media_{i}",
                    media_type=media_desc.get("type", "unknown"),
                    description=media_desc.get("description", ""),
                    context=media_desc.get("context", ""),
                    source_url=search_result.url,
                    local_path=None,  # Will be populated during download
                )
            )

        return FilteredWebResult(
            source_id=search_result.source_id,
            original_url=search_result.url,
            title=search_result.title,
            author=search_result.author,
            published_at=search_result.published_at,
            relevant_text=extracted.get("relevant_text", ""),
            key_points=extracted.get("key_points", []),
            media_files=media_files,
            credibility_score=float(extracted.get("credibility_score", 0.5)),
            relevance_score=float(extracted.get("relevance_score", 0.5)),
            meta=search_result.meta,
            extracted_at=datetime.utcnow(),
        )

    except Exception as e:
        print(f"Content extraction failed for {search_result.url}: {e}")
        # Fallback: return original content with minimal processing
        return FilteredWebResult(
            source_id=search_result.source_id,
            original_url=search_result.url,
            title=search_result.title,
            author=search_result.author,
            published_at=search_result.published_at,
            relevant_text=search_result.content[:2000],  # Truncate
            key_points=[],
            media_files=[],
            credibility_score=0.5,
            relevance_score=0.5,
            meta=search_result.meta,
            extracted_at=datetime.utcnow(),
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
) -> list[FilteredWebResult]:
    """
    Filter multiple search results through LLM and download media files.

    Args:
        search_results: List of raw search results
        query: Original search query
        output_dir: Directory for downloaded media files
        user_context: Optional user context for filtering
        max_concurrent: Max concurrent operations

    Returns:
        List of filtered results with downloaded media
    """
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    # Extract relevant content from all results
    semaphore = asyncio.Semaphore(max_concurrent)

    async def extract_with_limit(result: SearchResult) -> FilteredWebResult:
        async with semaphore:
            return await extract_relevant_content(result, query, user_context)

    filtered_results = await asyncio.gather(
        *[extract_with_limit(r) for r in search_results],
        return_exceptions=True,
    )

    # Filter out exceptions
    valid_results = [r for r in filtered_results if isinstance(r, FilteredWebResult)]

    # Download all media files
    download_tasks = []
    for result in valid_results:
        for media in result.media_files:
            download_tasks.append(download_media_file(media, output_dir))

    if download_tasks:
        await asyncio.gather(*download_tasks, return_exceptions=True)

    return valid_results


def format_for_agent_history(filtered_result: FilteredWebResult) -> str:
    """
    Format filtered result as text for agent message history.

    Args:
        filtered_result: Filtered web result

    Returns:
        Formatted string for message history
    """
    lines = []

    # Header
    lines.append(f"## {filtered_result.title or 'Web Source'}")
    lines.append(f"**URL**: {filtered_result.original_url}")

    if filtered_result.author:
        lines.append(f"**Author**: {filtered_result.author}")

    if filtered_result.published_at:
        lines.append(f"**Published**: {filtered_result.published_at.strftime('%Y-%m-%d')}")

    lines.append(f"**Relevance**: {filtered_result.relevance_score:.2f}")
    lines.append(f"**Credibility**: {filtered_result.credibility_score:.2f}")
    lines.append("")

    # Relevant text
    lines.append("### Extracted Content")
    lines.append(filtered_result.relevant_text)
    lines.append("")

    # Key points
    if filtered_result.key_points:
        lines.append("### Key Points")
        for point in filtered_result.key_points:
            lines.append(f"- {point}")
        lines.append("")

    # Media files
    if filtered_result.media_files:
        lines.append("### Media Files")
        for media in filtered_result.media_files:
            if media.local_path:
                lines.append(f"- **{media.media_type.capitalize()}**: {media.description}")
                lines.append(f"  - File: `{media.local_path}`")
                if media.context:
                    lines.append(f"  - Context: {media.context}")
            else:
                lines.append(f"- **{media.media_type.capitalize()}** (not downloaded): {media.description}")
        lines.append("")

    lines.append("---")
    lines.append("")

    return "\n".join(lines)
