# Content Filtering System

## Overview

The content filtering system uses LLM to intelligently extract relevant information from web search results. Instead of using raw web page content, it:

1. **Filters text**: Extracts only information relevant to the query
2. **Identifies media**: Detects and describes images, charts, videos
3. **Downloads files**: Saves media to local storage
4. **Structures output**: Formats for agent message history

## Quick Start

### Basic Usage

```python
from src.retrieval.web_search import search_and_filter

# Search and filter in one call
filtered_results, raw_results = await search_and_filter(
    query="What does Jason Hickel say about degrowth?",
    top_k=10,
    user_context="Need quotes and data for a tweet"
)

# Format for agent history
from src.retrieval.content_filter import format_for_agent_history

for result in filtered_results:
    message = format_for_agent_history(result)
    # Add to agent message history
    print(message)
```

### Configuration

Add to `.env`:

```bash
# Content filtering (optional, has defaults)
ENABLE_CONTENT_FILTERING=true
FILTER_MODEL=gpt-4o-mini
FILTER_TEMPERATURE=0.2
MAX_FILTER_CONCURRENT=5
MEDIA_OUTPUT_DIR=data/media
MEDIA_DOWNLOAD_TIMEOUT=10
```

## Architecture

### Data Flow

```
1. Web Search (EXA/Serper)
   ↓
2. Fetch Full Content (trafilatura)
   ↓
3. LLM Filtering (extract_relevant_content)
   - Extract relevant text
   - Identify key points
   - Detect media references
   ↓
4. Media Download (download_media_file)
   - Download images/videos/charts
   - Save to local directory
   ↓
5. Format for Agent (format_for_agent_history)
   - Text goes to message history
   - File paths provided for media
```

### Models

**FilteredWebResult**: LLM-filtered search result
- `relevant_text`: Extracted relevant content (prose)
- `key_points`: List of key facts/claims
- `media_files`: List of MediaFile objects
- `credibility_score`: 0.0-1.0 (LLM assessment)
- `relevance_score`: 0.0-1.0 (LLM assessment)

**MediaFile**: Reference to downloaded media
- `media_type`: 'image', 'video', 'audio', etc.
- `description`: What the media shows
- `context`: Why it's relevant
- `source_url`: Original URL
- `local_path`: Downloaded file path

## LLM Filtering Prompt

The LLM receives:
- Original query
- Optional user context
- Full web page content (up to 8000 chars)

It extracts:
- Relevant text passages
- Key points as bullet list
- Media descriptions with context
- Credibility score (source quality)
- Relevance score (query alignment)

## Media Handling

### Supported Types
- Images (jpg, png, gif, webp)
- Videos (mp4, webm)
- Audio (mp3, wav)
- PDFs
- Other binary files

### Download Process
1. LLM identifies media in content
2. Async download with configurable timeout
3. Files saved as: `{media_id}_{hash}{extension}`
4. Local path added to MediaFile object

### Output Directory Structure
```
data/media/
├── web_0_abc123_media_0_def456.jpg
├── web_0_abc123_media_1_789ghi.png
└── web_1_xyz789_media_0_klm012.pdf
```

## Integration with LangGraph Agent

### Option 1: Add Filtering Node (Recommended)

Update `src/orchestrator/tools.py` to add content filtering after web search:

```python
# Add new node
async def content_filter_node(state: AgentState) -> dict[str, Any]:
    """Step 4b: Filter web results through LLM and download media."""
    from src.retrieval.web_search import search_and_filter
    from src.retrieval.content_filter import format_for_agent_history

    config = get_config()
    query = state["query"]
    web_results = state["web_results"] or []

    if not web_results or not config.enable_content_filtering:
        return {"filtered_results": [], "agent_messages": []}

    # Apply LLM filtering
    filtered_results, _ = await search_and_filter(
        query,
        top_k=config.web_top_k,
        user_context=f"Original query: {query}"
    )

    # Format for agent message history
    messages = [format_for_agent_history(r) for r in filtered_results]

    return {"filtered_results": filtered_results, "agent_messages": messages}
```

Then update `src/orchestrator/agent.py`:

```python
# Add node to graph
graph.add_node("content_filter", content_filter_node)

# Update edges
graph.add_edge("web_search", "content_filter")
graph.add_edge("content_filter", "merge_dedupe")  # Or use filtered results
```

And update `src/orchestrator/state.py`:

```python
class AgentState(TypedDict):
    # ... existing fields
    filtered_results: Optional[list[FilteredWebResult]]
    agent_messages: Optional[list[str]]  # Formatted messages for agent history
```

### Option 2: Replace web_search_node Implementation

Modify the existing `web_search_node` in `src/orchestrator/tools.py`:

```python
async def web_search_node(state: AgentState) -> dict[str, Any]:
    """Step 4: Execute parallel web searches and apply LLM filtering."""
    config = get_config()
    gap_queries = state["gap_queries"] or []

    if not gap_queries:
        return {"web_results": []}

    if config.enable_content_filtering:
        # Use filtered search
        from src.retrieval.web_search import search_and_filter

        filter_tasks = [
            search_and_filter(query, config.web_top_k)
            for query in gap_queries
        ]
        results_list = await asyncio.gather(*filter_tasks)

        # Extract raw results for pipeline (backward compatible)
        web_results = []
        for filtered_results, raw_results in results_list:
            web_results.extend(raw_results)

        return {"web_results": web_results}
    else:
        # Original implementation
        search_tasks = [search_web(query, config.web_top_k) for query in gap_queries]
        search_results_list = await asyncio.gather(*search_tasks)

        web_results = []
        for results in search_results_list:
            web_results.extend(results)

        web_results = await fetch_and_extract(web_results)
        return {"web_results": web_results}
```

## Agent Message History Format

```markdown
## IPCC Climate Report 2023: Key Findings
**URL**: https://example.com/ipcc-2023
**Author**: Dr. Jane Smith
**Published**: 2023-03-15
**Relevance**: 0.95
**Credibility**: 0.90

### Extracted Content
The IPCC 2023 report shows global temperatures have risen 1.1°C
since pre-industrial times. Scientists warn we must limit warming
to 1.5°C to avoid catastrophic impacts. Renewable energy adoption
has increased 45% in the last decade...

### Key Points
- Global temperatures up 1.1°C since pre-industrial era
- Critical threshold is 1.5°C warming
- Renewable energy adoption up 45% in last decade
- Fossil fuel subsidies remain major barrier

### Media Files
- **Image**: Temperature trend chart 1850-2023
  - File: `data/media/web_0_abc123_media_0.jpg`
  - Context: Shows accelerating warming in recent decades
- **Chart**: Emission pathways for 1.5°C and 2°C scenarios
  - File: `data/media/web_0_abc123_media_1.png`
  - Context: Compares required emission reductions

---
```

## Performance Considerations

### Costs
- **LLM calls**: 1 per search result (gpt-4o-mini default)
- **Typical cost**: ~$0.001 per result
- **10 results**: ~$0.01 per query

### Latency
- **LLM filtering**: ~2-3s per result (parallel)
- **Media download**: ~1-2s per file (parallel)
- **Total overhead**: ~3-5s for 10 results

### Optimization Tips
1. Use `max_filter_concurrent` to control parallelism
2. Set `enable_content_filtering=false` to disable
3. Use `filter_model=gpt-4o-mini` for speed
4. Increase `media_download_timeout` for slow connections

## Testing

```bash
# Run filter tests
uv run pytest tests/retrieval/test_content_filter.py -v

# Test with real API (requires keys)
uv run pytest tests/retrieval/test_content_filter.py::test_extract_relevant_content -v

# Test without API (uses mocks)
uv run pytest tests/retrieval/test_content_filter.py::test_format_for_agent_history -v
```

## Error Handling

### Graceful Degradation
- **LLM fails**: Returns truncated original content
- **Download fails**: MediaFile has `local_path=None`
- **No content**: Skips result in pipeline

### Logging
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# See filtering details
from src.retrieval import content_filter
```

## Future Enhancements

1. **Image analysis**: Use vision model to analyze images
2. **Video transcription**: Extract text from videos
3. **Caching**: Cache filtered results by URL hash
4. **Streaming**: Stream filtered results as they complete
5. **Fact extraction**: Extract structured facts directly

## Related Files

- **Implementation**: `src/retrieval/content_filter.py`
- **Integration**: `src/retrieval/web_search.py`
- **Models**: `src/core/models.py` (FilteredWebResult, MediaFile)
- **Config**: `src/core/config.py`
- **Tests**: `tests/retrieval/test_content_filter.py`
