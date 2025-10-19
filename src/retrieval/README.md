# Retrieval Module

## Overview
Handles hybrid retrieval combining internal vector search with live web search, result merging, and reranking.

## Responsibilities
- **Web Search**: Query external search engines (EXA, Serper) for recent information
- **Result Merging**: Combine and deduplicate internal + web results
- **Reranking**: Cross-encoder reranking for relevance optimization
- **Content Extraction**: Fetch and parse web pages

## Key Files
- `web_search.py`: Web search providers (EXA, Serper) and content extraction
- `merger.py`: Result merging and deduplication logic
- `reranker.py`: Cross-encoder reranking with BGE-reranker-large

## Development Guidelines

### ⚠️ MANDATORY: Test-First Development

**YOU MUST WRITE TESTS BEFORE IMPLEMENTATION**

Follow the TDD cycle: 🔴 RED → 🟢 GREEN → 🔵 REFACTOR

**Example: Adding Brave Search Provider**
```bash
# 1. Write test FIRST (RED)
cat > tests/retrieval/test_brave_search.py << 'EOF'
from src.retrieval.web_search import BraveSearchProvider

@pytest.mark.asyncio
async def test_brave_search_returns_results():
    provider = BraveSearchProvider(api_key="test")
    results = await provider.search("AI safety", top_k=10)

    assert len(results) <= 10
    assert all(r.source_type == "web" for r in results)
EOF

# 2. Run and watch fail (RED)
uv run pytest tests/retrieval/test_brave_search.py -v

# 3. Implement (GREEN)
# Add BraveSearchProvider to src/retrieval/web_search.py

# 4. Run until pass
uv run pytest tests/retrieval/test_brave_search.py -v

# 5. Refactor (REFACTOR)
uv run pytest tests/retrieval/ -v
```

**Why Test-First?**
- Web APIs change frequently - tests catch breakage
- Parallel safety - define interface before implementation
- Quality - test error handling, rate limits, edge cases

### Working on Web Search (`web_search.py`)
**What you can do in parallel:**
- Add new search providers (Bing, Google, Brave Search)
- Improve content extraction (better HTML parsing, JavaScript rendering)
- Add result filtering (by date, domain, content type)
- Add search query optimization (query expansion, reformulation)

**What requires coordination:**
- Changing SearchResult format (impacts merger and downstream)
- Modifying default search parameters (may affect result quality)
- Changing rate limiting logic (may impact API costs)

**Testing requirements:**
```bash
# Test search providers
uv run pytest tests/retrieval/test_web_search.py

# Test with live APIs (requires keys)
export EXA_API_KEY="your_key"
uv run pytest tests/retrieval/test_web_search.py::test_exa_search

# Test content extraction
uv run pytest tests/retrieval/test_web_search.py::test_content_extraction
```

**Code style:**
```python
# GOOD: Abstract search provider interface
from abc import ABC, abstractmethod
from src.core.models import SearchResult

class SearchProvider(ABC):
    """Base interface for search providers."""

    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict | None = None
    ) -> list[SearchResult]:
        """Execute search and return results."""
        pass

class EXASearchProvider(SearchProvider):
    """EXA search implementation."""

    async def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Search using EXA API."""
        response = await self._call_api(query, num_results=top_k)

        return [
            SearchResult(
                source_id=f"exa_{result['id']}",
                content=result['text'],
                title=result['title'],
                url=result['url'],
                published_at=result.get('published_date'),
                score=result['score'],
                source_type="web"
            )
            for result in response['results']
        ]

# GOOD: Handle API errors gracefully
async def search_web(query: str, top_k: int = 10) -> list[SearchResult]:
    """Search web with fallback providers."""
    # Try primary provider (EXA)
    try:
        return await exa_provider.search(query, top_k)
    except APIError as e:
        logger.warning(f"EXA failed: {e}, falling back to Serper")

    # Fallback to Serper
    try:
        return await serper_provider.search(query, top_k)
    except APIError as e:
        logger.error(f"All search providers failed: {e}")
        return []

# GOOD: Extract clean content from web pages
async def fetch_and_extract(results: list[SearchResult]) -> list[SearchResult]:
    """Fetch full content for search results."""
    async def fetch_one(result: SearchResult) -> SearchResult:
        try:
            html = await fetch_url(result.url)
            content = extract_main_content(html)
            result.content = content
            return result
        except Exception as e:
            logger.warning(f"Failed to fetch {result.url}: {e}")
            return result  # Return with original snippet

    # Fetch in parallel with rate limiting
    fetched = await asyncio.gather(*[
        fetch_one(result) for result in results
    ])

    return fetched

def extract_main_content(html: str) -> str:
    """Extract main content from HTML, removing boilerplate."""
    soup = BeautifulSoup(html, 'html.parser')

    # Remove script, style, nav, footer
    for tag in soup(['script', 'style', 'nav', 'footer', 'aside']):
        tag.decompose()

    # Extract main content
    main = soup.find('main') or soup.find('article') or soup.find('body')
    text = main.get_text(separator=' ', strip=True)

    # Clean whitespace
    text = ' '.join(text.split())

    return text
```

### Working on Merging (`merger.py`)
**What you can do in parallel:**
- Add sophisticated deduplication (fuzzy matching, semantic similarity)
- Add result diversity optimization (avoid redundant sources)
- Add source quality scoring (domain authority, freshness)
- Add merging strategies (RRF, weighted combination)

**What requires coordination:**
- Changing deduplication threshold (affects result count)
- Modifying internal/web priority weights (impacts result mix)
- Changing final result count (affects downstream processing)

**Testing requirements:**
```bash
# Test merging logic
uv run pytest tests/retrieval/test_merger.py

# Test deduplication accuracy
uv run pytest tests/retrieval/test_merger.py::test_deduplication

# Test source diversity
uv run pytest tests/retrieval/test_merger.py::test_diversity
```

**Code style:**
```python
# GOOD: Clear merging strategy with configurable weights
from src.core.models import SearchResult
from sklearn.metrics.pairwise import cosine_similarity

def merge_and_dedupe_results(
    internal_results: list[SearchResult],
    web_results: list[SearchResult],
    internal_embeddings: list[list[float]],
    web_embeddings: list[list[float]],
    final_k: int = 20,
    similarity_threshold: float = 0.85
) -> list[SearchResult]:
    """Merge internal and web results with deduplication.

    Strategy:
    1. Prioritize internal results (higher quality)
    2. Add web results that are sufficiently different
    3. Remove near-duplicates using semantic similarity

    Args:
        internal_results: Results from internal database
        web_results: Results from web search
        internal_embeddings: Embeddings for internal results
        web_embeddings: Embeddings for web results
        final_k: Target number of final results
        similarity_threshold: Cosine similarity threshold for deduplication

    Returns:
        Merged and deduplicated results
    """
    merged = []
    merged_embeddings = []

    # Add all internal results first (prioritize internal)
    for result, embedding in zip(internal_results, internal_embeddings):
        merged.append(result)
        merged_embeddings.append(embedding)

    # Add web results that are sufficiently different
    for web_result, web_embedding in zip(web_results, web_embeddings):
        # Check similarity to all existing results
        is_duplicate = False
        for existing_embedding in merged_embeddings:
            similarity = cosine_similarity(
                [web_embedding],
                [existing_embedding]
            )[0][0]

            if similarity > similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            merged.append(web_result)
            merged_embeddings.append(web_embedding)

        # Stop when we have enough results
        if len(merged) >= final_k:
            break

    return merged[:final_k]

# GOOD: Add diversity optimization
def diversify_results(
    results: list[SearchResult],
    embeddings: list[list[float]],
    final_k: int = 10,
    diversity_weight: float = 0.3
) -> list[SearchResult]:
    """Select diverse results using MMR (Maximal Marginal Relevance).

    Args:
        results: Candidate results
        embeddings: Result embeddings
        final_k: Number of results to select
        diversity_weight: Weight for diversity vs relevance (0-1)

    Returns:
        Diverse subset of results
    """
    selected = []
    selected_embeddings = []
    remaining_indices = list(range(len(results)))

    # Select first result (highest score)
    first_idx = max(remaining_indices, key=lambda i: results[i].score)
    selected.append(results[first_idx])
    selected_embeddings.append(embeddings[first_idx])
    remaining_indices.remove(first_idx)

    # Select remaining results with MMR
    while len(selected) < final_k and remaining_indices:
        mmr_scores = []

        for idx in remaining_indices:
            # Relevance score
            relevance = results[idx].score

            # Maximum similarity to selected results
            max_sim = max(
                cosine_similarity([embeddings[idx]], [emb])[0][0]
                for emb in selected_embeddings
            )

            # MMR = (1-λ) * relevance + λ * diversity
            mmr = (1 - diversity_weight) * relevance + diversity_weight * (1 - max_sim)
            mmr_scores.append(mmr)

        # Select result with highest MMR
        best_idx = remaining_indices[mmr_scores.index(max(mmr_scores))]
        selected.append(results[best_idx])
        selected_embeddings.append(embeddings[best_idx])
        remaining_indices.remove(best_idx)

    return selected
```

### Working on Reranking (`reranker.py`)
**What you can do in parallel:**
- Add multiple reranker models (cross-encoder, ColBERT)
- Add reranking strategies (two-stage, ensemble)
- Add reranking cache to avoid redundant computation
- Add query-specific reranking optimization

**What requires coordination:**
- Changing reranker model (may affect quality and speed)
- Modifying top_k parameter (impacts final result count)
- Changing score normalization (may affect downstream thresholds)

**Testing requirements:**
```bash
# Test reranking quality
uv run pytest tests/retrieval/test_reranker.py

# Test reranking improves ranking
uv run pytest tests/retrieval/test_reranker.py::test_ranking_improvement

# Benchmark reranking speed
uv run pytest tests/retrieval/test_reranker.py::test_reranking_speed
```

**Code style:**
```python
# GOOD: Use cross-encoder for accurate reranking
from sentence_transformers import CrossEncoder

class Reranker:
    """Cross-encoder reranker for result refinement."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 10
    ) -> list[SearchResult]:
        """Rerank results using cross-encoder.

        Args:
            query: Search query
            results: Candidate results to rerank
            top_k: Number of top results to return

        Returns:
            Reranked results with updated scores
        """
        # Prepare query-document pairs
        pairs = [[query, result.content] for result in results]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Update result scores
        for result, score in zip(results, scores):
            result.score = float(score)

        # Sort by new scores
        reranked = sorted(results, key=lambda r: r.score, reverse=True)

        return reranked[:top_k]

# GOOD: Add batch processing for efficiency
def rerank_results(
    query: str,
    results: list[SearchResult],
    top_k: int = 10,
    batch_size: int = 32
) -> list[SearchResult]:
    """Rerank results in batches for efficiency."""
    reranker = get_reranker()

    # Process in batches
    all_scored = []
    for i in range(0, len(results), batch_size):
        batch = results[i:i+batch_size]
        scored_batch = reranker.rerank(query, batch, top_k=len(batch))
        all_scored.extend(scored_batch)

    # Sort all results by score
    final = sorted(all_scored, key=lambda r: r.score, reverse=True)

    return final[:top_k]

# GOOD: Cache reranker model
from functools import lru_cache

@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    """Get cached reranker instance."""
    return Reranker()
```

## Interface Contract

### Web Search
```python
from src.retrieval.web_search import search_web, fetch_and_extract

# Search web
results = await search_web(
    query="AI safety research",
    top_k=10
)
# Returns: list[SearchResult] with snippets

# Fetch full content
results = await fetch_and_extract(results)
# Returns: list[SearchResult] with full text
```

### Result Merging
```python
from src.retrieval.merger import merge_and_dedupe_results

# Merge internal + web results
merged = merge_and_dedupe_results(
    internal_results=internal_results,
    web_results=web_results,
    internal_embeddings=internal_embeddings,
    web_embeddings=web_embeddings,
    final_k=20
)
# Returns: list[SearchResult]
```

### Reranking
```python
from src.retrieval.reranker import rerank_results

# Rerank for relevance
reranked = rerank_results(
    query="AI safety research",
    results=merged_results,
    top_k=10
)
# Returns: list[SearchResult] sorted by relevance
```

## Pipeline Flow
```
Query → [Internal Search] → Internal Results
            ↓                       ↓
      [Gap Analysis]          [Merge + Dedupe]
            ↓                       ↓
      [Web Search] → Web Results → [Rerank]
            ↓                       ↓
      [Fetch Content]         Final Results
```

## Dependencies
**External:**
- `httpx`: Async HTTP client for API calls
- `beautifulsoup4`: HTML parsing for content extraction
- `sentence-transformers`: Cross-encoder reranking
- `scikit-learn`: Cosine similarity for deduplication

**Internal:**
- `src.core.models`: SearchResult
- `src.core.config`: API keys, search settings

## Performance Considerations
- **Web Search**: ~500ms-2s per query (API latency)
- **Content Extraction**: ~200ms per page
- **Deduplication**: ~50ms for 100 results
- **Reranking**: ~500ms for 20 results (batch of 32)
- **Parallelization**: Fetch multiple pages concurrently

### Optimization Tips
```python
# GOOD: Parallel web searches for gap queries
import asyncio
results = await asyncio.gather(*[
    search_web(query) for query in gap_queries
])

# GOOD: Rate limiting for API calls
from asyncio import Semaphore

semaphore = Semaphore(5)  # Max 5 concurrent requests

async def search_with_limit(query: str):
    async with semaphore:
        return await search_web(query)

# GOOD: Timeout for slow content fetches
async def fetch_url(url: str, timeout: float = 5.0):
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            return response.text
    except httpx.TimeoutException:
        logger.warning(f"Timeout fetching {url}")
        return ""
```

## Common Pitfalls
- **DON'T** fetch all web results synchronously (use parallel requests)
- **DON'T** skip deduplication (wastes reranking compute)
- **DON'T** rerank too many results (diminishing returns after ~50)
- **DON'T** forget to handle API rate limits
- **DO** cache reranker model (expensive to load)
- **DO** handle network errors gracefully
- **DO** validate search result quality (non-empty content)

## Testing Checklist
- [ ] Web search returns relevant results (manual inspection)
- [ ] Content extraction removes boilerplate (nav, footer, ads)
- [ ] Deduplication removes near-duplicates (>85% similarity)
- [ ] Reranking improves NDCG@10 by >10%
- [ ] Error handling for failed API calls
- [ ] Rate limiting prevents API quota exhaustion
- [ ] Performance: <3s for full retrieval pipeline

## Local Development Setup
```bash
# Set API keys in .env
export EXA_API_KEY="your_exa_key"
export SERPER_API_KEY="your_serper_key"

# Install dependencies
uv sync

# Run tests
uv run pytest tests/retrieval/ -v

# Test with live APIs (requires keys)
uv run pytest tests/retrieval/test_web_search.py -v
```

## Contact for Coordination
When modifying retrieval pipeline:
1. Measure impact on result quality (NDCG@10, Precision@10)
2. Benchmark latency (target: <3s for full pipeline)
3. Monitor API costs (EXA: $5/1000 queries)
4. Coordinate with generation module on result format changes
