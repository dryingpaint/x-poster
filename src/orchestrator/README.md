# Orchestrator Module

## Overview
The orchestrator is the main pipeline coordinator that deterministically sequences all operations from query to final tweet generation.

## Responsibilities
- **Pipeline Coordination**: Execute the full generation pipeline in correct order
- **Error Handling**: Handle failures gracefully and provide useful error messages
- **Progress Tracking**: Log pipeline progress for observability
- **Result Assembly**: Combine outputs from all modules into final response

## Key Files
- `pipeline.py`: Main deterministic pipeline implementation

## Development Guidelines

### ⚠️ MANDATORY: Test-First Development

**YOU MUST WRITE INTEGRATION TESTS BEFORE IMPLEMENTATION**

**TDD for Pipeline Changes:**
```bash
# 1. Write integration test FIRST (RED)
cat > tests/orchestrator/test_fast_pipeline.py << 'EOF'
from src.orchestrator.pipeline import run_fast_pipeline

@pytest.mark.asyncio
async def test_fast_pipeline_completes_under_5s():
    """Test fast pipeline variant meets latency target."""
    request = GenerateRequest(prompt="AI safety")

    start = time.time()
    response = await run_fast_pipeline(request)
    duration = time.time() - start

    assert duration < 5.0  # Fast mode target
    assert len(response.variants) > 0
EOF

# 2. Run and watch fail (RED)
uv run pytest tests/orchestrator/test_fast_pipeline.py -v

# 3. Implement fast pipeline (GREEN)
# Add run_fast_pipeline() to src/orchestrator/pipeline.py
```

**Why Test-First for Orchestrator?**
- **Integration safety**: Catches module interaction bugs
- **Performance**: Tests enforce latency budgets
- **Correctness**: End-to-end validation

### Working on Pipeline (`pipeline.py`)
**What you can do in parallel:**
- Add pipeline observability (metrics, tracing)
- Add pipeline configuration (enable/disable stages)
- Add retry logic for transient failures
- Add pipeline variants (fast mode, high-quality mode)

**What requires coordination:**
- Changing pipeline stage order (may break correctness)
- Modifying GenerateRequest/GenerateResponse (breaks API contract)
- Adding/removing pipeline stages (impacts latency)
- Changing error handling strategy (affects user experience)

**Testing requirements:**
```bash
# Test full pipeline end-to-end
uv run pytest tests/orchestrator/test_pipeline.py

# Test with various query types
uv run pytest tests/orchestrator/test_pipeline.py::test_technical_query
uv run pytest tests/orchestrator/test_pipeline.py::test_current_events_query

# Test error handling
uv run pytest tests/orchestrator/test_pipeline.py::test_no_results
uv run pytest tests/orchestrator/test_pipeline.py::test_api_failure
```

**Code style:**
```python
# GOOD: Clear pipeline stages with progress logging
async def run_generation_pipeline(request: GenerateRequest) -> GenerateResponse:
    """Execute full deterministic pipeline.

    Pipeline Stages:
    1. Embed query
    2. Internal retrieval
    3. Gap analysis
    4. Web retrieval (gap-filling)
    5. Merge and dedupe
    6. Rerank
    7. Evidence assembly
    8. Tweet generation
    9. Fact-checking
    10. Response assembly

    Args:
        request: GenerateRequest with prompt and parameters

    Returns:
        GenerateResponse with tweets and sources
    """
    config = get_config()

    # Stage 1: Embed query
    print(f"📝 Query: {request.prompt}")
    print("🔢 Embedding query...")
    query_embedding = embed_text(request.prompt)

    # Stage 2: Internal retrieval
    print("📚 Retrieving from internal dataset...")
    internal_results = await search_internal(
        query=request.prompt,
        query_embedding=query_embedding,
        top_k=config.internal_top_k,
    )
    print(f"   Found {len(internal_results)} internal results")

    # Continue with other stages...
    # Each stage logs progress and handles errors

    return response

# GOOD: Add error recovery
async def run_generation_pipeline_with_retry(
    request: GenerateRequest,
    max_retries: int = 2
) -> GenerateResponse:
    """Run pipeline with retry logic."""
    for attempt in range(max_retries + 1):
        try:
            return await run_generation_pipeline(request)
        except TransientError as e:
            if attempt < max_retries:
                logger.warning(f"Pipeline failed (attempt {attempt + 1}), retrying: {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise

# GOOD: Add pipeline variants
async def run_fast_pipeline(request: GenerateRequest) -> GenerateResponse:
    """Fast pipeline variant: skip web search for low-latency."""
    # Only internal retrieval + generation
    query_embedding = embed_text(request.prompt)
    internal_results = await search_internal(
        query=request.prompt,
        query_embedding=query_embedding,
        top_k=10
    )

    evidence = await create_evidence_pack(request.prompt, internal_results)
    variants, thread = await generate_tweets(
        request.prompt,
        evidence,
        max_variants=request.max_variants
    )

    return GenerateResponse(variants=variants, thread=thread, sources=[])

# GOOD: Add pipeline observability
from contextlib import asynccontextmanager
import time

@asynccontextmanager
async def pipeline_stage(stage_name: str):
    """Context manager for tracking pipeline stage timing."""
    start_time = time.time()
    print(f"▶ Starting stage: {stage_name}")

    try:
        yield
    finally:
        duration = time.time() - start_time
        print(f"✓ Completed stage: {stage_name} ({duration:.2f}s)")

async def run_instrumented_pipeline(request: GenerateRequest) -> GenerateResponse:
    """Pipeline with detailed timing instrumentation."""
    async with pipeline_stage("Query Embedding"):
        query_embedding = embed_text(request.prompt)

    async with pipeline_stage("Internal Retrieval"):
        internal_results = await search_internal(...)

    # ... rest of pipeline with timing

    return response
```

## Interface Contract

### Main Pipeline
```python
from src.orchestrator.pipeline import run_generation_pipeline
from src.core.models import GenerateRequest

# Run full pipeline
request = GenerateRequest(
    prompt="AI safety research in 2025",
    max_variants=3,
    max_thread_tweets=6
)

response = await run_generation_pipeline(request)
# Returns: GenerateResponse with variants, thread, and sources
```

### Request Format
```python
class GenerateRequest(BaseModel):
    prompt: str  # User's query/topic
    max_variants: int = 3  # Number of single tweet variants
    max_thread_tweets: int = 6  # Max tweets in thread
```

### Response Format
```python
class GenerateResponse(BaseModel):
    variants: list[Tweet]  # Single tweet variants
    thread: list[Tweet]  # Thread of tweets
    sources: list[Source]  # All cited sources
```

## Pipeline Architecture

### Linear Deterministic Flow
```
Input: GenerateRequest
  ↓
[1] Embed Query
  ↓
[2] Internal Search → Results
  ↓
[3] Gap Analysis → Gap Queries
  ↓
[4] Web Search (gaps) → Web Results
  ↓
[5] Merge + Dedupe → Combined Results
  ↓
[6] Rerank → Top K Results
  ↓
[7] Evidence Pack → Facts + Sources
  ↓
[8] Tweet Generation → Draft Tweets
  ↓
[9] Fact-Check → Verified Tweets
  ↓
[10] Assemble Response
  ↓
Output: GenerateResponse
```

### Key Design Principles
1. **Deterministic**: Same input → same output (given same data)
2. **Sequential**: Each stage completes before next starts
3. **Fail-Fast**: Early exit if critical stage fails
4. **Observable**: Progress logging at each stage
5. **Composable**: Stages can be skipped or reordered for variants

## Dependencies
**External:**
- `asyncio`: Async/await pipeline execution

**Internal:**
- `src.core.config`: Pipeline configuration
- `src.core.models`: All data models
- `src.db.operations`: Internal search
- `src.retrieval.*`: Web search, merging, reranking
- `src.generation.*`: Embeddings, evidence, writing, fact-checking

## Performance Considerations
- **Total Latency**: Target <12s p50, <20s p95
- **Stage Breakdown**:
  - Embed: ~100ms
  - Internal Search: ~500ms
  - Gap Analysis: ~2s
  - Web Search: ~2s
  - Rerank: ~500ms
  - Evidence: ~3s
  - Generation: ~4s
  - Fact-Check: ~2s
- **Optimization**: Parallel operations within stages (e.g., multiple web searches)

### Latency Optimization
```python
# GOOD: Parallel gap-filling web searches
web_tasks = [
    search_web(query=gap_query, top_k=config.web_top_k // len(gap_queries))
    for gap_query in gap_queries
]
web_results_list = await asyncio.gather(*web_tasks)

# GOOD: Skip optional stages for fast mode
if config.fast_mode:
    # Skip web search entirely
    final_results = internal_results
else:
    # Full pipeline with web search
    gap_queries = await analyze_gaps(...)
    web_results = await search_web(...)

# GOOD: Early exit on empty results
if not internal_results:
    print("⚠️  No internal results found")
    if not config.web_fallback:
        return GenerateResponse(variants=[], thread=[], sources=[])
```

## Error Handling Strategy

### Error Categories
1. **User Errors**: Invalid input → return error message
2. **No Results**: Valid query but no data → return empty response
3. **Transient Errors**: API timeout, rate limit → retry
4. **Critical Errors**: Database down, model loading failed → raise exception

### Error Handling Pattern
```python
# GOOD: Graceful degradation
try:
    web_results = await search_web(query)
except APIError as e:
    logger.warning(f"Web search failed: {e}")
    web_results = []  # Continue with internal results only

# GOOD: Meaningful error messages
if not final_results:
    print("⚠️  No results found for query. Try:")
    print("   - More specific keywords")
    print("   - Different phrasing")
    print("   - Checking document database is populated")
    return GenerateResponse(variants=[], thread=[], sources=[])

# GOOD: Critical errors bubble up
try:
    query_embedding = embed_text(request.prompt)
except ModelLoadError:
    # Can't proceed without embeddings
    raise PipelineError("Embedding model failed to load")
```

## Common Pitfalls
- **DON'T** skip error handling (errors will cascade)
- **DON'T** run stages in wrong order (breaks correctness)
- **DON'T** forget to log progress (hard to debug failures)
- **DON'T** modify request during pipeline (breaks reproducibility)
- **DO** validate request before starting pipeline
- **DO** handle empty results at each stage
- **DO** provide clear progress indicators

## Testing Checklist
- [ ] Full pipeline completes successfully with valid input
- [ ] Pipeline handles empty internal results
- [ ] Pipeline handles web search failures
- [ ] Pipeline handles LLM API errors
- [ ] Pipeline returns empty response when no evidence found
- [ ] Pipeline latency < 15s for typical queries
- [ ] All stages log progress correctly

## Testing Patterns

### End-to-End Tests
```python
# Test full pipeline
async def test_full_pipeline():
    request = GenerateRequest(
        prompt="AI safety research",
        max_variants=3
    )

    response = await run_generation_pipeline(request)

    assert len(response.variants) > 0
    assert len(response.sources) > 0
    assert all(len(tweet.citations) > 0 for tweet in response.variants)

# Test with no internal results
async def test_pipeline_web_only():
    # Mock empty internal results
    with patch('src.db.operations.search_internal', return_value=[]):
        request = GenerateRequest(prompt="breaking news today")
        response = await run_generation_pipeline(request)

        # Should still get results from web search
        assert len(response.variants) > 0

# Test error handling
async def test_pipeline_api_failure():
    # Mock API failure
    with patch('src.retrieval.web_search.search_web', side_effect=APIError):
        request = GenerateRequest(prompt="test query")
        response = await run_generation_pipeline(request)

        # Should gracefully handle and continue with internal only
        assert isinstance(response, GenerateResponse)
```

### Integration Tests
```python
# Test with real APIs (slow, use sparingly)
@pytest.mark.integration
async def test_pipeline_integration():
    """Test with real database and APIs."""
    request = GenerateRequest(
        prompt="latest AI developments",
        max_variants=2
    )

    response = await run_generation_pipeline(request)

    # Validate response quality
    assert len(response.variants) == 2
    assert all(len(tweet.text) <= 280 for tweet in response.variants)
    assert all(validate_citations(tweet) for tweet in response.variants)
```

## Configuration

### Pipeline Configuration
```python
# In src/core/config.py
class Config(BaseModel):
    # Retrieval
    internal_top_k: int = 15
    web_top_k: int = 10
    rerank_k: int = 20
    final_top_k: int = 8

    # Generation
    max_gap_queries: int = 3
    min_confidence: float = 0.7

    # Performance
    timeout_seconds: int = 30
    enable_caching: bool = True
```

### Usage
```python
from src.core.config import get_config

config = get_config()

# Use config throughout pipeline
internal_results = await search_internal(
    query=request.prompt,
    top_k=config.internal_top_k  # Configurable
)
```

## Monitoring and Observability

### Metrics to Track
- Pipeline latency (p50, p95, p99)
- Success rate (% successful completions)
- Error rate by stage
- Result quality (citation count, source diversity)
- API usage (OpenAI tokens, search queries)

### Logging Best Practices
```python
# GOOD: Structured logging with context
logger.info("pipeline_stage_complete", extra={
    "stage": "internal_retrieval",
    "duration_ms": duration_ms,
    "result_count": len(internal_results),
    "query": request.prompt
})

# GOOD: Log decision points
if not gap_queries:
    logger.info("gap_analysis_skipped", extra={
        "reason": "sufficient_internal_knowledge",
        "internal_result_count": len(internal_results)
    })

# GOOD: Log performance issues
if duration_ms > 3000:
    logger.warning("slow_stage", extra={
        "stage": "web_search",
        "duration_ms": duration_ms,
        "threshold_ms": 3000
    })
```

## Contact for Coordination
When modifying orchestrator:
1. Document pipeline changes in CLAUDE.md
2. Update latency budgets for new stages
3. Test end-to-end with diverse queries
4. Measure impact on success rate and quality
5. Coordinate with all affected modules
