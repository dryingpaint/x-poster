# Orchestrator: LangGraph Agent

## Overview

The orchestrator implements a **LangGraph state machine** that coordinates the entire tweet generation pipeline. The agent executes a linear flow of 10 nodes, each responsible for a specific step in the process.

## Architecture

### State Machine Flow

```
START
  ↓
1. embed_query         → Embed user query with BGE-M3
  ↓
2. internal_search     → Hybrid search (vector + FTS) internal docs
  ↓
3. gap_analysis        → LLM identifies missing information
  ↓
4. web_search          → Parallel targeted web searches to fill gaps
  ↓
5. merge_dedupe        → Merge internal + web, dedupe, diversity
  ↓
6. rerank              → Cross-encoder reranking (BGE-reranker)
  ↓
7. evidence_pack       → LLM extracts facts from top results
  ↓
8. tweet_generation    → LLM generates tweet variants + thread
  ↓
9. fact_check          → LLM verifies all citations present
  ↓
10. prepare_response   → Assemble final response with sources
  ↓
END
```

### Files

| File | Purpose | Lines |
|------|---------|-------|
| `agent.py` | Graph definition and execution | ~120 |
| `state.py` | AgentState schema (TypedDict) | ~60 |
| `tools.py` | Node functions (async tools) | ~190 |

## Usage

### Basic Execution

```python
from src.core.models import GenerateRequest
from src.orchestrator.agent import run_agent

# Create request
request = GenerateRequest(
    prompt="What does Jason Hickel say about degrowth?",
    max_variants=3,
    max_thread_tweets=6,
)

# Run agent
response = await run_agent(request)

# Access results
for variant in response.variants:
    print(variant.text)
    print([c.source_id for c in variant.citations])

for source in response.sources:
    print(f"[{source.source_id}] {source.title} - {source.url}")
```

### Direct Graph Access

```python
from src.orchestrator.agent import create_agent_graph
from src.orchestrator.state import AgentState

# Create compiled graph
agent = create_agent_graph()

# Initialize state
initial_state: AgentState = {
    "query": "degrowth",
    "max_variants": 3,
    "max_thread_tweets": 6,
    "query_embedding": None,
    "internal_results": None,
    "gap_queries": None,
    "web_results": None,
    "merged_results": None,
    "final_results": None,
    "evidence": None,
    "variants": None,
    "thread": None,
    "response": None,
    "error": None,
}

# Execute graph
final_state = await agent.ainvoke(initial_state)

# Access final state
response = final_state["response"]
```

## State Schema

### AgentState (TypedDict)

All data flows through a shared state object:

```python
class AgentState(TypedDict):
    # Input parameters
    query: str
    max_variants: int
    max_thread_tweets: int

    # Step 1: Query embedding
    query_embedding: Optional[list[float]]

    # Step 2: Internal retrieval
    internal_results: Optional[list[SearchResult]]

    # Step 3: Gap analysis
    gap_queries: Optional[list[str]]

    # Step 4: Web search
    web_results: Optional[list[SearchResult]]

    # Step 6: Merge & dedupe
    merged_results: Optional[list[SearchResult]]

    # Step 7: Rerank
    final_results: Optional[list[SearchResult]]

    # Step 8: Evidence pack
    evidence: Optional[EvidencePack]

    # Step 9: Tweet generation
    variants: Optional[list[Tweet]]
    thread: Optional[list[Tweet]]

    # Step 11: Final response
    response: Optional[GenerateResponse]

    # Error handling
    error: Optional[str]
```

## Node Functions (Tools)

Each node is an async function that:
1. Reads relevant fields from state
2. Performs its operation
3. Returns dict with updated fields

### Node Details

| Node | Reads | Writes | External Calls |
|------|-------|--------|----------------|
| **embed_query** | query | query_embedding | BGE-M3 model |
| **internal_search** | query, query_embedding | internal_results | Supabase RPC |
| **gap_analysis** | query, internal_results | gap_queries | OpenAI LLM |
| **web_search** | gap_queries | web_results | EXA/Serper API |
| **merge_dedupe** | internal_results, web_results | merged_results | BGE-M3 embeddings |
| **rerank** | query, merged_results | final_results | BGE-reranker model |
| **evidence_pack** | query, final_results | evidence | OpenAI LLM |
| **tweet_generation** | query, evidence, params | variants, thread | OpenAI LLM |
| **fact_check** | variants, thread, evidence | variants, thread | OpenAI LLM |
| **prepare_response** | variants, thread, evidence | response | None |

## Performance

### Latency Breakdown

| Step | Typical Latency | Parallelizable |
|------|-----------------|----------------|
| 1. Embed query | ~100ms | No |
| 2. Internal search | ~300ms | No |
| 3. Gap analysis | ~2s | No |
| 4. Web search | ~3s | Yes (per gap query) |
| 5. Merge/dedupe | ~500ms | No |
| 6. Rerank | ~400ms | No |
| 7. Evidence pack | ~2s | No |
| 8. Tweet generation | ~4s | No |
| 9. Fact check | ~1s | Yes (variants) |
| 10. Prepare response | ~10ms | No |
| **Total** | **~12s** | - |

## Testing

### Unit Tests (Per Node)

Test individual nodes with mock state:

```python
@pytest.mark.asyncio
async def test_embed_query_node():
    state: AgentState = {
        "query": "test query",
        # ... other fields
    }

    result = await embed_query_node(state)

    assert "query_embedding" in result
    assert len(result["query_embedding"]) == 1024  # BGE-M3 dimension
```

### Integration Tests (Full Graph)

```python
@pytest.mark.asyncio
async def test_full_agent():
    request = GenerateRequest(
        prompt="test query",
        max_variants=1,
        max_thread_tweets=3,
    )

    response = await run_agent(request)

    assert response.variants
    assert response.sources
```

### Run Tests

```bash
# Unit tests
uv run pytest tests/orchestrator/ -v

# Integration tests
uv run pytest tests/test_pipeline.py -v

# Full suite
uv run pytest tests/ -v
```

## Extending the Agent

### Adding a New Node

1. **Define node function** in `tools.py`:
   ```python
   async def my_new_node(state: AgentState) -> dict[str, Any]:
       """My new processing step."""
       input_data = state["some_field"]
       output_data = process(input_data)
       return {"new_field": output_data}
   ```

2. **Update state schema** in `state.py`:
   ```python
   class AgentState(TypedDict):
       # ... existing fields
       new_field: Optional[MyType]
   ```

3. **Add to graph** in `agent.py`:
   ```python
   graph.add_node("my_node", my_new_node)
   graph.add_edge("previous_node", "my_node")
   graph.add_edge("my_node", "next_node")
   ```

### Conditional Branching

LangGraph supports conditional edges:

```python
def should_do_web_search(state: AgentState) -> str:
    """Decide whether to do web search based on gap analysis."""
    if state["gap_queries"]:
        return "web_search"
    else:
        return "merge_dedupe"  # Skip web search

# In graph
graph.add_conditional_edges(
    "gap_analysis",
    should_do_web_search,
    {
        "web_search": "web_search",
        "merge_dedupe": "merge_dedupe",
    }
)
```

## Best Practices

1. **Keep nodes pure**: Each node should only use data from state
2. **Minimize external calls**: Cache models, reuse clients
3. **Handle errors gracefully**: Set `state["error"]` instead of raising
4. **Use type hints**: AgentState is typed, maintain type safety
5. **Test nodes independently**: Unit test each node before integration
6. **Profile performance**: Measure node latency
7. **Document state changes**: Comment what each node reads/writes

## Related Documentation

- **Main docs**: [CLAUDE.md](../../CLAUDE.md) - Project overview
- **Parallel dev**: [PARALLEL_DEVELOPMENT.md](../../PARALLEL_DEVELOPMENT.md) - Team coordination
- **Content filtering**: [src/retrieval/CONTENT_FILTER_README.md](../retrieval/CONTENT_FILTER_README.md) - LLM filtering
- **Module READMEs**: Each `src/<module>/README.md` has detailed docs
