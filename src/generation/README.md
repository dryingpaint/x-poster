# Generation Module

## Overview
Handles all LLM-based generation: embeddings, evidence assembly, gap analysis, tweet writing, and fact-checking.

## Responsibilities
- **Embeddings**: Generate BGE-M3 embeddings for semantic search
- **Gap Analysis**: Identify missing information in internal results
- **Evidence Assembly**: Extract facts and quotes from search results
- **Tweet Generation**: Write evidence-based tweets with inline citations
- **Fact-Checking**: Verify citations and accuracy

## Key Files
- `embeddings.py`: BGE-M3 embedding generation
- `gap_analysis.py`: Knowledge gap identification
- `evidence.py`: Evidence pack creation from search results
- `writer.py`: Tweet generation with citations
- `factcheck.py`: Citation and fact verification

## Development Guidelines

### ⚠️ MANDATORY: Test-First Development

**YOU MUST WRITE TESTS/EVALUATIONS BEFORE IMPLEMENTATION**

For LLM-based features, "tests" include both unit tests AND quality evaluations.

**TDD Cycle for LLM Features:**
```bash
# 1. Write evaluation test FIRST (RED)
cat > tests/generation/test_gap_analysis_quality.py << 'EOF'
import pytest
from src.generation.gap_analysis import analyze_gaps

@pytest.mark.asyncio
async def test_gap_analysis_identifies_missing_stats():
    """Eval: Gap analysis should identify missing statistics."""
    query = "AI safety progress in 2024"
    internal_results = [
        # Mock results with only qualitative info
        SearchResult(content="AI safety is important...")
    ]

    gaps = await analyze_gaps(query, internal_results)

    # Should identify need for statistics/data
    assert any("statistic" in gap.lower() or "data" in gap.lower()
              for gap in gaps)
    assert len(gaps) <= 3  # Don't over-query
EOF

# 2. Run and watch fail (RED)
uv run pytest tests/generation/test_gap_analysis_quality.py -v

# 3. Implement improved prompt (GREEN)
# Update prompt in src/generation/gap_analysis.py

# 4. Run until eval passes
uv run pytest tests/generation/test_gap_analysis_quality.py -v

# 5. Refactor prompt (REFACTOR)
```

**Why Test-First for LLM Modules?**
- **Eval-driven**: Tests define quality bar before coding
- **Regression detection**: Prompt changes can break quality
- **Determinism**: Test with fixed examples for reproducibility

### Working on Embeddings (`embeddings.py`)
**What you can do in parallel:**
- Add embedding model variants (multilingual, domain-specific)
- Add embedding caching to reduce API calls
- Add batch processing optimization
- Add embedding quality monitoring

**What requires coordination:**
- Changing embedding model (requires database migration)
- Modifying embedding dimensions (breaks vector search)
- Changing normalization strategy (affects similarity scores)

**Testing requirements:**
```bash
# Test embedding generation
uv run pytest tests/generation/test_embeddings.py

# Test embedding quality
uv run pytest tests/generation/test_embeddings.py::test_embedding_similarity

# Benchmark embedding speed
uv run pytest tests/generation/test_embeddings.py::test_batch_performance
```

**Code style:**
```python
# GOOD: Singleton model loading with caching
from functools import lru_cache
from sentence_transformers import SentenceTransformer

@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Get cached embedding model instance."""
    return SentenceTransformer('BAAI/bge-m3')

def embed_text(text: str) -> list[float]:
    """Generate embedding for single text.

    Args:
        text: Input text

    Returns:
        1024-dimensional embedding vector
    """
    model = get_embedding_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()

def embed_batch(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Generate embeddings for batch of texts.

    Args:
        texts: List of input texts
        batch_size: Batch size for processing

    Returns:
        List of embedding vectors
    """
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return embeddings.tolist()

# GOOD: Add embedding cache for repeated queries
from functools import lru_cache

@lru_cache(maxsize=1000)
def embed_text_cached(text: str) -> list[float]:
    """Generate embedding with caching for repeated texts."""
    return embed_text(text)
```

### Working on Gap Analysis (`gap_analysis.py`)
**What you can do in parallel:**
- Improve gap detection prompts
- Add structured gap types (stats, definitions, recent events)
- Add gap scoring to prioritize searches
- Add user feedback integration for gap quality

**What requires coordination:**
- Changing gap query format (impacts web search)
- Modifying number of gap queries (affects latency)
- Changing LLM model (may affect cost/quality)

**Testing requirements:**
```bash
# Test gap analysis
uv run pytest tests/generation/test_gap_analysis.py

# Test gap relevance
uv run pytest tests/generation/test_gap_analysis.py::test_gap_quality

# Test with various query types
uv run pytest tests/generation/test_gap_analysis.py::test_technical_gaps
uv run pytest tests/generation/test_gap_analysis.py::test_temporal_gaps
```

**Code style:**
```python
# GOOD: Structured gap analysis with clear prompts
from openai import AsyncOpenAI
from src.core.models import SearchResult

async def analyze_gaps(
    query: str,
    internal_results: list[SearchResult]
) -> list[str]:
    """Identify knowledge gaps in internal results.

    Args:
        query: User's query
        internal_results: Results from internal database

    Returns:
        List of targeted search queries to fill gaps
    """
    client = AsyncOpenAI()

    # Summarize what we have
    content_summary = "\n\n".join([
        f"Source {i+1}: {r.title}\n{r.content[:200]}..."
        for i, r in enumerate(internal_results[:5])
    ])

    prompt = f"""You are analyzing whether internal knowledge is sufficient to answer this query:
Query: {query}

Available Internal Knowledge:
{content_summary}

Identify up to 3 specific knowledge gaps that would improve the answer. Focus on:
1. Recent statistics, data, or events (post-2023)
2. Specific examples or case studies
3. Expert opinions or authoritative sources

For each gap, generate a targeted web search query.

Output format (one query per line):
<query>specific search query 1</query>
<query>specific search query 2</query>
<query>specific search query 3</query>"""

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200
    )

    # Parse gap queries
    content = response.choices[0].message.content
    queries = []
    for line in content.split('\n'):
        if line.strip().startswith('<query>') and line.strip().endswith('</query>'):
            query_text = line.strip()[7:-8]  # Remove <query></query> tags
            queries.append(query_text)

    return queries[:3]  # Limit to 3 gap queries

# GOOD: Add gap scoring to prioritize
async def analyze_gaps_with_priority(
    query: str,
    internal_results: list[SearchResult]
) -> list[tuple[str, float]]:
    """Identify and score knowledge gaps by priority."""
    gaps = await analyze_gaps(query, internal_results)

    # Score each gap by urgency (0-1)
    scored_gaps = []
    for gap_query in gaps:
        # Heuristics: recent data and stats are high priority
        score = 0.5
        if any(word in gap_query.lower() for word in ['2024', '2025', 'recent']):
            score += 0.3
        if any(word in gap_query.lower() for word in ['statistics', 'data', 'numbers']):
            score += 0.2

        scored_gaps.append((gap_query, min(score, 1.0)))

    # Sort by priority
    scored_gaps.sort(key=lambda x: x[1], reverse=True)

    return scored_gaps
```

### Working on Evidence Assembly (`evidence.py`)
**What you can do in parallel:**
- Improve fact extraction prompts
- Add confidence scoring for facts
- Add fact deduplication
- Add quote quality validation (length, coherence)

**What requires coordination:**
- Changing EvidenceFact format (impacts writer and fact-checker)
- Modifying fact extraction criteria (affects quality)
- Changing LLM model (impacts cost/speed)

**Testing requirements:**
```bash
# Test evidence extraction
uv run pytest tests/generation/test_evidence.py

# Test fact quality
uv run pytest tests/generation/test_evidence.py::test_fact_accuracy

# Test quote extraction
uv run pytest tests/generation/test_evidence.py::test_quote_quality
```

**Code style:**
```python
# GOOD: Structured evidence extraction
from openai import AsyncOpenAI
from src.core.models import EvidenceFact, EvidencePack, SearchResult
import json

async def create_evidence_pack(
    query: str,
    search_results: list[SearchResult]
) -> EvidencePack:
    """Extract structured facts from search results.

    Args:
        query: User's query
        search_results: Retrieved and reranked results

    Returns:
        EvidencePack with facts and source mapping
    """
    client = AsyncOpenAI()

    # Prepare sources
    sources_text = ""
    for i, result in enumerate(search_results):
        sources_text += f"\n\n[Source {i+1}: {result.source_id}]\n"
        sources_text += f"Title: {result.title}\n"
        sources_text += f"Content: {result.content[:1000]}\n"

    prompt = f"""Extract factual claims from these sources to answer the query.

Query: {query}

Sources:
{sources_text}

For each relevant fact:
1. Paraphrase the fact clearly
2. Include a direct quote (max 20 words)
3. Reference the source ID
4. Assign confidence (0.0-1.0)

Output JSON array:
[
  {{
    "fact": "Clear paraphrased fact",
    "quote": "Direct quote from source",
    "source_id": "source_id_from_above",
    "confidence": 0.95
  }}
]

Extract 5-10 high-quality facts."""

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        response_format={"type": "json_object"}
    )

    # Parse facts
    content = response.choices[0].message.content
    facts_data = json.loads(content)

    # Create EvidenceFact objects
    facts = []
    sources_dict = {r.source_id: r for r in search_results}

    for fact_data in facts_data.get('facts', []):
        source_result = sources_dict.get(fact_data['source_id'])
        if source_result:
            facts.append(EvidenceFact(
                fact=fact_data['fact'],
                quote=fact_data['quote'],
                source_id=fact_data['source_id'],
                url=source_result.url,
                title=source_result.title,
                author=source_result.author,
                published_at=source_result.published_at,
                confidence=fact_data.get('confidence', 0.8)
            ))

    return EvidencePack(facts=facts, sources=sources_dict)

# GOOD: Validate extracted facts
def validate_fact(fact: EvidenceFact, source_content: str) -> bool:
    """Validate that fact is supported by source."""
    # Check quote appears in source
    if fact.quote not in source_content:
        return False

    # Check quote length
    if len(fact.quote.split()) > 20:
        return False

    # Check fact is not trivial
    if len(fact.fact.split()) < 5:
        return False

    return True
```

### Working on Writer (`writer.py`)
**What you can do in parallel:**
- Improve tweet generation prompts for different styles
- Add thread generation strategies
- Add character counting and truncation
- Add citation placement optimization

**What requires coordination:**
- Changing Tweet format (impacts fact-checker)
- Modifying citation syntax (affects parsing)
- Changing LLM model (impacts style/quality)

**Testing requirements:**
```bash
# Test tweet generation
uv run pytest tests/generation/test_writer.py

# Test citation format
uv run pytest tests/generation/test_writer.py::test_citation_format

# Test character limits
uv run pytest tests/generation/test_writer.py::test_character_limits
```

**Code style:**
```python
# GOOD: Generate tweets with structured citations
from openai import AsyncOpenAI
from src.core.models import Tweet, Citation, EvidencePack
import json

async def generate_tweets(
    query: str,
    evidence: EvidencePack,
    max_variants: int = 3,
    max_thread_tweets: int = 6
) -> tuple[list[Tweet], list[Tweet]]:
    """Generate tweet variants and thread with citations.

    Args:
        query: User's query/topic
        evidence: Evidence pack with facts and sources
        max_variants: Number of single tweet variants
        max_thread_tweets: Max tweets in thread

    Returns:
        Tuple of (variants, thread)
    """
    client = AsyncOpenAI()

    # Prepare evidence
    facts_text = "\n".join([
        f"[{i+1}] {fact.fact} (Source: {fact.source_id})"
        for i, fact in enumerate(evidence.facts)
    ])

    prompt = f"""Write evidence-based tweets about: {query}

Available Evidence:
{facts_text}

Requirements:
1. Generate {max_variants} single tweet variants (each <280 chars)
2. Generate 1 thread (up to {max_thread_tweets} tweets)
3. Use inline citations [1], [2] for each claim
4. Every fact MUST have a citation
5. Be concise and engaging

Output JSON:
{{
  "variants": [
    {{"text": "Tweet text [1][2]", "citations": [{{"n": 1, "source_id": "..."}}, ...]}},
    ...
  ],
  "thread": [
    {{"text": "Thread tweet 1 [1]", "citations": [...]}},
    ...
  ]
}}"""

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        response_format={"type": "json_object"}
    )

    # Parse response
    content = response.choices[0].message.content
    data = json.loads(content)

    # Create Tweet objects
    variants = [
        Tweet(
            text=v['text'],
            citations=[Citation(**c) for c in v['citations']]
        )
        for v in data.get('variants', [])
    ]

    thread = [
        Tweet(
            text=t['text'],
            citations=[Citation(**c) for c in t['citations']]
        )
        for t in data.get('thread', [])
    ]

    return variants, thread

# GOOD: Validate tweet format
def validate_tweet(tweet: Tweet) -> bool:
    """Validate tweet meets requirements."""
    # Check character limit
    if len(tweet.text) > 280:
        return False

    # Check all citations are in text
    cited_numbers = {c.n for c in tweet.citations}
    for n in cited_numbers:
        if f'[{n}]' not in tweet.text:
            return False

    # Check no orphan citations in text
    import re
    text_citations = set(int(m.group(1)) for m in re.finditer(r'\[(\d+)\]', tweet.text))
    if text_citations != cited_numbers:
        return False

    return True
```

### Working on Fact-Checking (`factcheck.py`)
**What you can do in parallel:**
- Improve verification prompts
- Add citation completeness checking
- Add hallucination detection
- Add fact confidence rescoring

**What requires coordination:**
- Changing verification criteria (affects accuracy)
- Modifying Tweet correction logic (may change output)
- Changing LLM model (impacts reliability)

**Testing requirements:**
```bash
# Test fact-checking
uv run pytest tests/generation/test_factcheck.py

# Test citation verification
uv run pytest tests/generation/test_factcheck.py::test_citation_check

# Test hallucination detection
uv run pytest tests/generation/test_factcheck.py::test_hallucination_detection
```

**Code style:**
```python
# GOOD: Comprehensive fact-checking
from openai import AsyncOpenAI
from src.core.models import Tweet, EvidencePack
import json

async def fact_check_tweets(
    tweets: list[Tweet],
    evidence: EvidencePack
) -> list[Tweet]:
    """Verify and fix citations in tweets.

    Args:
        tweets: Generated tweets with citations
        evidence: Evidence pack used for generation

    Returns:
        Verified tweets with corrected citations
    """
    client = AsyncOpenAI()
    verified_tweets = []

    for tweet in tweets:
        # Get cited sources
        cited_sources = {c.source_id: evidence.sources[c.source_id] for c in tweet.citations}

        sources_text = "\n".join([
            f"[{c.n}] Source: {s.title}\nContent: {s.content[:500]}"
            for c in tweet.citations
            for s in [cited_sources[c.source_id]]
        ])

        prompt = f"""Verify this tweet's factual accuracy against its cited sources.

Tweet: {tweet.text}

Cited Sources:
{sources_text}

Check:
1. Every claim is supported by cited sources
2. No hallucinated facts
3. Citations are correctly placed

Output JSON:
{{
  "verified": true/false,
  "issues": ["list of issues if any"],
  "corrected_text": "corrected tweet if needed (keep citations)"
}}"""

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)

        if result['verified']:
            verified_tweets.append(tweet)
        elif result.get('corrected_text'):
            # Use corrected version
            corrected_tweet = Tweet(
                text=result['corrected_text'],
                citations=tweet.citations
            )
            verified_tweets.append(corrected_tweet)
        else:
            # Drop unverifiable tweet
            continue

    return verified_tweets

# GOOD: Add citation completeness check
def check_citation_completeness(tweet: Tweet) -> bool:
    """Verify all citations are present and valid."""
    import re

    # Extract citation numbers from text
    text_citations = set(int(m.group(1)) for m in re.finditer(r'\[(\d+)\]', tweet.text))

    # Get citation numbers from Citation objects
    object_citations = {c.n for c in tweet.citations}

    # Must match exactly
    return text_citations == object_citations and len(text_citations) > 0
```

## Interface Contract

### Embeddings
```python
from src.generation.embeddings import embed_text, embed_batch

# Single text
embedding = embed_text("AI safety research")
# Returns: list[float] of length 1024

# Batch
embeddings = embed_batch(["text 1", "text 2", "text 3"])
# Returns: list[list[float]]
```

### Gap Analysis
```python
from src.generation.gap_analysis import analyze_gaps

# Identify knowledge gaps
gap_queries = await analyze_gaps(
    query="AI safety in 2025",
    internal_results=internal_results
)
# Returns: list[str] of targeted search queries
```

### Evidence Assembly
```python
from src.generation.evidence import create_evidence_pack

# Extract facts from results
evidence = await create_evidence_pack(
    query="AI safety research",
    search_results=final_results
)
# Returns: EvidencePack with facts and sources
```

### Tweet Generation
```python
from src.generation.writer import generate_tweets

# Generate tweets with citations
variants, thread = await generate_tweets(
    query="AI safety research",
    evidence=evidence,
    max_variants=3,
    max_thread_tweets=6
)
# Returns: (list[Tweet], list[Tweet])
```

### Fact-Checking
```python
from src.generation.factcheck import fact_check_tweets

# Verify and correct tweets
verified = await fact_check_tweets(tweets, evidence)
# Returns: list[Tweet] with verified citations
```

## Pipeline Flow
```
Query → Gap Analysis → Gap Queries
  ↓
Search Results → Evidence Pack → Tweet Generation → Fact-Check → Final Tweets
                      ↓
                  Embeddings (for deduplication)
```

## Dependencies
**External:**
- `openai`: GPT models for generation and verification
- `sentence-transformers`: BGE-M3 embeddings and reranking
- `torch`: ML framework for sentence-transformers

**Internal:**
- `src.core.models`: All data models
- `src.core.config`: OpenAI API key, model settings

## Performance Considerations
- **Embeddings**: ~100ms for batch of 10 texts
- **Gap Analysis**: ~1-2s per query (LLM call)
- **Evidence Pack**: ~2-3s (LLM call)
- **Tweet Generation**: ~3-5s (LLM call)
- **Fact-Checking**: ~1-2s per tweet (LLM call)
- **Total Pipeline**: ~10-15s for full generation

### Optimization Tips
```python
# GOOD: Batch embeddings
embeddings = embed_batch(texts)  # Not: [embed_text(t) for t in texts]

# GOOD: Parallel fact-checking
import asyncio
verified = await asyncio.gather(*[
    fact_check_tweets([tweet], evidence)
    for tweet in tweets
])

# GOOD: Cache model instances
@lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer('BAAI/bge-m3')

# GOOD: Use smaller models for faster ops
async def quick_gap_analysis():
    # Use gpt-4o-mini for gap analysis (faster, cheaper)
    response = await client.chat.completions.create(
        model="gpt-4o-mini",  # Not gpt-4
        ...
    )
```

## Common Pitfalls
- **DON'T** call LLM APIs synchronously (use async)
- **DON'T** generate embeddings one at a time (use batches)
- **DON'T** skip fact-checking (hallucinations are common)
- **DON'T** forget to handle API errors (rate limits, timeouts)
- **DO** use structured output (JSON mode) for parsing
- **DO** validate all LLM outputs before returning
- **DO** cache model instances (expensive to load)

## Testing Checklist
- [ ] Embeddings are normalized (cosine similarity works correctly)
- [ ] Gap queries are specific and actionable
- [ ] Evidence facts have supporting quotes
- [ ] Tweets are <280 characters
- [ ] All citations are present in text and Citation objects
- [ ] Fact-checking catches hallucinations
- [ ] Performance: <15s for full generation pipeline

## Local Development Setup
```bash
# Set OpenAI API key
export OPENAI_API_KEY="your_key"

# Install dependencies
uv sync

# Download embedding model (first run only)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

# Run tests
uv run pytest tests/generation/ -v
```

## Contact for Coordination
When modifying generation pipeline:
1. Test with diverse queries (technical, current events, etc.)
2. Measure factuality (>95% post-fact-check accuracy)
3. Monitor LLM costs (OpenAI API usage)
4. Coordinate with retrieval module on evidence format changes
