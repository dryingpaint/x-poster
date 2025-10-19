# Agent Tweeter Architecture

This document describes the architecture and design decisions for the Agent Tweeter system.

## Overview

Agent Tweeter is a deterministic pipeline that generates evidence-based tweets with inline citations. It combines internal document retrieval with web search to assemble evidence, then uses LLMs to write and fact-check tweets.

## System Architecture

```
┌─────────────┐
│   CLI/API   │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────────┐
│       Orchestrator Pipeline          │
│  (Deterministic, Async)              │
│  Internal-First Strategy             │
└──────┬───────────────────────────────┘
       │
       ├─── 1. Embed Query (BGE-M3)
       │
       ├─── 2. Internal Retrieval (PRIMARY)
       │      └─ Hybrid: FTS + Vector (Supabase)
       │
       ├─── 3. Gap Analysis (LLM)
       │      └─ Identify missing: stats, visuals, recent data
       │
       ├─── 4. Targeted Web Search (FILL GAPS)
       │      └─ Multiple specific queries → EXA/Serper
       │
       ├─── 5. Merge & Dedupe & Rerank
       │      ├─ Boost internal sources (1.5x)
       │      └─ Cross-encoder (BGE Reranker)
       │
       ├─── 6. Evidence Pack (LLM)
       │      └─ Extract facts + quotes + sources
       │
       ├─── 7. Writer (LLM)
       │      └─ Generate tweets with [n] citations
       │
       ├─── 8. Fact-Check (LLM)
       │      └─ Verify claims against evidence
       │
       └─── 9. Return Response
            └─ Tweets + Source Map
```

## Core Components

### 1. Data Layer (`src/db/`)

- **Database**: Supabase Postgres + pgvector
- **Schema**:
  - `items` - Multimodal documents (PDF, images, etc.)
  - `item_chunks` - Text chunks with embeddings and tsvector
  - `web_cache` - Cached web pages
- **Operations**: Hybrid search combining full-text and vector similarity

### 2. Ingestion (`src/ingestion/`)

- **PDF Processing**: PyMuPDF for text extraction
- **OCR**: OCRmyPDF for scanned documents
- **Chunking**: Token-based chunking with overlap (300-500 tokens)

### 3. Retrieval (`src/retrieval/`)

#### Internal Retrieval (PRIMARY)

- **Priority**: Internal documents are the main knowledge base
- Full-text search (PostgreSQL tsvector)
- Vector similarity search (pgvector)
- Merge top-k from both approaches
- Results get 1.5x score boost in final ranking

#### Gap Analysis (`src/generation/gap_analysis.py`)

- LLM analyzes internal results to identify gaps
- Generates targeted queries for:
  - Missing statistics or recent data
  - Visual evidence (charts, graphs)
  - Expert opinions or authoritative sources
  - Current news or developments

#### Web Retrieval (GAP-FILLING)

- **Purpose**: Fill specific gaps identified in internal knowledge
- Multiple targeted queries (not just one broad search)
- Primary: EXA (includes clean content)
- Fallback: Serper (Google SERP) + Trafilatura extraction
- Parallel fetching with timeout controls

#### Reranking

- Cross-encoder: BAAI/bge-reranker-large
- Rerank merged candidates (internal + web)
- Internal sources already boosted 1.5x
- Select top-k for evidence assembly

#### Merging & Deduplication

- URL canonicalization
- Cosine similarity (embeddings) > 0.95
- MinHash similarity > 0.9
- Domain diversity: max 2 passages per domain

### 4. Generation (`src/generation/`)

#### Embeddings

- Model: BGE-M3 (1024-dim, multilingual)
- Batch processing for efficiency
- Normalized embeddings

#### Evidence Pack

- LLM extracts facts from top-k passages
- Each fact includes:
  - Paraphrased statement
  - Direct quote (≤20 words)
  - Source attribution
  - Confidence score

#### Writer

- Generates 1-3 single tweet variants
- Auto-generates thread if needed (2-6 tweets)
- Inline citations [1][2]
- Final "Sources:" tweet for threads

#### Fact-Checker

- Sentence-level entailment checking
- Verifies every claim has citation
- Removes unsupported superlatives
- Minimal edits to preserve style

### 5. Orchestrator (`src/orchestrator/`)

- **Deterministic pipeline**: Fixed sequence of steps
- **Async execution**: Parallel retrieval and batch operations
- **Error handling**: Graceful degradation
- **Logging**: Progress tracking throughout pipeline

### 6. Utilities (`src/utils/`)

- **Caching**: Redis for web page caching (24h TTL)
- **Text processing**: Truncation, cleaning, token counting

## Data Flow (Internal-First Strategy)

1. **Input**: User prompt
2. **Query Processing**: Embed prompt with BGE-M3
3. **Internal Retrieval** (PRIMARY):
   - Hybrid search (FTS + vector) on internal docs → top 50
4. **Gap Analysis**:
   - LLM analyzes internal results
   - Identifies missing information (stats, visuals, recent data)
   - Generates 2-5 targeted web search queries
5. **Web Retrieval** (GAP-FILLING):
   - Run targeted queries in parallel
   - Search → Fetch → Extract
   - Focus on filling specific gaps
6. **Consolidation**:
   - Merge: Internal + web candidates
   - Boost: Internal sources get 1.5x score multiplier
   - Embed: Batch embedding for deduplication
   - Dedupe: Remove duplicates and near-duplicates
   - Diversify: Cap passages per domain
   - Rerank: Cross-encoder to top 8 (internal-weighted)
7. **Evidence Assembly**: LLM extracts 5-10 facts with quotes
8. **Generation**: LLM writes tweets with [n] citations
9. **Verification**: LLM fact-checks against evidence
10. **Output**: Tweets + source map

## Configuration

All configuration via environment variables (`.env`):

- API keys (Supabase, OpenAI, EXA, Serper)
- Model selections
- Retrieval parameters (top-k values)
- Generation parameters (temperature, max tweets)
- Performance tuning (timeouts, concurrency)

## Performance Targets

- **Latency**: ~12s p50 for full pipeline
- **Factuality**: ≥95% post-fact-check
- **Retrieval**: Precision@8 optimized
- **Concurrency**: Up to 10 parallel web fetches

## Design Decisions

### Why Internal-First Strategy?

- **Grounding**: Tweets should primarily reflect your curated knowledge
- **Trust**: Internal documents are vetted, web content is supplementary
- **Targeted Web Search**: Instead of equal-weight retrieval, web fills specific gaps
- **Better Attribution**: More citations to internal sources builds authority
- **Cost Efficiency**: Smaller web search scope reduces API costs

### Why Hybrid Search?

Combines strengths of:

- **Full-text**: Good for exact matches, named entities
- **Vector**: Good for semantic similarity, paraphrases

### Why Cross-Encoder Reranking?

- More accurate than bi-encoder similarity
- Worth the latency for final top-k selection
- Significantly improves relevance

### Why Separate Evidence Pack Step?

- Cleaner separation of concerns
- LLM focuses on extraction vs. generation
- Enables better fact-checking
- Reusable evidence across multiple drafts

### Why Fact-Check After Writing?

- Writer focuses on engagement + accuracy
- Fact-checker focuses purely on verification
- Two-stage approach catches more errors
- Uses cheaper model for fact-checking

### Why Gap Analysis Step?

- **Intelligent Web Search**: Ask specific questions instead of broad queries
- **Cost Effective**: Fewer, more targeted web searches
- **Better Results**: Get exactly what's missing (stats, visuals, recent data)
- **Maintains Focus**: Web content complements rather than dilutes internal knowledge

### Why Deterministic Pipeline?

- Predictable, debuggable
- Easy to add observability
- Simple to test and iterate
- No complex agent loops

## Extension Points

Future enhancements can add:

1. **Scheduling**: Cron jobs for automated posting
2. **API Server**: FastAPI wrapper around pipeline
3. **Tone Controls**: Persona/style parameters
4. **Safety Guardrails**: Content filtering
5. **Tracing**: OpenTelemetry instrumentation
6. **A/B Testing**: Multiple generation strategies
7. **User Feedback**: Rating system for tweets
8. **Multi-language**: BGE-M3 already supports it

## Tech Stack Rationale

- **BGE-M3**: Best open-source multilingual embedder
- **BGE Reranker**: State-of-the-art cross-encoder
- **Supabase**: Managed Postgres + vector, easy setup
- **EXA**: Clean web content without scraping headaches
- **Trafilatura**: Best open-source content extraction
- **PyMuPDF**: Fast, accurate PDF text extraction
- **OpenAI**: Reliable, high-quality LLMs
- **Pydantic**: Type safety and validation
- **Click + Rich**: Great CLI UX

## Testing Strategy

- **Unit tests**: Individual components
- **Integration tests**: Pipeline with mocked externals
- **E2E tests**: Full pipeline with test database
- **Evaluation**: Retrieval precision, factuality, citation accuracy
