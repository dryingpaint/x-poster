# Agent Tweeter - MVP

An AI-powered system that generates evidence-based tweets with inline citations from internal documents and web sources.

## Architecture

- **Orchestrator**: Deterministic async pipeline with three main tools
- **Retrieval**: Hybrid search (internal: Supabase+pgvector, web: EXA/Serper)
- **Reranking**: BGE-reranker-large cross-encoder
- **Generation**: OpenAI for evidence assembly, writing, and fact-checking
- **Storage**: Supabase Postgres with pgvector for multimodal data

## Features

- 🔍 **Hybrid Retrieval**: Combines internal documents (PDF, docs, images) with live web search
- 📊 **Evidence-Based**: Every claim is grounded in retrieved sources
- 🔗 **Inline Citations**: Numeric citations [1][2] mapped to real URLs
- ✅ **Fact-Checking**: LLM-based verification against evidence pack
- 🧵 **Smart Threading**: Auto-splits into threads when needed
- 🌍 **Multimodal**: Handles PDFs, images (with OCR), audio transcripts, code

## Project Structure

```
agent-tweeter/
├── src/
│   ├── core/           # Configuration and data models
│   ├── db/             # Database client and migrations
│   ├── ingestion/      # Document processing (PDF, OCR, chunking)
│   ├── retrieval/      # Internal & web retrieval, reranking
│   ├── generation/     # Embeddings, evidence, writer, fact-check
│   ├── orchestrator/   # Main pipeline
│   └── utils/          # Caching, web utils, text processing
├── cli.py              # CLI entry point
├── pyproject.toml      # Project config & dependencies
└── tests/              # Unit tests
```

## Setup

### 1. Install Dependencies

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync

# Or install with dev dependencies
uv sync --extra dev
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Initialize Database

```bash
# Run migrations in Supabase SQL editor or via psql
psql $DATABASE_URL -f src/db/migrations/001_initial_schema.sql
```

### 4. Ingest Documents

```bash
uv run python cli.py ingest --source files/ --kind pdf
```

## Usage

### Generate Tweets (CLI)

```bash
uv run python cli.py generate "State of small LMs in 2025" --max-variants 3

# Or use the CLI entry point
uv run agent-tweeter generate "State of small LMs in 2025" --max-variants 3
```

### API Response Format

```json
{
  "variants": [
    {
      "text": "Hook + fact … [1][2]",
      "citations": [{"n":1,"source_id":"src_web_3"},{"n":2,"source_id":"src_int_5"}]
    }
  ],
  "thread": [
    {"text":"Pt1 … [1]"},
    {"text":"Pt2 … [2]"},
    {"text":"Sources: [1] https://short.ly/a | [2] https://short.ly/b"}
  ],
  "sources": [...]
}
```

## Performance

- **Target Latency**: ~12s p50
- **Factuality**: ≥95% post-fact-check
- **Retrieval**: Precision@8 optimized

## Tech Stack

- **Embeddings**: BGE-M3 (1024-dim, multilingual)
- **Reranker**: BAAI/bge-reranker-large
- **LLM**: OpenAI GPT-4/3.5-turbo
- **Vector DB**: Supabase + pgvector
- **Web Search**: EXA (primary), Serper (fallback)
- **Web Crawl**: Firecrawl
- **PDF**: PyMuPDF + OCRmyPDF
- **Caching**: Redis (optional)

## Development

```bash
# Run tests
uv run pytest tests/

# Type checking
uv run mypy src/

# Linting
uv run ruff check src/

# Format code
uv run black src/
```

## License

MIT
