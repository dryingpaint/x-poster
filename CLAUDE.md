# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agent Tweeter is an AI-powered system that generates evidence-based tweets with inline citations from internal documents and web sources. The system uses a hybrid retrieval approach combining internal documents (stored in Supabase + pgvector) with live web search.

## Project Management

This project uses **uv** for all dependency management and Python environment handling. Always use uv commands instead of pip/pipx.

## Common Commands

### Development Setup
```bash
# Install dependencies with uv
uv sync

# Install with dev dependencies
uv sync --extra dev
```

### Important Setup Notes

After running the initial database migration (`001_initial_schema.sql`), you must also run the type fix migration:

```bash
# Run this in Supabase SQL editor after the initial migration
psql $DATABASE_URL -f src/db/migrations/002_fix_fts_function.sql
```

This fixes a PostgreSQL type mismatch where `ts_rank` returns `real` but the function expects `double precision`.

### Database Setup
```bash
# Initialize database (run in Supabase SQL editor or via psql)
psql $DATABASE_URL -f src/db/migrations/001_initial_schema.sql
```

### Document Ingestion
```bash
# Ingest PDF documents
uv run python cli.py ingest --source files/ --kind pdf
```

### Generate Tweets
```bash
# Generate tweets via CLI
uv run python cli.py generate "State of small LMs in 2025" --max-variants 3

# Or use the CLI entry point
uv run agent-tweeter generate "State of small LMs in 2025" --max-variants 3
```

### Testing & Quality
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

## Architecture

The system is organized into distinct modules:

- **`src/core/`**: Configuration management (`config.py`) and data models (`models.py`)
- **`src/db/`**: Database client, operations, and migrations
- **`src/ingestion/`**: Document processing (PDF, OCR, chunking)
- **`src/retrieval/`**: Hybrid search (internal + web), reranking, merging
- **`src/generation/`**: Embeddings, evidence assembly, tweet generation

### Key Components

1. **Orchestrator**: Deterministic async pipeline coordinating three main tools
2. **Retrieval**: Uses BGE-M3 embeddings with pgvector for internal docs, EXA/Serper for web search
3. **Reranking**: BGE-reranker-large cross-encoder for result refinement
4. **Generation**: OpenAI models for evidence assembly and tweet generation
5. **Storage**: Supabase Postgres with pgvector for multimodal document storage

## Configuration

Configuration is managed through `src/core/config.py` using pydantic-settings. Key settings include:

- **Supabase**: Database connection and service role key
- **OpenAI**: API key for generation models
- **Web Search**: EXA and Serper API keys (EXA is primary)
- **Embeddings**: BGE-M3 model with 1024 dimensions
- **Performance**: Timeouts, concurrency limits, retrieval parameters

Environment variables are loaded from `.env` file (see `.env.example` for template).

## Data Models

Core data models in `src/core/models.py`:

- **`Item`**: Multimodal documents (PDFs, images, audio, etc.)
- **`ItemChunk`**: Chunked text for retrieval with embeddings
- **`SearchResult`**: Unified format for internal and web results  
- **`EvidenceFact`**: Extracted facts with confidence scores
- **`Tweet`**: Generated tweets with citations
- **`GenerateResponse`**: API response format with variants, threads, and sources

## Key Features

- **Hybrid Retrieval**: Combines internal documents with live web search
- **Evidence-Based**: Every claim is grounded in retrieved sources
- **Inline Citations**: Numeric citations [1][2] mapped to real URLs
- **Fact-Checking**: LLM-based verification against evidence pack
- **Smart Threading**: Auto-splits into threads when content exceeds limits
- **Multimodal**: Handles PDFs, images (with OCR), audio transcripts

## Performance Targets

- **Target Latency**: ~12s p50 for tweet generation
- **Factuality**: ≥95% post-fact-check accuracy
- **Retrieval**: Optimized for Precision@8

## Coding Rules

### Import Management
- **Always keep `__init__.py` files empty**
- **Always use full import paths** (e.g., `from src.core.config import get_config` not `from core.config import get_config`)
- This ensures clarity and prevents import conflicts in the modular architecture

### Project Structure
- Follow the established module organization in `src/`
- Use absolute imports from the project root
- Maintain separation between core, db, ingestion, retrieval, and generation modules

## Known Issues & Fixes Applied

### PDF Ingestion
- **Fixed**: Variable name conflict in `chunker.py` that caused `chunk_text` error during ingestion
- **Fixed**: Large PDF timeouts by implementing smaller batch processing (16 chunks at a time) with progress reporting
- **Performance**: PDFs with 300+ chunks now process successfully but may take 5-10 minutes for embeddings

### Database Functions
- **Fixed**: PostgreSQL type mismatch in `search_chunks_fts` function - `ts_rank` returns `real` but function expected `double precision`
- **Solution**: Added type casting to `double precision` and created migration file `002_fix_fts_function.sql`

### Code Quality
- **Fixed**: All linting issues including unused imports, deprecated configuration, missing `strict=` parameters
- **Updated**: Moved ruff configuration to new `[tool.ruff.lint]` section format

## Tech Stack

- **Python**: 3.11+ with uv for dependency management
- **Embeddings**: BGE-M3 (multilingual, 1024-dim)
- **Reranker**: BAAI/bge-reranker-large
- **LLM**: OpenAI GPT-4/3.5-turbo
- **Vector DB**: Supabase + pgvector
- **Web Search**: EXA (primary), Serper (fallback)
- **PDF Processing**: PyMuPDF + OCRmyPDF + Tesseract
- **Text Processing**: scikit-learn for cosine similarity
- **Caching**: Redis (optional)