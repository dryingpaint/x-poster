# Quick Start Guide

Get Agent Tweeter running in minutes.

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Supabase account (free tier works)
- OpenAI API key
- EXA API key (or Serper as fallback)

## 1. Setup Environment

```bash
# Clone or navigate to the repo
cd agent-tweeter

# Install uv if you haven't
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync --extra dev
```

## 2. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```bash
# Required
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
OPENAI_API_KEY=sk-...
EXA_API_KEY=your-exa-key  # Primary web search

# Optional
SERPER_API_KEY=your-serper-key  # Fallback
FIRECRAWL_API_KEY=your-firecrawl-key
REDIS_URL=redis://localhost:6379/0
```

## 3. Initialize Database

### Option A: Supabase CLI (Recommended)

```bash
# Link to your project
uv run python cli.py db link YOUR_PROJECT_REF

# Push the initial migration
uv run python cli.py db push
```

### Option B: Manual SQL

1. Go to your Supabase project → SQL Editor
2. Copy contents of `supabase/migrations/20251019203208_initial_schema.sql`
3. Run the SQL

## 4. Test Connections

```bash
uv run python cli.py test-connection
```

You should see:

- ✅ Supabase connection: OK
- ✅ OpenAI API: OK
- ✅ EXA API: OK

## 5. Ingest Your First Document

You already have some PDFs in `files/`:

```bash
# Ingest a single PDF
uv run python cli.py ingest \
  --source "files/Jason Hickel - Less is More.pdf" \
  --kind pdf

# Or ingest all PDFs in a directory
uv run python cli.py ingest \
  --source files/ \
  --kind pdf
```

This will:

1. Extract text from PDF
2. Apply OCR if needed
3. Chunk text (300-500 tokens)
4. Generate embeddings (BGE-M3)
5. Store in Supabase

## 6. Generate Your First Tweet

```bash
uv run python cli.py generate \
  "What does Jason Hickel say about degrowth and climate change?"
```

### What happens:

1. **Internal Search** 📚

   - Searches your ingested documents
   - Uses hybrid search (full-text + vector)

2. **Gap Analysis** 🔍

   - LLM identifies missing info
   - Generates targeted web queries

3. **Web Search** 🌐

   - Fills gaps with recent data, stats, visuals
   - Uses EXA for clean content

4. **Ranking** 🎯

   - Merges internal + web
   - Boosts internal sources (1.5x)
   - Cross-encoder reranking

5. **Evidence Assembly** 📚

   - LLM extracts facts + quotes

6. **Writing** ✍️

   - Generates tweet variants
   - Adds inline citations [1][2]

7. **Fact-Check** ✅
   - Verifies against evidence
   - Ensures all claims cited

### Output Example

```
📝 Single Tweet Variants:

┌─ Variant 1 ────────────────────────┐
│ Jason Hickel argues that degrowth  │
│ is essential for climate action:   │
│ rich nations must reduce GDP while │
│ maintaining quality of life [1][2] │
│ #ClimateAction #Degrowth           │
└────────────────────────────────────┘
   Citations: [1], [2]

🧵 Thread:

┌─ Tweet 1 ──────────────────────────┐
│ Is endless growth compatible with  │
│ climate stability? According to    │
│ Jason Hickel, absolutely not [1]   │
└────────────────────────────────────┘

┌─ Tweet 2 ──────────────────────────┐
│ Rich nations need "degrowth" -     │
│ reducing GDP while improving lives │
│ through public services, reduced   │
│ working hours, and redistribution  │
│ [1][2]                             │
└────────────────────────────────────┘

┌─ Tweet 3 ──────────────────────────┐
│ Sources:                           │
│ [1] Less is More - J. Hickel       │
│ [2] Degrowth explained - exa.ai/.. │
└────────────────────────────────────┘

📚 Sources (2):
  • Less is More
    file:///path/to/Jason Hickel - Less is More.pdf

  • What is Degrowth?
    https://example.com/degrowth
```

## 7. Save Output to File

```bash
uv run python cli.py generate \
  "Yanis Varoufakis on technofeudalism" \
  --output output.json
```

## Tips

### Adjust Retrieval Parameters

Edit `.env`:

```bash
# Get more internal results
INTERNAL_TOP_K=100

# Limit web results
WEB_TOP_K=20

# Final results after reranking
FINAL_TOP_K=10
```

### Control Generation

```bash
# More variants
uv run python cli.py generate "query" --max-variants 5

# Longer threads
uv run python cli.py generate "query" --max-thread-tweets 10
```

### Debug Mode

The CLI shows progress with emojis. Watch for:

- 📚 Internal retrieval count (should be high)
- 🔍 Gap queries (2-5 targeted searches)
- 🌐 Web results (supplementary)
- Source breakdown: (X internal, Y web)

**Goal**: Most final sources should be internal!

## Common Issues

### No Internal Results

- Check if documents are ingested: Supabase → `items` table
- Verify embeddings exist: Supabase → `item_chunks` table
- Try broader query

### Poor Web Results

- Check EXA_API_KEY is set
- Verify SERPER_API_KEY as fallback
- Check web search quota

### Slow Performance

- Reduce `INTERNAL_TOP_K` and `WEB_TOP_K`
- Reduce `FINAL_TOP_K`
- Enable Redis caching
- Use faster OpenAI model (gpt-3.5-turbo)

## Next Steps

1. **Ingest more documents**: Add your full library
2. **Customize prompts**: Modify LLM prompts in `src/generation/`
3. **Add observability**: Insert logging/tracing
4. **Build API**: Wrap pipeline in FastAPI
5. **Schedule tweets**: Add cron jobs for automation

## Architecture Overview

```
User Query
    ↓
Internal Search (PRIMARY)
    ↓
Gap Analysis (LLM identifies missing info)
    ↓
Targeted Web Search (fill gaps)
    ↓
Merge + Boost Internal (1.5x)
    ↓
Rerank (cross-encoder)
    ↓
Evidence Pack (LLM extracts facts)
    ↓
Writer (LLM generates tweets)
    ↓
Fact-Check (LLM verifies)
    ↓
Output with Citations
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for details.

## Questions?

- Read the full [README.md](README.md)
- Check [ARCHITECTURE.md](ARCHITECTURE.md) for design decisions
- Review code in `src/` for implementation details
