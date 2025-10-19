# Database Module

## Overview
Manages all database interactions with Supabase PostgreSQL + pgvector. Handles connection pooling, query execution, and vector operations.

## Responsibilities
- **Database Client**: Connection management and query execution
- **CRUD Operations**: Type-safe database operations for all entities
- **Vector Search**: pgvector similarity search with hybrid ranking
- **Migrations**: Schema versioning and upgrades

## Key Files
- `client.py`: Database client singleton with connection pooling
- `operations.py`: High-level database operations (search, insert, update)
- `migrations/`: SQL migration files

## Development Guidelines

### Working on Database Client (`client.py`)
**What you can do in parallel:**
- Add connection pooling optimizations
- Add query performance monitoring
- Add connection retry logic
- Add read replica support

**What requires coordination:**
- Changing the client singleton pattern (impacts all database users)
- Modifying connection parameters (may affect performance)
- Changing transaction isolation levels

**Testing requirements:**
```bash
# Run against test database
export DATABASE_URL="postgresql://test:test@localhost:5432/test_db"
uv run pytest tests/db/test_client.py

# Test connection pooling
uv run pytest tests/db/test_client.py::test_connection_pooling
```

**Code style:**
```python
# GOOD: Use context manager for connections
async def query_example():
    client = get_db_client()
    async with client.connection() as conn:
        result = await conn.fetch("SELECT * FROM items")
        return result

# GOOD: Use parameterized queries (prevent SQL injection)
async def safe_query(item_id: str):
    query = "SELECT * FROM items WHERE item_id = $1"
    return await conn.fetchrow(query, item_id)
```

### Working on Operations (`operations.py`)
**What you can do in parallel:**
- Add new CRUD operations (e.g., `delete_item`, `update_chunk`)
- Add new search strategies (e.g., `search_by_date_range`)
- Add batch operations (e.g., `insert_chunks_batch`)
- Add aggregation queries (e.g., `count_items_by_kind`)

**What requires coordination:**
- Changing signatures of existing operations (breaks callers)
- Modifying search ranking algorithms (may affect results quality)
- Changing transaction boundaries (may affect consistency)

**Testing requirements:**
```bash
# Test all operations
uv run pytest tests/db/test_operations.py

# Test vector search quality
uv run pytest tests/db/test_vector_search.py

# Test data integrity
uv run pytest tests/db/test_transactions.py
```

**Code style:**
```python
# GOOD: Type hints for all parameters and returns
async def search_internal(
    query: str,
    query_embedding: list[float],
    top_k: int = 10,
) -> list[SearchResult]:
    """Search internal documents using hybrid vector + text search.

    Args:
        query: Natural language query
        query_embedding: Query vector (1024-dim for BGE-M3)
        top_k: Number of results to return

    Returns:
        List of SearchResult ordered by relevance
    """
    # Implementation
    pass

# GOOD: Use transactions for multi-step operations
async def insert_item_with_chunks(item: Item, chunks: list[ItemChunk]):
    async with client.transaction():
        await insert_item(item)
        await insert_chunks_batch(chunks)
```

### Working on Migrations (`migrations/`)
**What you can do in parallel:**
- Add new tables for new features
- Add indexes for query optimization
- Add new columns (as nullable or with defaults)

**What requires coordination:**
- Modifying existing tables (requires migration strategy)
- Dropping columns or tables (requires deprecation period)
- Changing primary keys or foreign keys

**Migration checklist:**
- [ ] Number migrations sequentially (`001_`, `002_`, etc.)
- [ ] Include both `UP` and `DOWN` migration paths
- [ ] Test on copy of production data
- [ ] Document breaking changes in migration comments
- [ ] Update schema documentation

**Code style:**
```sql
-- GOOD: Use IF NOT EXISTS for idempotency
CREATE TABLE IF NOT EXISTS items (
    item_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    kind TEXT NOT NULL,
    title TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- GOOD: Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_items_kind ON items(kind);
CREATE INDEX IF NOT EXISTS idx_items_created_at ON items(created_at DESC);

-- GOOD: Add column with default for backward compatibility
ALTER TABLE items ADD COLUMN IF NOT EXISTS meta JSONB DEFAULT '{}'::jsonb;
```

## Interface Contract

### Database Client Pattern
```python
from src.db.client import get_db_client

# Get singleton client
client = get_db_client()

# Execute queries with connection context
async with client.connection() as conn:
    result = await conn.fetch("SELECT * FROM items LIMIT 10")
```

### Search Operations
```python
from src.db.operations import search_internal

# Vector + text hybrid search
results = await search_internal(
    query="AI safety research",
    query_embedding=embedding_vector,  # [float] of length 1024
    top_k=10
)
# Returns: list[SearchResult]
```

### CRUD Operations
```python
from src.db.operations import insert_item, insert_chunks

# Insert document
await insert_item(item)

# Insert chunks with embeddings
await insert_chunks(chunks)  # Automatically generates chunk_id if not set
```

## Database Schema

### Core Tables
```sql
-- Items: Multimodal documents
items (
    item_id UUID PRIMARY KEY,
    kind TEXT,  -- 'pdf', 'doc', 'image', etc.
    title TEXT,
    source_uri TEXT,
    content_text TEXT,
    meta JSONB,
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ
)

-- Item chunks: Vector search units
item_chunks (
    chunk_id UUID PRIMARY KEY,
    item_id UUID REFERENCES items,
    content TEXT,
    embedding vector(1024),  -- BGE-M3 embeddings
    tsv tsvector  -- Full-text search
)

-- Web cache: Cached web pages
web_cache (
    url_hash TEXT PRIMARY KEY,
    url TEXT,
    domain TEXT,
    title TEXT,
    content TEXT,
    embedding vector(1024),
    fetched_at TIMESTAMPTZ
)
```

### Key Indexes
```sql
-- Vector similarity search (HNSW for speed)
CREATE INDEX idx_chunks_embedding ON item_chunks
USING hnsw (embedding vector_cosine_ops);

-- Full-text search
CREATE INDEX idx_chunks_tsv ON item_chunks USING gin(tsv);

-- Lookup indexes
CREATE INDEX idx_chunks_item_id ON item_chunks(item_id);
CREATE INDEX idx_items_kind ON items(kind);
```

## Dependencies
**External:**
- `asyncpg`: Async PostgreSQL driver
- `pgvector`: Vector extension for similarity search

**Internal:**
- `src.core.models`: Data models (Item, ItemChunk, SearchResult)
- `src.core.config`: Database connection settings

## Performance Considerations
- **Connection Pooling**: Client maintains pool of 10-20 connections
- **Batch Inserts**: Use `executemany()` for bulk inserts (10x faster)
- **HNSW Index**: Approximate nearest neighbor for sub-100ms vector search
- **Hybrid Search**: Combines vector similarity (0.7 weight) + text search (0.3 weight)
- **Query Timeout**: 30s default, configurable per query

## Common Pitfalls
- **DON'T** use string formatting for SQL (use parameterized queries)
- **DON'T** fetch large result sets without pagination
- **DON'T** forget to close connections (use async context managers)
- **DON'T** mix sync and async database calls
- **DO** use transactions for multi-step operations
- **DO** add indexes for new query patterns
- **DO** test migrations on staging data before production

## Testing Checklist
- [ ] All operations tested against real PostgreSQL (not mocks)
- [ ] Vector search returns relevant results (Precision@10 > 0.8)
- [ ] Batch operations complete in <1s for 100 items
- [ ] Concurrent operations don't cause deadlocks
- [ ] Connection pool doesn't leak connections
- [ ] Migrations apply and rollback cleanly

## Local Development Setup
```bash
# Start local PostgreSQL with pgvector
docker run -d \
  --name postgres-dev \
  -e POSTGRES_PASSWORD=dev \
  -e POSTGRES_DB=agent_tweeter \
  -p 5432:5432 \
  ankane/pgvector

# Apply migrations
export DATABASE_URL="postgresql://postgres:dev@localhost:5432/agent_tweeter"
psql $DATABASE_URL -f src/db/migrations/001_initial_schema.sql

# Run tests
uv run pytest tests/db/
```

## Contact for Coordination
When making schema changes:
1. Discuss migration strategy (zero-downtime?)
2. Review impact on query performance
3. Test with production-scale data
4. Coordinate deployment with service restart
