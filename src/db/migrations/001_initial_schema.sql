-- Initial schema for Agent Tweeter
-- Run this in Supabase SQL editor or via psql

-- Enable required extensions
create extension if not exists vector;
create extension if not exists pg_trgm;

-- Internal items (multimodal)
create table if not exists items (
  item_id        uuid primary key default gen_random_uuid(),
  kind           text check (kind in ('pdf','doc','note','slide','image','audio','code','other')),
  title          text,
  source_uri     text not null,      -- storage://bucket/key or external ref
  content_text   text,               -- canonical text (OCR or caption if needed)
  meta           jsonb default '{}', -- {pages, author, tags, ...}
  created_at     timestamptz default now(),
  updated_at     timestamptz default now()
);

-- Index for faster queries
create index if not exists idx_items_kind on items(kind);
create index if not exists idx_items_created_at on items(created_at desc);

-- Chunked text representation for retrieval
create table if not exists item_chunks (
  chunk_id       uuid primary key default gen_random_uuid(),
  item_id        uuid references items(item_id) on delete cascade,
  content        text not null,
  embedding      vector(1024),       -- BGE-M3 embedding dimension
  tsv            tsvector            -- Full-text search vector
);

-- Indexes for hybrid search
create index if not exists idx_chunks_item_id on item_chunks(item_id);
create index if not exists idx_chunks_tsv on item_chunks using gin(tsv);
create index if not exists idx_chunks_embedding on item_chunks using ivfflat (embedding vector_cosine_ops) with (lists = 100);

-- Trigger to auto-update tsvector on insert/update
create or replace function update_chunk_tsv() returns trigger as $$
begin
  new.tsv := to_tsvector('english', coalesce(new.content, ''));
  return new;
end;
$$ language plpgsql;

drop trigger if exists trigger_update_chunk_tsv on item_chunks;
create trigger trigger_update_chunk_tsv
  before insert or update on item_chunks
  for each row execute function update_chunk_tsv();

-- Cached web pages (normalized & deduped)
create table if not exists web_cache (
  url_hash       text primary key,
  url            text not null,
  domain         text not null,
  title          text,
  published_at   timestamptz,
  content        text not null,
  embedding      vector(1024),
  fetched_at     timestamptz default now()
);

-- Indexes for web cache
create index if not exists idx_web_cache_domain on web_cache(domain);
create index if not exists idx_web_cache_fetched_at on web_cache(fetched_at desc);

-- RPC function for full-text search
create or replace function search_chunks_fts(
  search_query text,
  match_count int default 50
)
returns table (
  chunk_id uuid,
  item_id uuid,
  content text,
  title text,
  source_uri text,
  meta jsonb,
  score float
) as $$
begin
  return query
  select
    c.chunk_id,
    c.item_id,
    c.content,
    i.title,
    i.source_uri,
    i.meta,
    ts_rank(c.tsv, plainto_tsquery('english', search_query)) as score
  from item_chunks c
  join items i on c.item_id = i.item_id
  where c.tsv @@ plainto_tsquery('english', search_query)
  order by score desc
  limit match_count;
end;
$$ language plpgsql;

-- RPC function for vector search
create or replace function search_chunks_vector(
  query_embedding vector(1024),
  match_count int default 50
)
returns table (
  chunk_id uuid,
  item_id uuid,
  content text,
  title text,
  source_uri text,
  meta jsonb,
  score float
) as $$
begin
  return query
  select
    c.chunk_id,
    c.item_id,
    c.content,
    i.title,
    i.source_uri,
    i.meta,
    1 - (c.embedding <=> query_embedding) as score
  from item_chunks c
  join items i on c.item_id = i.item_id
  where c.embedding is not null
  order by c.embedding <=> query_embedding
  limit match_count;
end;
$$ language plpgsql;

-- Trigger to auto-update updated_at
create or replace function update_updated_at() returns trigger as $$
begin
  new.updated_at := now();
  return new;
end;
$$ language plpgsql;

drop trigger if exists trigger_update_items_updated_at on items;
create trigger trigger_update_items_updated_at
  before update on items
  for each row execute function update_updated_at();

