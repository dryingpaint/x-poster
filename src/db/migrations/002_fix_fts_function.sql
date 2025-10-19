-- Fix the FTS function type mismatch
-- Run this in Supabase SQL editor

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
  score double precision
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
    ts_rank(c.tsv, plainto_tsquery('english', search_query))::double precision as score
  from item_chunks c
  join items i on c.item_id = i.item_id
  where c.tsv @@ plainto_tsquery('english', search_query)
  order by score desc
  limit match_count;
end;
$$ language plpgsql;