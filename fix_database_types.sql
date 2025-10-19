-- Quick fix for the PostgreSQL type mismatch
-- Run this in Supabase Dashboard → SQL Editor

CREATE OR REPLACE FUNCTION search_chunks_fts(
  search_query text,
  match_count int default 50
)
RETURNS TABLE (
  chunk_id uuid,
  item_id uuid,
  content text,
  title text,
  source_uri text,
  meta jsonb,
  score double precision
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    c.chunk_id,
    c.item_id,
    c.content,
    i.title,
    i.source_uri,
    i.meta,
    ts_rank(c.tsv, plainto_tsquery('english', search_query))::double precision as score
  FROM item_chunks c
  JOIN items i ON c.item_id = i.item_id
  WHERE c.tsv @@ plainto_tsquery('english', search_query)
  ORDER BY score DESC
  LIMIT match_count;
END;
$$ LANGUAGE plpgsql;