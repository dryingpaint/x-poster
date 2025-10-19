"""Main deterministic pipeline for tweet generation."""

import asyncio
from typing import Any

from src.core.config import get_config
from src.core.models import GenerateRequest, GenerateResponse, Source
from src.db.operations import search_internal
from src.generation.embeddings import embed_text, embed_batch
from src.generation.evidence import create_evidence_pack
from src.generation.factcheck import fact_check_tweets
from src.generation.writer import generate_tweets
from src.retrieval.merger import merge_and_dedupe_results
from src.retrieval.reranker import rerank_results
from src.retrieval.web_search import fetch_and_extract, search_web


async def run_generation_pipeline(request: GenerateRequest) -> GenerateResponse:
    """
    Run the full deterministic pipeline for tweet generation.

    Flow:
    1. Embed query
    2. Parallel retrieval: internal (hybrid) + web (search→fetch→extract→embed)
    3. Merge→dedupe→rerank to final k=8
    4. Evidence Pack (LLM)
    5. Writer (LLM) → drafts with [n] markers
    6. Fact-check (LLM) → guarantee citations present
    7. Return drafts + source map

    Args:
        request: GenerateRequest with prompt and parameters

    Returns:
        GenerateResponse with variants, thread, and sources
    """
    config = get_config()

    # Step 1: Embed query
    print(f"📝 Query: {request.prompt}")
    print("🔢 Embedding query...")
    query_embedding = embed_text(request.prompt)

    # Step 2: Parallel retrieval
    print("🔍 Retrieving from internal and web sources...")
    internal_task = search_internal(
        query=request.prompt,
        query_embedding=query_embedding,
        top_k=config.internal_top_k,
    )
    web_task = search_web(query=request.prompt, top_k=config.web_top_k)

    internal_results, web_results = await asyncio.gather(internal_task, web_task)

    print(f"   Found {len(internal_results)} internal results")
    print(f"   Found {len(web_results)} web results")

    # Fetch full content for web results if needed
    if web_results:
        print("🌐 Fetching web content...")
        web_results = await fetch_and_extract(web_results)
        print(f"   Extracted content from {len(web_results)} pages")

    # Step 3: Embed all results for deduplication
    all_results = internal_results + web_results
    if all_results:
        print("🔢 Embedding results...")
        result_texts = [r.content for r in all_results]
        all_embeddings = embed_batch(result_texts)

        internal_embeddings = all_embeddings[: len(internal_results)]
        web_embeddings = all_embeddings[len(internal_results) :]
    else:
        internal_embeddings = []
        web_embeddings = []

    # Step 3b: Merge and dedupe
    print("🔄 Merging and deduplicating...")
    merged_results = merge_and_dedupe_results(
        internal_results=internal_results,
        web_results=web_results,
        internal_embeddings=internal_embeddings,
        web_embeddings=web_embeddings,
        final_k=config.rerank_k,  # Get more for reranking
    )

    # Step 3c: Rerank
    print(f"🎯 Reranking {len(merged_results)} results...")
    final_results = rerank_results(
        query=request.prompt, results=merged_results, top_k=config.final_top_k
    )

    print(f"   Selected top {len(final_results)} results")

    if not final_results:
        print("⚠️  No results found. Returning empty response.")
        return GenerateResponse(variants=[], thread=[], sources=[])

    # Step 4: Create evidence pack
    print("📚 Creating evidence pack...")
    evidence = await create_evidence_pack(query=request.prompt, search_results=final_results)
    print(f"   Extracted {len(evidence.facts)} facts")

    if not evidence.facts:
        print("⚠️  No evidence extracted. Returning empty response.")
        return GenerateResponse(variants=[], thread=[], sources=[])

    # Step 5: Generate tweets
    print("✍️  Generating tweets...")
    variants, thread = await generate_tweets(
        query=request.prompt,
        evidence=evidence,
        max_variants=request.max_variants,
        max_thread_tweets=request.max_thread_tweets,
    )

    print(f"   Generated {len(variants)} variants and {len(thread)} thread tweets")

    # Step 6: Fact-check
    print("✅ Fact-checking...")
    if variants:
        variants = await fact_check_tweets(variants, evidence)
    if thread:
        thread = await fact_check_tweets(thread, evidence)

    # Step 7: Prepare response with sources
    print("📦 Preparing response...")
    sources = []
    for source_id, result in evidence.sources.items():
        sources.append(
            Source(
                source_id=source_id,
                title=result.title,
                url=result.url,
                source_uri=result.source_uri,
                author=result.author,
                published_at=result.published_at,
                meta=result.meta,
            )
        )

    response = GenerateResponse(variants=variants, thread=thread, sources=sources)

    print("✨ Pipeline complete!")
    return response

