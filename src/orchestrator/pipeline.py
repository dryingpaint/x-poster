"""Main deterministic pipeline for tweet generation."""

import asyncio

from src.core.config import get_config
from src.core.models import GenerateRequest, GenerateResponse, Source
from src.db.operations import search_internal, get_chunk_embeddings
from src.generation.embeddings import embed_batch, embed_text
from src.generation.evidence import create_evidence_pack
from src.generation.factcheck import fact_check_tweets
from src.generation.gap_analysis import analyze_gaps
from src.generation.writer import generate_tweets
from src.retrieval.merger import merge_and_dedupe_results
from src.retrieval.reranker import rerank_results
from src.retrieval.web_search import fetch_and_extract, search_web


async def run_generation_pipeline(request: GenerateRequest) -> GenerateResponse:
    """
    Run the full deterministic pipeline for tweet generation.

    Flow (Internal-First Strategy):
    1. Embed query
    2. Retrieve from internal dataset (hybrid search)
    3. Analyze gaps in internal knowledge
    4. Targeted web search to fill gaps (stats, visuals, recent data)
    5. Merge→dedupe→rerank to final k=8
    6. Evidence Pack (LLM)
    7. Writer (LLM) → drafts with [n] markers
    8. Fact-check (LLM) → guarantee citations present
    9. Return drafts + source map

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

    # Step 2: Internal retrieval FIRST (this is our primary knowledge base)
    print("📚 Retrieving from internal dataset...")
    internal_results = await search_internal(
        query=request.prompt,
        query_embedding=query_embedding,
        top_k=config.internal_top_k,
    )
    print(f"   Found {len(internal_results)} internal results")

    # Step 3: Analyze gaps in internal knowledge
    print("🔍 Analyzing knowledge gaps...")
    gap_queries = await analyze_gaps(request.prompt, internal_results)

    # Step 4: Targeted web search to fill gaps
    web_results = []
    if gap_queries:
        print(f"🌐 Running {len(gap_queries)} targeted web searches...")

        # Run all gap-filling queries in parallel
        web_tasks = [
            search_web(query=gap_query, top_k=config.web_top_k // len(gap_queries))
            for gap_query in gap_queries
        ]

        web_results_list = await asyncio.gather(*web_tasks)

        # Flatten results from all queries
        for results in web_results_list:
            web_results.extend(results)

        print(f"   Found {len(web_results)} web results across all gap queries")

        # Fetch full content for web results if needed
        if web_results:
            print("🌐 Fetching web content...")
            web_results = await fetch_and_extract(web_results)
            print(f"   Extracted content from {len(web_results)} pages")

    # Step 5: Embed all results for deduplication
    all_results = internal_results + web_results
    if all_results:
        print("🔢 Embedding results...")

        # Fetch stored embeddings for internal results by chunk_id
        internal_chunk_ids = [r.meta.get("chunk_id") for r in internal_results]
        internal_chunk_ids = [cid for cid in internal_chunk_ids if isinstance(cid, str)]
        internal_embeddings = await get_chunk_embeddings(internal_chunk_ids)

        # Compute embeddings only for web results
        if web_results:
            web_texts = [r.content for r in web_results]
            web_embeddings = embed_batch(web_texts)
        else:
            web_embeddings = []
    else:
        internal_embeddings = []
        web_embeddings = []

    # Step 6: Merge and dedupe (prioritizing internal results)
    print("🔄 Merging and deduplicating (prioritizing internal sources)...")
    merged_results = merge_and_dedupe_results(
        internal_results=internal_results,
        web_results=web_results,
        internal_embeddings=internal_embeddings,
        web_embeddings=web_embeddings,
        final_k=config.rerank_k,  # Get more for reranking
    )

    # Step 7: Rerank (using original query for relevance)
    print(f"🎯 Reranking {len(merged_results)} results...")
    final_results = rerank_results(
        query=request.prompt, results=merged_results, top_k=config.final_top_k
    )

    print(f"   Selected top {len(final_results)} results")

    # Show source breakdown
    internal_count = sum(1 for r in final_results if r.source_type == "internal")
    web_count = sum(1 for r in final_results if r.source_type == "web")
    print(f"   ({internal_count} internal, {web_count} web)")

    if not final_results:
        print("⚠️  No results found. Returning empty response.")
        return GenerateResponse(variants=[], thread=[], sources=[])

    # Step 8: Create evidence pack
    print("📚 Creating evidence pack...")
    evidence = await create_evidence_pack(query=request.prompt, search_results=final_results)
    print(f"   Extracted {len(evidence.facts)} facts")

    if not evidence.facts:
        print("⚠️  No evidence extracted. Returning empty response.")
        return GenerateResponse(variants=[], thread=[], sources=[])

    # Step 9: Generate tweets
    print("✍️  Generating tweets...")
    variants, thread = await generate_tweets(
        query=request.prompt,
        evidence=evidence,
        max_variants=request.max_variants,
        max_thread_tweets=request.max_thread_tweets,
    )

    print(f"   Generated {len(variants)} variants and {len(thread)} thread tweets")

    # Step 10: Fact-check
    print("✅ Fact-checking...")
    if variants:
        variants = await fact_check_tweets(variants, evidence)
    if thread:
        thread = await fact_check_tweets(thread, evidence)

    # Step 11: Prepare response with sources
    print("📦 Preparing response with source attribution...")
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
