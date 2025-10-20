"""Inspect internal database retrieval results.

Usage:
    uv run python examples/inspect_retrieval.py "your query here"
    uv run python examples/inspect_retrieval.py "degrowth"
    uv run python examples/inspect_retrieval.py "climate change" --top-k 10
"""

import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.db.operations import search_internal
from src.generation.embeddings import embed_text

console = Console()


async def inspect_internal_search(query: str, top_k: int = 5):
    """Search internal database and display detailed results."""

    console.print(f"\n[bold blue]🔍 Searching for:[/bold blue] {query}\n")

    # Step 1: Embed the query
    console.print("[yellow]Step 1: Generating query embedding...[/yellow]")
    query_embedding = embed_text(query)
    console.print(f"✅ Embedding dimension: {len(query_embedding)}\n")

    # Step 2: Search internal database
    console.print(f"[yellow]Step 2: Searching internal database (top {top_k} results)...[/yellow]")
    results = await search_internal(query=query, query_embedding=query_embedding, top_k=top_k)

    console.print(f"✅ Found {len(results)} results\n")

    if not results:
        console.print("[red]No results found in database.[/red]")
        console.print("\n[yellow]💡 Tip: Make sure you've ingested documents:[/yellow]")
        console.print("   uv run python cli.py ingest --source files/ --kind pdf")
        return

    # Display results table
    console.print("[bold green]📊 Results Overview:[/bold green]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Type", width=10)
    table.add_column("Title", width=40)
    table.add_column("Content Length", justify="right", width=15)

    for i, result in enumerate(results, 1):
        table.add_row(
            str(i),
            f"{result.score:.4f}",
            result.source_type,
            (result.title or "Untitled")[:40],
            f"{len(result.content):,} chars",
        )

    console.print(table)

    # Display detailed content for each result
    console.print("\n[bold green]📄 Detailed Content:[/bold green]\n")

    for i, result in enumerate(results, 1):
        console.print(f"[bold cyan]Result #{i}[/bold cyan]")
        console.print(f"[dim]Score: {result.score:.4f} | Type: {result.source_type}[/dim]")

        # Metadata
        meta_info = []
        if result.title:
            meta_info.append(f"Title: {result.title}")
        if result.source_uri:
            meta_info.append(f"Source: {result.source_uri}")
        if result.url:
            meta_info.append(f"URL: {result.url}")
        if result.author:
            meta_info.append(f"Author: {result.author}")
        if result.published_at:
            meta_info.append(f"Published: {result.published_at}")

        if meta_info:
            console.print("[dim]" + " | ".join(meta_info) + "[/dim]")

        # Content preview
        content_preview = result.content[:500] if len(result.content) > 500 else result.content
        if len(result.content) > 500:
            content_preview += f"\n\n[dim]... ({len(result.content) - 500:,} more characters)[/dim]"

        console.print(
            Panel(
                content_preview,
                title=f"Content Preview ({len(result.content):,} chars total)",
                border_style="blue",
                expand=False,
            )
        )
        console.print()

    # Additional metadata summary
    console.print("[bold yellow]📈 Summary Statistics:[/bold yellow]")
    console.print(f"  • Total results: {len(results)}")
    console.print(f"  • Average score: {sum(r.score for r in results) / len(results):.4f}")
    console.print(f"  • Total content: {sum(len(r.content) for r in results):,} characters")
    console.print(
        f"  • Avg content length: {sum(len(r.content) for r in results) // len(results):,} chars/result"
    )

    # Show unique sources
    unique_sources = set()
    for r in results:
        if r.source_uri:
            unique_sources.add(r.source_uri)
        elif r.url:
            unique_sources.add(r.url)

    if unique_sources:
        console.print(f"  • Unique sources: {len(unique_sources)}")
        console.print("\n[bold yellow]📚 Sources:[/bold yellow]")
        for source in sorted(unique_sources)[:10]:
            console.print(f"    • {source}")
        if len(unique_sources) > 10:
            console.print(f"    ... and {len(unique_sources) - 10} more")


async def compare_hybrid_vs_vector(query: str, top_k: int = 5):
    """Compare hybrid search vs pure vector search."""
    from src.db.client import get_db_client

    console.print(f"\n[bold blue]🔬 Comparing Search Methods:[/bold blue] {query}\n")

    # Generate embedding
    query_embedding = embed_text(query)

    # Hybrid search (default)
    console.print("[yellow]Running hybrid search (vector + FTS)...[/yellow]")
    hybrid_results = await search_internal(query, query_embedding, top_k=top_k)

    # Pure vector search
    console.print("[yellow]Running pure vector search...[/yellow]")
    db = get_db_client()
    vector_results = (
        db.get_client()
        .rpc("search_chunks_vector", {"query_embedding": query_embedding, "match_count": top_k})
        .execute()
    )

    # Pure FTS search
    console.print("[yellow]Running pure full-text search...[/yellow]")
    fts_results = (
        db.get_client()
        .rpc("search_chunks_fts", {"search_query": query, "match_count": top_k})
        .execute()
    )

    console.print("\n[bold green]📊 Comparison Results:[/bold green]\n")

    # Create comparison table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Method", width=20)
    table.add_column("Results", justify="right", width=10)
    table.add_column("Avg Score", justify="right", width=12)
    table.add_column("Top Title", width=40)

    # Hybrid
    table.add_row(
        "Hybrid (Vector + FTS)",
        str(len(hybrid_results)),
        f"{sum(r.score for r in hybrid_results) / len(hybrid_results):.4f}"
        if hybrid_results
        else "N/A",
        (hybrid_results[0].title or "Untitled")[:40] if hybrid_results else "N/A",
    )

    # Vector
    vector_count = len(vector_results.data) if vector_results.data else 0
    vector_avg = (
        sum(r.get("similarity", 0) for r in vector_results.data) / vector_count
        if vector_count > 0
        else 0
    )
    table.add_row(
        "Pure Vector",
        str(vector_count),
        f"{vector_avg:.4f}",
        vector_results.data[0].get("title", "Untitled")[:40] if vector_count > 0 else "N/A",
    )

    # FTS
    fts_count = len(fts_results.data) if fts_results.data else 0
    fts_avg = sum(r.get("rank", 0) for r in fts_results.data) / fts_count if fts_count > 0 else 0
    table.add_row(
        "Pure Full-Text",
        str(fts_count),
        f"{fts_avg:.4f}",
        fts_results.data[0].get("title", "Untitled")[:40] if fts_count > 0 else "N/A",
    )

    console.print(table)
    console.print("\n[dim]💡 Hybrid search combines both methods for better coverage[/dim]")


def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect internal database retrieval results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inspect_retrieval.py "degrowth"
  python inspect_retrieval.py "climate change" --top-k 10
  python inspect_retrieval.py "AI safety" --compare
        """,
    )
    parser.add_argument("query", help="Search query")
    parser.add_argument(
        "--top-k", "-k", type=int, default=5, help="Number of results to retrieve (default: 5)"
    )
    parser.add_argument(
        "--compare",
        "-c",
        action="store_true",
        help="Compare hybrid vs vector vs FTS search methods",
    )

    args = parser.parse_args()

    if args.compare:
        asyncio.run(compare_hybrid_vs_vector(args.query, args.top_k))
    else:
        asyncio.run(inspect_internal_search(args.query, args.top_k))


if __name__ == "__main__":
    main()
