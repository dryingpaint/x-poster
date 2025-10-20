"""Inspect web search functionality with EXA and LLM filtering.

Test the complete web search node to see EXA results + LLM content filtering.

Usage:
    # Default: Full node test with filtering (recommended)
    uv run python examples/inspect_websearch.py "climate change" "climate policy 2024" "emissions data"

    # Specify custom query and gap queries
    uv run python examples/inspect_websearch.py "degrowth" "degrowth 2024 stats" "Jason Hickel latest"

    # Basic search only (no filtering)
    uv run python examples/inspect_websearch.py "climate change" --basic
"""

import asyncio
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.core.config import get_config
from src.retrieval.content_filter import filter_and_download
from src.retrieval.web_search import fetch_and_extract, search_web

console = Console()


async def inspect_web_search(query: str, top_k: int = 5):
    """Test web search with a single query using EXA."""

    console.print("\n[bold blue]🌐 Web Search Test[/bold blue]\n")
    console.print(f"[yellow]Query:[/yellow] {query}")
    console.print(f"[yellow]Top K:[/yellow] {top_k}")
    console.print("[yellow]Provider:[/yellow] EXA\n")

    # Step 1: Search web
    console.print("[cyan]Step 1: Searching web...[/cyan]")
    results = await search_web(query, top_k=top_k)
    console.print(f"✅ Got {len(results)} initial results\n")

    if not results:
        console.print("[red]No results found![/red]")
        console.print("\n[yellow]💡 Check your API keys:[/yellow]")
        console.print("   - EXA_API_KEY in .env")
        console.print("   - SERPER_API_KEY in .env")
        return

    # Step 2: Fetch full content
    console.print("[cyan]Step 2: Fetching full content...[/cyan]")
    results = await fetch_and_extract(results)
    console.print(f"✅ Got {len(results)} results with full content\n")

    # Display results table
    console.print("[bold green]📊 Results Overview:[/bold green]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Title", width=50)
    table.add_column("Domain", width=30)
    table.add_column("Content", justify="right", width=12)

    for i, result in enumerate(results, 1):
        domain = result.meta.get("domain", "unknown")
        # Show score if it's a real relevance score from EXA
        score_display = f"{result.score:.4f}" if result.score else "N/A"
        table.add_row(
            str(i),
            score_display,
            (result.title or "Untitled")[:50],
            domain[:30],
            f"{len(result.content):,} chars",
        )

    console.print(table)

    # Display detailed content for each result
    console.print("\n[bold green]📄 Detailed Results:[/bold green]\n")

    for i, result in enumerate(results, 1):
        console.print(f"[bold cyan]Result #{i}[/bold cyan]")
        if result.score:
            console.print(f"[dim]EXA Relevance Score: {result.score:.4f}[/dim]")

        # Metadata
        meta_info = []
        if result.title:
            meta_info.append(f"Title: {result.title}")
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

    # Summary stats
    console.print("[bold yellow]📈 Summary Statistics:[/bold yellow]")
    console.print(f"  • Total results: {len(results)}")

    # Show scores if available
    scores = [r.score for r in results if r.score is not None and r.score > 0]
    if scores:
        console.print(f"  • Average relevance: {sum(scores) / len(scores):.4f}")

    console.print(f"  • Total content: {sum(len(r.content) for r in results):,} characters")
    console.print(
        f"  • Avg content length: {sum(len(r.content) for r in results) // len(results):,} chars/result"
    )

    # Show unique domains
    unique_domains = set(r.meta.get("domain", "unknown") for r in results)
    console.print(f"  • Unique domains: {len(unique_domains)}")
    console.print("\n[bold yellow]🌐 Domains:[/bold yellow]")
    for domain in sorted(unique_domains):
        count = sum(1 for r in results if r.meta.get("domain") == domain)
        console.print(f"    • {domain} ({count} results)")


async def test_node_with_filtering(gap_queries: list[str], query: str, top_k: int = 3):
    """Test web search node with LLM filtering (exactly as in pipeline)."""

    console.print("\n[bold blue]🧠 Testing Web Search Node (with LLM Filtering)[/bold blue]\n")
    console.print(f"[yellow]Original Query:[/yellow] {query}")
    console.print(f"[yellow]Gap Queries ({len(gap_queries)}):[/yellow]")
    for i, q in enumerate(gap_queries, 1):
        console.print(f"  {i}. {q}")
    console.print()

    config = get_config()

    # Run parallel searches (same as web_search_node)
    console.print("[cyan]Step 1: Running parallel gap query searches...[/cyan]")
    search_tasks = [search_web(q, top_k) for q in gap_queries]
    search_results_list = await asyncio.gather(*search_tasks)

    # Flatten results
    web_results = []
    for results in search_results_list:
        web_results.extend(results)

    console.print(f"✅ Got {len(web_results)} total results\n")

    # Fetch full content
    console.print("[cyan]Step 2: Fetching full content...[/cyan]")
    web_results = await fetch_and_extract(web_results)
    console.print(f"✅ {len(web_results)} results with full content\n")

    # Apply LLM filtering
    if config.enable_content_filtering:
        console.print("[cyan]Step 3: Applying LLM content filtering...[/cyan]")
        user_context = f"Looking for evidence to write tweets about: {query}"

        filtered_results = await filter_and_download(
            web_results,
            query=query,
            output_dir=Path(config.media_output_dir),
            user_context=user_context,
            max_concurrent=config.max_filter_concurrent,
        )

        console.print(f"✅ {len(filtered_results)} results after filtering\n")

        # Display filtered results
        console.print("[bold green]📊 Filtered Results:[/bold green]\n")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=3)
        table.add_column("Title", width=40)
        table.add_column("Relevance", justify="right", width=10)
        table.add_column("Credibility", justify="right", width=12)
        table.add_column("Media", justify="right", width=8)
        table.add_column("Key Points", justify="right", width=12)

        for i, result in enumerate(filtered_results, 1):
            table.add_row(
                str(i),
                (result.title or "Untitled")[:40],
                f"{result.relevance_score:.2f}",
                f"{result.credibility_score:.2f}",
                str(len(result.media_files)),
                str(len(result.key_points)),
            )

        console.print(table)

        # Show detailed view of top result
        if filtered_results:
            console.print("\n[bold green]📄 Top Result Detail:[/bold green]\n")
            result = filtered_results[0]

            console.print(f"[bold cyan]{result.title}[/bold cyan]")
            console.print(f"[dim]URL: {result.original_url}[/dim]")
            console.print(
                f"[dim]Relevance: {result.relevance_score:.2f} | Credibility: {result.credibility_score:.2f}[/dim]\n"
            )

            console.print("[yellow]Extracted Relevant Text:[/yellow]")
            console.print(
                Panel(
                    result.relevant_text[:500] + "..."
                    if len(result.relevant_text) > 500
                    else result.relevant_text,
                    border_style="green",
                )
            )

            if result.key_points:
                console.print("\n[yellow]Key Points:[/yellow]")
                for point in result.key_points[:5]:
                    console.print(f"  • {point}")

            if result.media_files:
                console.print(f"\n[yellow]Media Files ({len(result.media_files)}):[/yellow]")
                for media in result.media_files[:3]:
                    console.print(f"  • {media.media_type}: {media.description[:80]}")
                    if media.local_path:
                        console.print(f"    [dim]Downloaded to: {media.local_path}[/dim]")
    else:
        console.print("[yellow]⚠️  Content filtering is disabled[/yellow]")


def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test web search node with EXA + LLM filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: Full node with filtering (query + gap queries)
  python inspect_websearch.py "climate change" "climate policy 2024" "emissions data"
  
  # Another example
  python inspect_websearch.py "degrowth" "degrowth 2024 stats" "Jason Hickel latest"
  
  # Basic search only (no filtering)
  python inspect_websearch.py "degrowth economics" --basic
        """,
    )
    parser.add_argument("query", help="Main search query / topic")
    parser.add_argument(
        "gap_queries",
        nargs="*",
        help="Gap queries for targeted web search (default mode requires at least one)",
    )
    parser.add_argument(
        "--top-k", "-k", type=int, default=3, help="Number of results per gap query (default: 3)"
    )
    parser.add_argument(
        "--basic",
        "-b",
        action="store_true",
        help="Run basic search only (no gap queries, no filtering)",
    )

    args = parser.parse_args()

    # Default: Full node with filtering
    if not args.basic:
        if not args.gap_queries:
            print("Error: Default mode requires gap queries.")
            print("\nUsage:")
            print('  python inspect_websearch.py "topic" "gap query 1" "gap query 2"')
            print("\nOr use --basic for simple search:")
            print('  python inspect_websearch.py "query" --basic')
            return
        asyncio.run(test_node_with_filtering(args.gap_queries, args.query, args.top_k))
    else:
        # Basic mode: just search, no filtering
        asyncio.run(inspect_web_search(args.query, args.top_k))


if __name__ == "__main__":
    main()
