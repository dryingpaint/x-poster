"""Inspect web search functionality.

Test the web search node to see what results come back from different providers.

Usage:
    uv run python examples/inspect_websearch.py "your query"
    uv run python examples/inspect_websearch.py "degrowth economics 2024"
    uv run python examples/inspect_websearch.py "climate policy" --top-k 5
    uv run python examples/inspect_websearch.py "AI safety" --provider exa
    uv run python examples/inspect_websearch.py "degrowth" --compare
    uv run python examples/inspect_websearch.py "climate" --filter
"""

import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.core.config import get_config
from src.retrieval.web_search import (
    fetch_and_extract,
    search_and_filter,
    search_exa,
    search_serper,
    search_web,
)

console = Console()


async def inspect_web_search(query: str, top_k: int = 5, provider: str | None = None):
    """Test web search with a single query."""

    console.print("\n[bold blue]🌐 Web Search Test[/bold blue]\n")
    console.print(f"[yellow]Query:[/yellow] {query}")
    console.print(f"[yellow]Top K:[/yellow] {top_k}")

    # Override provider if specified
    config = get_config()
    original_provider = config.primary_search_provider

    if provider:
        config.primary_search_provider = provider
        console.print(f"[yellow]Provider:[/yellow] {provider}\n")
    else:
        console.print(
            f"[yellow]Provider:[/yellow] {config.primary_search_provider} (with fallback)\n"
        )

    try:
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
            table.add_row(
                str(i),
                f"{result.score:.4f}",
                (result.title or "Untitled")[:50],
                domain[:30],
                f"{len(result.content):,} chars",
            )

        console.print(table)

        # Display detailed content for each result
        console.print("\n[bold green]📄 Detailed Results:[/bold green]\n")

        for i, result in enumerate(results, 1):
            console.print(f"[bold cyan]Result #{i}[/bold cyan]")
            console.print(f"[dim]Score: {result.score:.4f}[/dim]")

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
                content_preview += (
                    f"\n\n[dim]... ({len(result.content) - 500:,} more characters)[/dim]"
                )

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
        console.print(f"  • Average score: {sum(r.score for r in results) / len(results):.4f}")
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

    finally:
        # Restore original provider
        config.primary_search_provider = original_provider


async def compare_providers(query: str, top_k: int = 5):
    """Compare EXA vs Serper search providers."""

    console.print("\n[bold blue]🔬 Comparing Search Providers[/bold blue]\n")
    console.print(f"[yellow]Query:[/yellow] {query}\n")

    # Test EXA
    console.print("[cyan]Testing EXA...[/cyan]")
    exa_results = await search_exa(query, num_results=top_k)
    console.print(f"✅ EXA: {len(exa_results)} results\n")

    # Test Serper
    console.print("[cyan]Testing Serper...[/cyan]")
    serper_results = await search_serper(query, num_results=top_k)
    console.print(f"✅ Serper: {len(serper_results)} results\n")

    # Comparison table
    console.print("[bold green]📊 Provider Comparison:[/bold green]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Provider", width=15)
    table.add_column("Results", justify="right", width=10)
    table.add_column("Has Full Content", justify="center", width=18)
    table.add_column("Top Title", width=50)

    # EXA
    exa_has_content = (
        sum(1 for r in exa_results if r.get("content") and len(r["content"]) > 200)
        if exa_results
        else 0
    )
    table.add_row(
        "EXA",
        str(len(exa_results)),
        f"{exa_has_content}/{len(exa_results)}" if exa_results else "N/A",
        (exa_results[0]["title"] if exa_results else "N/A")[:50],
    )

    # Serper
    serper_has_content = (
        sum(1 for r in serper_results if r.get("content") and len(r["content"]) > 200)
        if serper_results
        else 0
    )
    table.add_row(
        "Serper",
        str(len(serper_results)),
        f"{serper_has_content}/{len(serper_results)}" if serper_results else "N/A",
        (serper_results[0]["title"] if serper_results else "N/A")[:50],
    )

    console.print(table)

    console.print("\n[dim]💡 Notes:[/dim]")
    console.print("[dim]  - EXA often includes full content immediately[/dim]")
    console.print("[dim]  - Serper returns snippets, needs separate fetch[/dim]")
    console.print("[dim]  - Both are queried in parallel with automatic fallback[/dim]")


async def test_parallel_gap_queries(gap_queries: list[str], top_k: int = 3):
    """Test web search with multiple gap queries (simulating the node behavior)."""

    console.print("\n[bold blue]🚀 Testing Parallel Gap Queries[/bold blue]\n")
    console.print(f"[yellow]Gap Queries ({len(gap_queries)}):[/yellow]")
    for i, q in enumerate(gap_queries, 1):
        console.print(f"  {i}. {q}")
    console.print()

    # Run parallel searches (same as web_search_node)
    console.print("[cyan]Running parallel searches...[/cyan]")
    search_tasks = [search_web(query, top_k) for query in gap_queries]
    search_results_list = await asyncio.gather(*search_tasks)

    console.print(f"✅ Completed {len(search_tasks)} searches\n")

    # Flatten results
    web_results = []
    for results in search_results_list:
        web_results.extend(results)

    console.print(f"[cyan]Total results before fetch: {len(web_results)}[/cyan]")

    # Fetch full content
    web_results = await fetch_and_extract(web_results)

    console.print(f"✅ Total results after fetch: {len(web_results)}\n")

    # Show breakdown by query
    console.print("[bold green]📊 Results by Gap Query:[/bold green]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Query", width=50)
    table.add_column("Results", justify="right", width=10)
    table.add_column("Avg Content", justify="right", width=15)

    for i, (query, results) in enumerate(zip(gap_queries, search_results_list, strict=False)):
        # Filter to results that passed fetch_and_extract
        valid_results = [r for r in results if r.content and len(r.content) > 50]
        avg_content = (
            sum(len(r.content) for r in valid_results) // len(valid_results) if valid_results else 0
        )

        table.add_row(query[:50], str(len(valid_results)), f"{avg_content:,} chars")

    console.print(table)

    console.print("\n[bold yellow]📈 Summary:[/bold yellow]")
    console.print(f"  • Total unique results: {len(web_results)}")
    console.print(f"  • Total content: {sum(len(r.content) for r in web_results):,} chars")


async def test_with_filtering(query: str, top_k: int = 5):
    """Test web search with LLM content filtering enabled."""

    console.print("\n[bold blue]🧠 Testing LLM Content Filtering[/bold blue]\n")
    console.print(f"[yellow]Query:[/yellow] {query}\n")

    config = get_config()

    if not config.enable_content_filtering:
        console.print(
            "[yellow]⚠️  Content filtering is disabled in config. Set ENABLE_CONTENT_FILTERING=true[/yellow]\n"
        )

    # Run search with filtering
    console.print("[cyan]Running search with LLM filtering...[/cyan]")
    filtered_results, raw_results = await search_and_filter(
        query=query,
        top_k=top_k,
        user_context="Looking for evidence to write a tweet about this topic",
    )

    console.print(f"✅ Raw results: {len(raw_results)}")
    console.print(f"✅ Filtered results: {len(filtered_results)}\n")

    if not config.enable_content_filtering:
        console.print("[yellow]Filtering was not applied (disabled in config)[/yellow]")
        return

    # Show filtered results
    console.print("[bold green]📊 Filtered Results:[/bold green]\n")

    for i, result in enumerate(filtered_results, 1):
        console.print(f"[bold cyan]Result #{i}[/bold cyan]")
        console.print(f"[dim]URL: {result.original_url}[/dim]")
        console.print(f"[dim]Title: {result.title}[/dim]")
        console.print(
            f"[dim]Relevance: {result.relevance_score:.2f} | Credibility: {result.credibility_score:.2f}[/dim]\n"
        )

        # Relevant text
        console.print("[yellow]Extracted Relevant Text:[/yellow]")
        console.print(
            Panel(
                result.relevant_text[:500] + "..."
                if len(result.relevant_text) > 500
                else result.relevant_text,
                border_style="green",
            )
        )

        # Key points
        if result.key_points:
            console.print("\n[yellow]Key Points:[/yellow]")
            for point in result.key_points:
                console.print(f"  • {point}")

        # Media files
        if result.media_files:
            console.print(f"\n[yellow]Media Files ({len(result.media_files)}):[/yellow]")
            for media in result.media_files:
                console.print(f"  • {media.media_type}: {media.description[:100]}")

        console.print("\n" + "─" * 80 + "\n")


def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect web search functionality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic search
  python inspect_websearch.py "degrowth economics"
  
  # Specify provider
  python inspect_websearch.py "climate policy" --provider exa
  
  # Compare providers
  python inspect_websearch.py "AI safety" --compare
  
  # Test parallel gap queries
  python inspect_websearch.py "degrowth" --gaps "degrowth 2024 stats" "Jason Hickel latest"
  
  # Test LLM filtering
  python inspect_websearch.py "climate change" --filter
        """,
    )
    parser.add_argument("query", help="Search query")
    parser.add_argument(
        "--top-k", "-k", type=int, default=5, help="Number of results per query (default: 5)"
    )
    parser.add_argument(
        "--provider",
        "-p",
        choices=["exa", "serper"],
        help="Force specific provider (default: use config setting)",
    )
    parser.add_argument(
        "--compare", "-c", action="store_true", help="Compare EXA vs Serper providers"
    )
    parser.add_argument(
        "--gaps",
        "-g",
        nargs="+",
        help="Test multiple gap queries in parallel (simulates web_search_node)",
    )
    parser.add_argument(
        "--filter",
        "-f",
        action="store_true",
        help="Test LLM content filtering (requires ENABLE_CONTENT_FILTERING=true)",
    )

    args = parser.parse_args()

    if args.compare:
        asyncio.run(compare_providers(args.query, args.top_k))
    elif args.gaps:
        asyncio.run(test_parallel_gap_queries(args.gaps, args.top_k))
    elif args.filter:
        asyncio.run(test_with_filtering(args.query, args.top_k))
    else:
        asyncio.run(inspect_web_search(args.query, args.top_k, args.provider))


if __name__ == "__main__":
    main()
