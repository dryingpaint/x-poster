"""Test individual LangGraph components/nodes independently."""

import asyncio
import json

from rich.console import Console
from rich.panel import Panel

from src.orchestrator.state import AgentState
from src.orchestrator.tools import (
    embed_query_node,
    evidence_pack_node,
    fact_check_node,
    gap_analysis_node,
    internal_search_node,
    merge_dedupe_node,
    prepare_response_node,
    rerank_node,
    tweet_generation_node,
    web_search_node,
)

console = Console()


async def test_embed_query(state: AgentState = None):
    """Test Step 1: Embed query."""
    console.print("\n[bold blue]Step 1: Embed Query[/bold blue]")

    if state is None:
        state: AgentState = {
        "query": "degrowth",
        "max_variants": 1,
        "max_thread_tweets": 3,
        "query_embedding": None,
        "internal_results": None,
        "gap_queries": None,
        "web_results": None,
        "merged_results": None,
        "final_results": None,
        "evidence": None,
        "variants": None,
        "thread": None,
        "response": None,
        "error": None,
        }

    result = await embed_query_node(state)
    console.print(f"   ✅ Embedding dimension: {len(result['query_embedding'])}")
    return {**state, **result}


async def test_internal_search(state: AgentState):
    """Test Step 2: Internal search."""
    console.print("\n[bold blue]Step 2: Internal Search[/bold blue]")

    result = await internal_search_node(state)
    console.print(f"   ✅ Found {len(result['internal_results'])} internal results")

    if result['internal_results']:
        console.print("\n   Sample result:")
        sample = result['internal_results'][0]
        console.print(f"   - Title: {sample.title}")
        console.print(f"   - Score: {sample.score:.3f}")
        console.print(f"   - Content preview: {sample.content[:100]}...")

    return {**state, **result}


async def test_gap_analysis(state: AgentState):
    """Test Step 3: Gap analysis."""
    console.print("\n[bold blue]Step 3: Gap Analysis[/bold blue]")

    result = await gap_analysis_node(state)
    console.print(f"   ✅ Identified {len(result['gap_queries'])} gap queries")

    for i, query in enumerate(result['gap_queries'], 1):
        console.print(f"   {i}. {query}")

    return {**state, **result}


async def test_web_search(state: AgentState):
    """Test Step 4: Web search."""
    console.print("\n[bold blue]Step 4: Web Search[/bold blue]")

    result = await web_search_node(state)
    console.print(f"   ✅ Found {len(result['web_results'])} web results")

    if result['web_results']:
        console.print("\n   Sample web result:")
        sample = result['web_results'][0]
        console.print(f"   - Title: {sample.title}")
        console.print(f"   - URL: {sample.url}")
        console.print(f"   - Score: {sample.score:.3f}")
        console.print(f"   - Content length: {len(sample.content)} chars")

    return {**state, **result}


async def test_merge_dedupe(state: AgentState):
    """Test Step 5: Merge & dedupe."""
    console.print("\n[bold blue]Step 5: Merge & Dedupe[/bold blue]")

    result = await merge_dedupe_node(state)
    console.print(f"   ✅ Merged to {len(result['merged_results'])} results")

    # Count internal vs web
    internal_count = sum(1 for r in result['merged_results'] if r.source_type == 'internal')
    web_count = sum(1 for r in result['merged_results'] if r.source_type == 'web')
    console.print(f"   - Internal: {internal_count}, Web: {web_count}")

    return {**state, **result}


async def test_rerank(state: AgentState):
    """Test Step 6: Rerank."""
    console.print("\n[bold blue]Step 6: Rerank[/bold blue]")

    result = await rerank_node(state)
    console.print(f"   ✅ Reranked to {len(result['final_results'])} final results")

    console.print("\n   Top 3 results:")
    for i, r in enumerate(result['final_results'][:3], 1):
        console.print(f"   {i}. [{r.source_type}] {r.title or 'Untitled'} (score: {r.score:.3f})")

    return {**state, **result}


async def test_evidence_pack(state: AgentState):
    """Test Step 7: Evidence pack."""
    console.print("\n[bold blue]Step 7: Evidence Pack Creation[/bold blue]")

    result = await evidence_pack_node(state)
    console.print(f"   ✅ Extracted {len(result['evidence'].facts)} facts")

    console.print("\n   Sample facts:")
    for i, fact in enumerate(result['evidence'].facts[:3], 1):
        console.print(f"   {i}. {fact.fact}")
        console.print(f"      Quote: \"{fact.quote}\"")
        console.print(f"      Confidence: {fact.confidence:.2f}\n")

    return {**state, **result}


async def test_tweet_generation(state: AgentState):
    """Test Step 8: Tweet generation."""
    console.print("\n[bold blue]Step 8: Tweet Generation[/bold blue]")

    result = await tweet_generation_node(state)
    console.print(f"   ✅ Generated {len(result['variants'])} variants")
    console.print(f"   ✅ Generated thread with {len(result['thread'])} tweets")

    if result['variants']:
        console.print("\n   Variant 1:")
        console.print(Panel(result['variants'][0].text, border_style="green"))

    return {**state, **result}


async def test_fact_check(state: AgentState):
    """Test Step 9: Fact checking."""
    console.print("\n[bold blue]Step 9: Fact Checking[/bold blue]")

    result = await fact_check_node(state)
    console.print(f"   ✅ Fact-checked {len(result['variants'])} variants")
    console.print(f"   ✅ Fact-checked {len(result['thread'])} thread tweets")

    if result['variants']:
        console.print("\n   Fact-checked variant 1:")
        console.print(Panel(result['variants'][0].text, border_style="cyan"))

    return {**state, **result}


async def test_prepare_response(state: AgentState):
    """Test Step 10: Prepare response."""
    console.print("\n[bold blue]Step 10: Prepare Response[/bold blue]")

    result = await prepare_response_node(state)
    console.print(f"   ✅ Prepared response with {len(result['response'].sources)} sources")

    return {**state, **result}


async def run_all_tests():
    """Run all component tests in sequence."""
    console.print("[bold green]🧪 Testing LangGraph Components[/bold green]")
    console.print("=" * 60)

    try:
        # Step 1: Embed query
        state = await test_embed_query()

        # Step 2: Internal search
        state = await test_internal_search(state)

        # Step 3: Gap analysis
        state = await test_gap_analysis(state)

        # Step 4: Web search
        state = await test_web_search(state)

        # Step 5: Merge & dedupe
        state = await test_merge_dedupe(state)

        # Step 6: Rerank
        state = await test_rerank(state)

        # Step 7: Evidence pack
        state = await test_evidence_pack(state)

        # Step 8: Tweet generation
        state = await test_tweet_generation(state)

        # Step 9: Fact checking
        state = await test_fact_check(state)

        # Step 10: Prepare response
        state = await test_prepare_response(state)

        console.print("\n[bold green]✅ All components tested successfully![/bold green]")
        console.print("=" * 60)

        # Show final response
        if state.get('response'):
            console.print("\n[bold yellow]Final Response:[/bold yellow]")
            console.print(f"\nVariants: {len(state['response'].variants)}")
            console.print(f"Thread tweets: {len(state['response'].thread)}")
            console.print(f"Sources: {len(state['response'].sources)}")

            if state['response'].variants:
                console.print("\n[bold]Final Tweet:[/bold]")
                console.print(Panel(state['response'].variants[0].text, border_style="green"))

    except Exception as e:
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())


async def test_single_component(component_name: str):
    """Test a single component by name."""

    # Map component names to test functions
    components = {
        "embed_query": (test_embed_query, None),
        "internal_search": (test_internal_search, test_embed_query),
        "gap_analysis": (test_gap_analysis, test_internal_search),
        "web_search": (test_web_search, test_gap_analysis),
        "merge_dedupe": (test_merge_dedupe, test_web_search),
        "rerank": (test_rerank, test_merge_dedupe),
        "evidence_pack": (test_evidence_pack, test_rerank),
        "tweet_generation": (test_tweet_generation, test_evidence_pack),
        "fact_check": (test_fact_check, test_tweet_generation),
        "prepare_response": (test_prepare_response, test_fact_check),
    }

    if component_name not in components:
        console.print(f"[red]Unknown component: {component_name}[/red]")
        console.print(f"Available components: {', '.join(components.keys())}")
        return

    test_func, prereq_func = components[component_name]

    console.print(f"[bold green]🧪 Testing: {component_name}[/bold green]")
    console.print("=" * 60)

    try:
        # Run prerequisites if needed
        if prereq_func:
            console.print(f"[yellow]Running prerequisites...[/yellow]")
            state = await test_embed_query()

            # Run all prerequisites in order
            prereq_order = [
                test_embed_query,
                test_internal_search,
                test_gap_analysis,
                test_web_search,
                test_merge_dedupe,
                test_rerank,
                test_evidence_pack,
                test_tweet_generation,
                test_fact_check,
            ]

            for prereq in prereq_order:
                if prereq == test_func:
                    break
                if prereq == test_embed_query:
                    continue  # Already ran
                state = await prereq(state)
        else:
            state = None

        # Run the target test
        result = await test_func(state)

        console.print(f"\n[bold green]✅ {component_name} tested successfully![/bold green]")

    except Exception as e:
        console.print(f"\n[bold red]❌ Error in {component_name}: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Test specific component
        component = sys.argv[1]
        asyncio.run(test_single_component(component))
    else:
        # Test all components
        asyncio.run(run_all_tests())
