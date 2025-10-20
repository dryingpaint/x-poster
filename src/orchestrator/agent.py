"""LangGraph agent for tweet generation pipeline.

This module defines the state graph that orchestrates the entire tweet generation
pipeline from query to final response. The graph is a linear pipeline with 10 nodes.
"""

from langgraph.graph import END, START, StateGraph

from src.core.models import GenerateRequest, GenerateResponse
from src.orchestrator.state import AgentState
from src.orchestrator.tools import (
    embed_query_node,
    gap_analysis_node,
    internal_search_node,
    merge_dedupe_node,
    prepare_response_node,
    rerank_node,
    tweet_generation_node,
    web_search_node,
)


def create_agent_graph() -> StateGraph:
    """Create and compile the LangGraph state graph for tweet generation.

    The graph executes a linear pipeline:
    1. Embed query
    2. Internal search
    3. Gap analysis
    4. Web search (parallel for each gap query)
    5. Merge & dedupe
    6. Rerank
    7. Tweet generation (from results)
    8. Prepare response

    Returns:
        Compiled StateGraph ready for execution
    """
    # Create the state graph
    graph = StateGraph(AgentState)

    # Add nodes (tools)
    graph.add_node("embed_query", embed_query_node)
    graph.add_node("internal_search", internal_search_node)
    graph.add_node("gap_analysis", gap_analysis_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("merge_dedupe", merge_dedupe_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("tweet_generation", tweet_generation_node)
    graph.add_node("prepare_response", prepare_response_node)

    # Define the linear pipeline flow
    graph.add_edge(START, "embed_query")
    graph.add_edge("embed_query", "internal_search")
    graph.add_edge("internal_search", "gap_analysis")
    graph.add_edge("gap_analysis", "web_search")
    graph.add_edge("web_search", "merge_dedupe")
    graph.add_edge("merge_dedupe", "rerank")
    graph.add_edge("rerank", "tweet_generation")
    graph.add_edge("tweet_generation", "prepare_response")
    graph.add_edge("prepare_response", END)

    # Compile the graph
    return graph.compile()


async def run_agent(request: GenerateRequest) -> GenerateResponse:
    """Execute the tweet generation agent.

    Args:
        request: GenerateRequest with query and parameters

    Returns:
        GenerateResponse with tweet variants, thread, and sources

    Raises:
        Exception: If any step in the pipeline fails
    """
    # Create the compiled graph
    agent = create_agent_graph()

    # Initialize state from request
    initial_state: AgentState = {
        "query": request.prompt,
        "max_variants": request.max_variants,
        "max_thread_tweets": request.max_thread_tweets,
        "query_embedding": None,
        "internal_results": None,
        "gap_queries": None,
        "web_results": None,
        "merged_results": None,
        "final_results": None,
        # no evidence layer in the new workflow
        "variants": None,
        "thread": None,
        "response": None,
        "error": None,
    }

    # Execute the graph
    final_state = await agent.ainvoke(initial_state)

    # Extract and return the response
    if final_state.get("error"):
        raise Exception(f"Agent execution failed: {final_state['error']}")

    response = final_state.get("response")
    if not response:
        raise Exception("Agent execution completed but no response was generated")

    return response
