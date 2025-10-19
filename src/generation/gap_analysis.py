"""Gap analysis to identify missing information for targeted web search."""

import json

from openai import AsyncOpenAI

from src.core.config import get_config
from src.core.models import SearchResult


async def analyze_gaps(query: str, internal_results: list[SearchResult]) -> list[str]:
    """
    Analyze internal search results to identify gaps in knowledge.

    Returns a list of targeted search queries to fill those gaps.

    Args:
        query: Original user prompt
        internal_results: Results from internal search

    Returns:
        List of specific search queries to fill knowledge gaps
    """
    config = get_config()
    client = AsyncOpenAI(api_key=config.openai_api_key)

    # If we have very few internal results, do a broad web search
    if len(internal_results) < 3:
        return [query]

    # Prepare internal content summary
    internal_content = []
    for i, result in enumerate(internal_results[:10], 1):  # Top 10 for context
        internal_content.append(f"[{i}] {result.title or 'Untitled'}\n{result.content[:300]}...")

    content_summary = "\n\n".join(internal_content)

    system_prompt = """You are a research assistant. Analyze internal documents to identify gaps in knowledge.

Given a query and internal search results, identify:
1. Missing statistics, numbers, or recent data
2. Missing expert opinions or authoritative sources
3. Missing visual evidence (charts, graphs, images)
4. Missing context or background information
5. Missing recent developments or news

Return 2-5 specific search queries that would fill these gaps. Make queries concrete and targeted.

Return JSON:
{
  "gaps": [
    {"type": "statistics", "query": "specific search query"},
    {"type": "expert_opinion", "query": "specific search query"},
    {"type": "visual", "query": "chart OR graph OR infographic specific topic"},
    ...
  ]
}"""

    user_prompt = f"""Original Query: {query}

Internal Documents Found:
{content_summary}

What information is missing that would strengthen a tweet about this topic?
Generate 2-5 targeted web search queries to fill gaps.

Focus on:
- Recent statistics or data (if internal docs are older)
- Visual evidence (charts, graphs, infographics)
- Expert quotes or authoritative sources
- Current news or developments

Return only valid JSON."""

    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        parsed = json.loads(content)

        gaps = parsed.get("gaps", [])

        # Extract queries
        queries = []
        for gap in gaps[:5]:  # Max 5 targeted queries
            query_text = gap.get("query", "")
            if query_text:
                queries.append(query_text)

        # If no gaps identified, fall back to original query
        if not queries:
            queries = [query]

        print(f"   Identified {len(queries)} gap-filling queries")
        for q in queries:
            print(f"      • {q}")

        return queries

    except Exception as e:
        print(f"Gap analysis failed: {e}")
        # Fall back to original query
        return [query]
