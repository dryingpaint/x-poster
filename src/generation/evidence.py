"""Evidence pack creation using LLM."""

import json
from datetime import datetime

from openai import AsyncOpenAI

from src.core.config import get_config
from src.core.models import EvidenceFact, EvidencePack, SearchResult


async def create_evidence_pack(
    query: str, search_results: list[SearchResult]
) -> EvidencePack:
    """
    Create an evidence pack from search results using LLM.

    The LLM extracts key facts, quotes, and source information.

    Args:
        query: The original query
        search_results: Retrieved passages

    Returns:
        EvidencePack with extracted facts and source mapping
    """
    config = get_config()
    client = AsyncOpenAI(api_key=config.openai_api_key)

    # Prepare passages for the prompt
    passages = []
    for i, result in enumerate(search_results):
        source_info = {
            "source_id": result.source_id,
            "title": result.title,
            "url": result.url,
            "author": result.author,
            "published_at": result.published_at.isoformat() if result.published_at else None,
        }

        passages.append(
            {
                "index": i,
                "content": result.content[:1000],  # Limit length
                "source": source_info,
            }
        )

    system_prompt = """You are a research assistant. Extract key facts from the provided passages that are relevant to the query.

For each fact:
1. Paraphrase it clearly
2. Include a direct quote (max 20 words) that supports it
3. Reference the source_id
4. Rate your confidence (0.0-1.0)

Return JSON array of facts:
[{
  "fact": "paraphrased fact",
  "quote": "direct quote max 20 words",
  "source_id": "src_...",
  "confidence": 0.95
}, ...]"""

    user_prompt = f"""Query: {query}

Passages:
{json.dumps(passages, indent=2)}

Extract 5-10 key facts relevant to the query. Focus on facts that are:
- Directly supported by the passages
- Specific and factual (not vague claims)
- Relevant to the query

Return only valid JSON array."""

    try:
        response = await client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        parsed = json.loads(content)

        # Handle both direct array and wrapped in "facts" key
        facts_data = parsed if isinstance(parsed, list) else parsed.get("facts", [])

        # Create EvidenceFact objects
        facts = []
        sources_map = {r.source_id: r for r in search_results}

        for fact_data in facts_data:
            source_id = fact_data.get("source_id", "")
            source = sources_map.get(source_id)

            if not source:
                continue

            fact = EvidenceFact(
                fact=fact_data.get("fact", ""),
                quote=fact_data.get("quote", ""),
                source_id=source_id,
                url=source.url,
                title=source.title,
                author=source.author,
                published_at=source.published_at,
                confidence=fact_data.get("confidence", 0.8),
            )
            facts.append(fact)

        return EvidencePack(facts=facts, sources=sources_map)

    except Exception as e:
        print(f"Evidence pack creation failed: {e}")
        # Return empty evidence pack on failure
        return EvidencePack(
            facts=[], sources={r.source_id: r for r in search_results}
        )

