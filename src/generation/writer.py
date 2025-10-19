"""Tweet generation with inline citations."""

import json
import re

from openai import AsyncOpenAI

from src.core.config import get_config
from src.core.models import Citation, EvidencePack, Tweet


async def generate_tweets(
    query: str, evidence: EvidencePack, max_variants: int = 3, max_thread_tweets: int = 6
) -> tuple[list[Tweet], list[Tweet]]:
    """
    Generate tweet variants with inline citations.

    Args:
        query: The original query/prompt
        evidence: Evidence pack with facts and sources
        max_variants: Max number of single tweet variants
        max_thread_tweets: Max tweets in a thread

    Returns:
        Tuple of (single_variants, thread_tweets)
    """
    config = get_config()
    client = AsyncOpenAI(api_key=config.openai_api_key)

    # Prepare evidence summary for the prompt
    evidence_items = []
    source_map = {}

    for i, fact in enumerate(evidence.facts, 1):
        evidence_items.append(
            f"[{i}] {fact.fact}\n   Quote: \"{fact.quote}\"\n   Source: {fact.title or fact.url or 'Unknown'}"
        )
        source_map[i] = fact.source_id

    evidence_text = "\n\n".join(evidence_items)

    system_prompt = """You are an expert tweet writer. Write engaging, factual tweets with inline citations.

Rules:
1. Use inline numeric citations like [1], [2] for each fact
2. Every factual claim MUST have a citation
3. Keep tweets under 280 characters (or split into thread)
4. For single tweets: include 1-2 citations inline
5. For threads: final tweet lists "Sources: [1] [2] [3]..."
6. Be engaging but accurate - no hype or unsupported claims
7. Use clear, accessible language

Return JSON:
{
  "variants": [
    {"text": "Tweet text with [1] citations", "citation_numbers": [1, 2]},
    ...
  ],
  "thread": [
    {"text": "Thread pt 1 [1]", "citation_numbers": [1]},
    {"text": "Thread pt 2 [2]", "citation_numbers": [2]},
    {"text": "Sources: [1] [2]", "citation_numbers": [1, 2]}
  ]
}"""

    user_prompt = f"""Topic: {query}

Evidence:
{evidence_text}

Generate:
1. {max_variants} single tweet variants (must be <280 chars each)
2. One thread (2-{max_thread_tweets} tweets) if topic needs more depth

Include inline citations [1][2] for every fact. Make it engaging and informative."""

    try:
        response = await client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=config.writer_temperature,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        parsed = json.loads(content)

        # Parse variants
        variants = []
        for variant_data in parsed.get("variants", []):
            text = variant_data.get("text", "")
            citation_nums = variant_data.get("citation_numbers", [])

            citations = [
                Citation(n=n, source_id=source_map.get(n, "unknown"))
                for n in citation_nums
                if n in source_map
            ]

            variants.append(Tweet(text=text, citations=citations))

        # Parse thread
        thread = []
        for tweet_data in parsed.get("thread", []):
            text = tweet_data.get("text", "")
            citation_nums = tweet_data.get("citation_numbers", [])

            citations = [
                Citation(n=n, source_id=source_map.get(n, "unknown"))
                for n in citation_nums
                if n in source_map
            ]

            thread.append(Tweet(text=text, citations=citations))

        return variants[:max_variants], thread[:max_thread_tweets]

    except Exception as e:
        print(f"Tweet generation failed: {e}")
        # Return empty on failure
        return [], []


def extract_citation_numbers(text: str) -> list[int]:
    """Extract citation numbers from text like [1], [2], etc."""
    matches = re.findall(r"\[(\d+)\]", text)
    return [int(m) for m in matches]

