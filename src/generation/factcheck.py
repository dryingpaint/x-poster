"""Fact-checking tweets against evidence."""

import json

from openai import AsyncOpenAI

from src.core.config import get_config
from src.core.models import EvidencePack, Tweet


async def fact_check_tweets(tweets: list[Tweet], evidence: EvidencePack) -> list[Tweet]:
    """
    Fact-check tweets against evidence pack.

    Ensures every factual claim is supported and has citations.
    Edits tweets minimally to fix unsupported claims.

    Args:
        tweets: List of tweets to check
        evidence: Evidence pack

    Returns:
        Corrected tweets
    """
    config = get_config()
    client = AsyncOpenAI(api_key=config.openai_api_key)

    if not tweets:
        return []

    # Prepare evidence for prompt
    evidence_items = []
    for i, fact in enumerate(evidence.facts, 1):
        evidence_items.append(f'[{i}] {fact.fact}\n   Quote: "{fact.quote}"')

    evidence_text = "\n\n".join(evidence_items)

    # Prepare tweets for checking
    tweets_text = []
    for i, tweet in enumerate(tweets):
        tweets_text.append(f"Tweet {i + 1}: {tweet.text}")

    tweets_str = "\n".join(tweets_text)

    system_prompt = """You are a fact-checker. Verify tweets against evidence.

Rules:
1. Every factual claim MUST be supported by evidence
2. Every fact MUST have a citation [1], [2], etc.
3. Remove or hedge unsupported claims
4. Don't add new information not in evidence
5. Minimal edits - only fix factual errors or missing citations
6. Keep engaging tone

Return JSON:
{
  "tweets": [
    {"text": "corrected tweet text [1]", "issues": "what was fixed"},
    ...
  ]
}"""

    user_prompt = f"""Evidence:
{evidence_text}

Tweets to check:
{tweets_str}

Verify each tweet:
- Is every fact supported by evidence?
- Does every fact have a citation?
- Are there unsupported superlatives or absolute claims?

Return corrected tweets (minimal edits only)."""

    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=config.fact_check_temperature,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        parsed = json.loads(content)

        corrected_tweets = []
        for i, tweet_data in enumerate(parsed.get("tweets", [])):
            if i >= len(tweets):
                break

            # Update tweet text, keep existing citations structure
            original_tweet = tweets[i]
            corrected_text = tweet_data.get("text", original_tweet.text)

            corrected_tweets.append(Tweet(text=corrected_text, citations=original_tweet.citations))

        return corrected_tweets

    except Exception as e:
        print(f"Fact-check failed: {e}")
        # Return original tweets on failure
        return tweets
