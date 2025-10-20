"""LLM-based reranking with windowed scoring and parallel requests.

We segment long passages into token windows (with overlap) and ask an LLM
to score each window 0..1; the passage score is the max window score.
Windows are scored in parallel batches to reduce latency.
"""

import asyncio
import tiktoken
from openai import AsyncOpenAI

from src.core.config import get_config
from src.core.models import SearchResult

def get_reranker():
    return None


async def rerank_results(
    query: str, results: list[SearchResult], top_k: int = 12
) -> list[SearchResult]:
    """
    Rerank search results using cross-encoder.

    Args:
        query: The search query
        results: List of SearchResult objects
        top_k: Number of top results to return

    Returns:
        Reranked and truncated list of SearchResult objects
    """
    if not results:
        return []

    config = get_config()
    # OpenAI LLM reranking only (parallelized)
    if config.reranker_provider in ("openai_llm", "local"):
        try:
            client = AsyncOpenAI(api_key=config.openai_api_key)
            # Prepare concise doc snippets (windowed as above)
            max_tokens_per_window = 384
            overlap_tokens = 64
            enc = tiktoken.get_encoding("cl100k_base")

            docs: list[str] = []
            doc_owner: list[int] = []  # index of original result per window
            for idx, result in enumerate(results):
                text = result.content or ""
                tokens = enc.encode(text)
                if len(tokens) <= max_tokens_per_window:
                    windows = [text]
                else:
                    windows = []
                    step = max(1, max_tokens_per_window - overlap_tokens)
                    for start in range(0, len(tokens), step):
                        end = start + max_tokens_per_window
                        window_tokens = tokens[start:end]
                        if not window_tokens:
                            continue
                        windows.append(enc.decode(window_tokens))
                for w in windows:
                    docs.append(w)
                    doc_owner.append(idx)

            if not docs:
                return []

            # Chunk docs into parallel requests
            batch_size = 32
            semaphore = asyncio.Semaphore(4)  # at most 4 concurrent requests

            async def score_chunk(start: int, end: int) -> list[float]:
                sub_docs = docs[start:end]
                system_prompt = (
                    "You are a reranker. Score each document for relevance to the query between 0 and 1. "
                    "Return a JSON object {\"scores\": [..]} with a score per input document in order."
                )
                user_payload = {"query": query, "documents": sub_docs}
                async with semaphore:
                    resp = await client.chat.completions.create(
                        model=config.reranker_llm_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": str(user_payload)},
                        ],
                        temperature=0.0,
                        response_format={"type": "json_object"},
                    )
                import json as _json
                content = resp.choices[0].message.content
                parsed = _json.loads(content)
                scores = parsed.get("scores", [])
                if not isinstance(scores, list) or len(scores) != len(sub_docs):
                    raise ValueError("Invalid LLM rerank response chunk")
                return [float(s) if isinstance(s, (int, float, str)) else 0.0 for s in scores]

            tasks = []
            indices: list[tuple[int, int]] = []
            for start in range(0, len(docs), batch_size):
                end = min(start + batch_size, len(docs))
                indices.append((start, end))
                tasks.append(score_chunk(start, end))

            chunks = await asyncio.gather(*tasks)
            window_scores: list[float] = [0.0] * len(docs)
            offset = 0
            for (start, end), scores in zip(indices, chunks, strict=True):
                window_scores[start:end] = scores

            # Reduce: max per original result
            per_result: list[float] = [0.0] * len(results)
            for score, owner in zip(window_scores, doc_owner, strict=True):
                if score > per_result[owner]:
                    per_result[owner] = score

            for r, s in zip(results, per_result, strict=True):
                r.score = float(s)

            return sorted(results, key=lambda x: x.score, reverse=True)[:top_k]
        except Exception as e:
            print(f"OpenAI LLM rerank failed: {e}")
    # If we reach here, return original order (no rerank)
    return results[:top_k]

