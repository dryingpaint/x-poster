"""Merge and deduplicate search results from multiple sources."""

from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.core.config import get_config
from src.core.models import SearchResult


def compute_minhash_similarity(text1: str, text2: str, num_hashes: int = 128) -> float:
    """
    Compute MinHash similarity between two texts.

    Returns similarity score between 0 and 1.
    """

    # Simple implementation using character n-grams
    def get_shingles(text: str, n: int = 3) -> set[str]:
        text = text.lower()
        return {text[i : i + n] for i in range(len(text) - n + 1)}

    shingles1 = get_shingles(text1)
    shingles2 = get_shingles(text2)

    if not shingles1 or not shingles2:
        return 0.0

    # Jaccard similarity
    intersection = len(shingles1 & shingles2)
    union = len(shingles1 | shingles2)

    return intersection / union if union > 0 else 0.0


def compute_cosine_similarity_from_embeddings(
    embedding1: list[float], embedding2: list[float]
) -> float:
    """Compute cosine similarity between two embeddings."""
    if not embedding1 or not embedding2:
        return 0.0

    vec1 = np.array(embedding1).reshape(1, -1)
    vec2 = np.array(embedding2).reshape(1, -1)

    return float(cosine_similarity(vec1, vec2)[0][0])


def deduplicate_results(
    results: list[SearchResult], embeddings: list[list[float]] | None = None
) -> list[SearchResult]:
    """
    Remove duplicate or very similar results.

    Uses both cosine similarity (if embeddings provided) and MinHash.
    """
    config = get_config()

    if not results:
        return []

    # Track which results to keep
    keep = [True] * len(results)

    for i in range(len(results)):
        if not keep[i]:
            continue

        for j in range(i + 1, len(results)):
            if not keep[j]:
                continue

            # Check URL exact match
            if results[i].url and results[j].url and results[i].url == results[j].url:
                keep[j] = False
                continue

            # Check embedding similarity if available
            if embeddings and i < len(embeddings) and j < len(embeddings):
                cos_sim = compute_cosine_similarity_from_embeddings(embeddings[i], embeddings[j])
                if cos_sim > config.cosine_similarity_threshold:
                    keep[j] = False
                    continue

            # Check MinHash similarity
            minhash_sim = compute_minhash_similarity(results[i].content, results[j].content)
            if minhash_sim > config.minhash_similarity_threshold:
                keep[j] = False
                continue

    return [result for i, result in enumerate(results) if keep[i]]


def enforce_domain_diversity(
    results: list[SearchResult], max_per_domain: int = 2
) -> list[SearchResult]:
    """
    Limit number of results from each domain to ensure diversity.
    """
    domain_counts: dict[str, int] = defaultdict(int)
    filtered_results = []

    for result in results:
        domain = result.meta.get("domain", "unknown")

        if domain_counts[domain] < max_per_domain:
            filtered_results.append(result)
            domain_counts[domain] += 1

    return filtered_results


def merge_and_dedupe_results(
    internal_results: list[SearchResult],
    web_results: list[SearchResult],
    internal_embeddings: list[list[float]] | None = None,
    web_embeddings: list[list[float]] | None = None,
    final_k: int = 8,
) -> list[SearchResult]:
    """
    Merge internal and web results, deduplicate, and enforce diversity.

    PRIORITIZES internal results by boosting their scores.

    Args:
        internal_results: Results from internal search (PRIORITIZED)
        web_results: Results from web search (gap-filling)
        internal_embeddings: Embeddings for internal results (optional)
        web_embeddings: Embeddings for web results (optional)
        final_k: Final number of results to return

    Returns:
        Merged, deduplicated, and diverse list of results
    """
    config = get_config()

    # Boost internal result scores to prioritize them
    # Internal sources get a 1.5x multiplier
    for result in internal_results:
        result.score = result.score * 1.5

    # Merge all results
    all_results = internal_results + web_results
    all_embeddings = None

    if internal_embeddings and web_embeddings:
        all_embeddings = internal_embeddings + web_embeddings

    # Deduplicate
    deduped_results = deduplicate_results(all_results, all_embeddings)

    # Enforce domain diversity
    diverse_results = enforce_domain_diversity(
        deduped_results, max_per_domain=config.max_passages_per_domain
    )

    # Sort by score (internal results will rank higher due to boost)
    diverse_results.sort(key=lambda x: x.score, reverse=True)

    return diverse_results[:final_k]
