"""Cross-encoder reranking using BGE reranker."""

from FlagEmbedding import FlagReranker

from src.core.config import get_config
from src.core.models import SearchResult

# Global reranker instance (lazy loaded)
_reranker = None


def get_reranker() -> FlagReranker:
    """Get or create reranker instance."""
    global _reranker

    if _reranker is None:
        config = get_config()
        _reranker = FlagReranker(config.reranker_model, use_fp16=True)

    return _reranker


def rerank_results(
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

    reranker = get_reranker()

    # Prepare pairs for reranking
    pairs = [[query, result.content] for result in results]

    # Get reranker scores
    scores = reranker.compute_score(pairs, normalize=True)

    # Handle single result case
    if isinstance(scores, float):
        scores = [scores]

    # Update scores and sort
    for result, score in zip(results, scores, strict=True):
        result.score = float(score)

    # Sort by score and take top_k
    reranked = sorted(results, key=lambda x: x.score, reverse=True)
    return reranked[:top_k]

