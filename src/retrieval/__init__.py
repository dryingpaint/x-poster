"""Retrieval modules for internal and web search."""

from .web_search import search_web, fetch_and_extract
from .reranker import rerank_results
from .merger import merge_and_dedupe_results

__all__ = [
    "search_web",
    "fetch_and_extract",
    "rerank_results",
    "merge_and_dedupe_results",
]

