"""Retrieval modules for internal and web search."""

from .merger import merge_and_dedupe_results
from .reranker import rerank_results
from .web_search import fetch_and_extract, search_web

__all__ = [
    "search_web",
    "fetch_and_extract",
    "rerank_results",
    "merge_and_dedupe_results",
]

