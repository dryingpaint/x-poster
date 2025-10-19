"""LLM-based generation modules."""

from .embeddings import embed_batch, embed_text, get_embedder
from .evidence import create_evidence_pack
from .factcheck import fact_check_tweets
from .gap_analysis import analyze_gaps
from .writer import generate_tweets

__all__ = [
    "get_embedder",
    "embed_text",
    "embed_batch",
    "create_evidence_pack",
    "generate_tweets",
    "fact_check_tweets",
    "analyze_gaps",
]

