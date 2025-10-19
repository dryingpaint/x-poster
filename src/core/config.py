"""Configuration management using pydantic-settings."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Supabase
    supabase_url: str
    supabase_service_role_key: str

    # OpenAI
    openai_api_key: str

    # Embeddings / Reranker
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    reranker_model: str = "BAAI/bge-reranker-large"

    # Web Search (choose primary; keep both keys for fallback)
    exa_api_key: str | None = None
    serper_api_key: str | None = None
    firecrawl_api_key: str | None = None
    primary_search_provider: Literal["exa", "serper"] = "exa"

    # Optional: Redis for caching
    redis_url: str = "redis://localhost:6379/0"

    # Performance tuning
    web_fetch_timeout: int = 6
    internal_query_timeout: int = 3
    max_concurrent_fetches: int = 10

    # Retrieval parameters
    internal_top_k: int = 50
    web_top_k: int = 50
    final_top_k: int = 8
    rerank_k: int = 12

    # Generation parameters
    max_thread_tweets: int = 6
    default_max_variants: int = 3
    fact_check_temperature: float = 0.1
    writer_temperature: float = 0.7

    # Cache TTL (seconds)
    web_cache_ttl: int = 86400  # 24 hours

    # Deduplication thresholds
    cosine_similarity_threshold: float = 0.95
    minhash_similarity_threshold: float = 0.9

    # Domain diversity
    max_passages_per_domain: int = 2


@lru_cache
def get_config() -> Config:
    """Get cached configuration instance."""
    return Config()

