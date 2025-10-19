"""Database client and operations."""

from .client import get_db_client, SupabaseClient
from .operations import (
    insert_item,
    insert_chunks,
    search_internal,
    cache_web_page,
    get_cached_web_page,
)

__all__ = [
    "get_db_client",
    "SupabaseClient",
    "insert_item",
    "insert_chunks",
    "search_internal",
    "cache_web_page",
    "get_cached_web_page",
]

