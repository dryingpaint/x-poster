"""Database client and operations."""

from .client import SupabaseClient, get_db_client
from .operations import (
    cache_web_page,
    get_cached_web_page,
    insert_chunks,
    insert_item,
    search_internal,
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

