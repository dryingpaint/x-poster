"""Supabase database client."""

from functools import lru_cache

from supabase import create_client, Client

from src.core.config import get_config


class SupabaseClient:
    """Wrapper around Supabase client for database operations."""

    def __init__(self, url: str, key: str):
        self.client: Client = create_client(url, key)

    def get_client(self) -> Client:
        """Get the underlying Supabase client."""
        return self.client


@lru_cache()
def get_db_client() -> SupabaseClient:
    """Get cached database client instance."""
    config = get_config()
    return SupabaseClient(
        url=config.supabase_url,
        key=config.supabase_service_role_key,
    )

