"""
Single source of truth for talking to Supabase from the backend.

This uses the SECRET key, which bypasses Row Level Security entirely.
That's correct for backend-internal/admin operations (e.g. batch ML jobs,
seeding data) but must NEVER be used to serve a normal user request without
its own manual authorization check -- RLS won't protect you here.

Regular per-user requests will get their own client later (Step 6+), built
with the user's own JWT, so RLS applies naturally.
"""
from __future__ import annotations

from functools import lru_cache

from supabase import create_client, Client

from app.core.config import get_settings


@lru_cache
def get_supabase_admin_client() -> Client:
    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_secret_key:
        raise RuntimeError(
            "Supabase is not configured. Check SUPABASE_URL and "
            "SUPABASE_SECRET_KEY in backend/.env."
        )
    return create_client(settings.supabase_url, settings.supabase_secret_key)