"""
Single source of truth for talking to Supabase from the backend.

This uses the SECRET key, which bypasses Row Level Security entirely.
That's correct for backend-internal/admin operations (e.g. batch ML jobs,
seeding data) but must NEVER be used to serve a normal user request without
its own manual authorization check -- RLS won't protect you here.

Regular per-user requests get their own client below, built with the
PUBLISHABLE key, so RLS applies naturally.
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


@lru_cache
def get_supabase_public_client() -> Client:
    """
    Client built with the PUBLISHABLE key -- this is what a real logged-in
    user's request looks like. RLS is fully enforced on every call made
    through this client. Never use the secret key for user-facing requests.
    """
    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_publishable_key:
        raise RuntimeError(
            "Supabase is not configured. Check SUPABASE_URL and "
            "SUPABASE_PUBLISHABLE_KEY in backend/.env."
        )
    return create_client(settings.supabase_url, settings.supabase_publishable_key)


def get_authed_client(access_token: str) -> Client:
    """
    Builds a FRESH Supabase client authenticated as one specific user's
    access token, so requests through it are subject to Row Level Security
    as that user. Deliberately NOT cached (unlike the two clients above) --
    caching a per-user session would leak one user's identity into other
    users' concurrent requests.
    """
    settings = get_settings()
    client = create_client(settings.supabase_url, settings.supabase_publishable_key)
    client.postgrest.auth(access_token)
    return client