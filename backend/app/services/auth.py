"""
Auth dependencies backed by real Supabase Auth tokens.

Supabase signs access tokens asymmetrically (ES256) and publishes its
public verification keys at a JWKS endpoint -- PyJWKClient fetches and
caches those keys automatically, so we never store a shared secret here.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient

from app.core.config import get_settings

bearer_scheme = HTTPBearer()


@lru_cache
def get_jwks_client() -> PyJWKClient:
    settings = get_settings()
    jwks_url = f"{settings.supabase_url}/auth/v1/.well-known/jwks.json"
    return PyJWKClient(jwks_url)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict[str, Any]:
    """
    Verifies the incoming Supabase access token and returns its decoded
    claims (a dict with at least 'sub' = the user's id and 'email').
    Raises 401 if the token is missing, expired, or invalid.
    """
    token = credentials.credentials
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        signing_key = get_jwks_client().get_signing_key_from_jwt(token)
        settings = get_settings()
        decode_kwargs: dict[str, Any] = {"options": {"require": ["exp", "sub"]}}
        if settings.supabase_url:
            # Audit fix: reject validly-signed tokens that were issued by a different auth server.
            decode_kwargs["issuer"] = f"{settings.supabase_url.rstrip('/')}/auth/v1"
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["ES256"],
            audience="authenticated",
            **decode_kwargs,
        )
    except jwt.PyJWTError:
        raise credentials_exception
    payload["_access_token"] = token
    return payload


def require_role(*roles: str):
    """
    Use as a dependency on any route: Depends(require_role("clinician")).
    Reads the role out of the caller's `profiles` row via a fresh lookup,
    since the JWT itself doesn't carry your app-specific role claim.
    """
    from app.core.supabase_client import get_supabase_admin_client

    def _checker(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
        admin = get_supabase_admin_client()
        result = (
            admin.table("profiles")
            .select("role")
            .eq("id", user["sub"])
            .single()
            .execute()
        )
        role = result.data["role"] if result.data else None
        if role not in roles:
            raise HTTPException(
                status_code=403,
                detail=f"Role '{role}' not permitted; requires one of {roles}",
            )
        return user

    return _checker