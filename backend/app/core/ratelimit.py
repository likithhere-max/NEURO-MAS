"""
Rate limiting (Phase 2, H).

Two keyings:
  * `rate_limit_auth`   - unauthenticated auth endpoints, keyed on the real client IP.
  * `rate_limit_user()` - authenticated expensive endpoints (workflows, chat, symptom assessment), keyed on user id.

Client IP: by default the socket peer. Behind N trusted reverse proxies (TRUSTED_PROXY_COUNT=N) the client is the
N-th entry from the RIGHT of X-Forwarded-For - entries to its left are client-supplied and spoofable, so they are
never trusted. With TRUSTED_PROXY_COUNT=0 the header is ignored entirely.

Store: in-memory sliding window (per process) by default; if RATE_LIMIT_REDIS_URL is set, a Redis fixed-window counter
(INCR + EXPIRE) is shared across workers/instances. If Redis is configured but errors, the limiter FAILS OPEN to the
in-memory store for that call and logs a warning (availability over strictness for a safety-adjacent API); this is a
deliberate, documented choice. 429 responses carry Retry-After.

It is not a substitute for an edge/WAF limiter.
"""
from __future__ import annotations

import logging
import math
import time
from collections import defaultdict, deque
from typing import Callable

from fastapi import Depends, HTTPException, Request

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_hits: dict[str, deque] = defaultdict(deque)
_redis_client = None
_redis_url = None


def reset() -> None:
    global _redis_client, _redis_url
    _hits.clear()
    _redis_client, _redis_url = None, None


def client_ip(request: Request) -> str:
    peer = request.client.host if request.client else "unknown"
    trusted = max(0, int(getattr(get_settings(), "trusted_proxy_count", 0) or 0))
    if trusted == 0:
        return peer
    xff = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
    if len(xff) < trusted:
        # Fewer hops than configured: the header is not what we expect, so do not trust it.
        return peer
    return xff[-trusted]


def _memory_hit(key: str, limit: int, window: int) -> int:
    """Returns 0 if allowed, else seconds until retry."""
    now = time.monotonic()
    q = _hits[key]
    while q and now - q[0] > window:
        q.popleft()
    if len(q) >= limit:
        return max(1, math.ceil(window - (now - q[0])))
    q.append(now)
    return 0


def _redis():
    global _redis_client, _redis_url
    url = getattr(get_settings(), "rate_limit_redis_url", "") or ""
    if not url:
        return None
    if _redis_client is None or _redis_url != url:
        import redis  # optional dependency
        _redis_client, _redis_url = redis.Redis.from_url(url, socket_timeout=0.25, socket_connect_timeout=0.25), url
    return _redis_client


def _hit(key: str, limit: int, window: int = 60) -> int:
    try:
        r = _redis()
        if r is not None:
            bucket = int(time.time() // window)
            k = f"nm:rl:{key}:{bucket}"
            n = r.incr(k)
            if n == 1:
                r.expire(k, window + 1)
            return 0 if n <= limit else max(1, window - int(time.time() % window))
    except Exception as exc:  # noqa: BLE001 - fail open to the local store, but say so
        logger.warning("rate_limit_redis_unavailable error_class=%s; using in-memory limiter", type(exc).__name__)
    return _memory_hit(key, limit, window)


def _enforce(key: str, limit: int) -> None:
    if limit <= 0:
        return
    wait = _hit(key, limit)
    if wait:
        raise HTTPException(429, "Too many requests; please wait before trying again.",
                            headers={"Retry-After": str(wait)})


def rate_limit_auth(request: Request) -> None:
    _enforce(f"auth:{request.url.path}:{client_ip(request)}", get_settings().auth_rate_limit_per_minute)


def rate_limit_user(scope: str, setting_name: str) -> Callable:
    """Dependency factory: per-authenticated-user limit read from settings.<setting_name> at request time."""
    from app.services.auth import get_current_user

    def _dep(user: dict = Depends(get_current_user)) -> None:
        uid = str(user.get("id") or user.get("sub") or "anonymous")
        _enforce(f"user:{scope}:{uid}", int(getattr(get_settings(), setting_name, 0)))

    return _dep
