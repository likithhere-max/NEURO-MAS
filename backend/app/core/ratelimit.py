"""Tiny in-memory sliding-window rate limiter for unauthenticated endpoints.

Per-process only (not shared across workers/instances) and keyed on the socket peer address, so
behind a reverse proxy/tunnel all clients may share one key. It blunts casual brute force and
signup spam; it is NOT a substitute for an edge/WAF limiter (see docs/audit/OUTSTANDING_ISSUES.md).
"""
from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

from app.core.config import get_settings

_hits: dict[str, deque] = defaultdict(deque)


def reset() -> None:
    _hits.clear()


def rate_limit_auth(request: Request) -> None:
    limit = get_settings().auth_rate_limit_per_minute
    if limit <= 0:
        return
    key = f"{request.url.path}:{request.client.host if request.client else 'unknown'}"
    now = time.monotonic()
    q = _hits[key]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(429, "too many attempts; try again in a minute")
    q.append(now)
