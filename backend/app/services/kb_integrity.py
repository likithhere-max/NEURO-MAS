"""
Knowledge-base citation integrity (Phase 2, E2): structure checks, link liveness, and a human review queue.

What this does NOT do: it never edits `data/diseases_curated.json` and never rewrites a clinical statement. It reports
problems and queues them for a clinician/curator; the queue file keeps reviewer decisions across re-runs.

A link's liveness says only that a URL answered. It does not say the page supports the claim (no passage-level
verification exists - every record is flagged for that), and a blocked/unreachable URL is "unverifiable", never "dead".
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Awaitable, Callable
from urllib.parse import urlparse

Fetch = Callable[[str, float], "Awaitable[FetchResult]"]


@dataclass
class FetchResult:
    status: int | None = None            # HTTP status, None on network error
    location: str | None = None          # Location header for 3xx
    error: str | None = None             # exception class name on failure (no URL/secret content)


async def default_fetch(url: str, timeout: float) -> FetchResult:
    """HEAD request without following redirects (so redirects can be classified). Runs in a worker thread."""
    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None

    def go() -> FetchResult:
        opener = urllib.request.build_opener(_NoRedirect)
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "neuro-mas-link-check/1.0 (contact: project owner)"})
        try:
            with opener.open(req, timeout=timeout) as r:
                return FetchResult(status=r.status)
        except urllib.error.HTTPError as e:
            return FetchResult(status=e.code, location=e.headers.get("Location") if e.headers else None)
        except Exception as e:  # DNS failure, TLS error, timeout, refused...
            return FetchResult(error=type(e).__name__)

    return await asyncio.to_thread(go)


# ----------------------------------------------------------------------------------------------- classification
LIVE, REDIRECTED, DEAD, BLOCKED, SERVER_ERROR, UNREACHABLE, INVALID = (
    "live", "redirected", "dead", "blocked_unverifiable", "server_error", "unreachable", "invalid_url")


def classify(url: str, r: FetchResult) -> tuple[str, str]:
    if r.error:
        return UNREACHABLE, f"network error ({r.error}); cannot verify - NOT evidence the page is gone"
    s = r.status or 0
    if 200 <= s < 300:
        return LIVE, f"HTTP {s}"
    if 300 <= s < 400:
        same_host = bool(r.location) and urlparse(r.location).netloc in ("", urlparse(url).netloc)
        return REDIRECTED, f"HTTP {s} to {'same host' if same_host else 'a different host'}: {r.location or 'no Location'}"
    if s in (404, 410):
        return DEAD, f"HTTP {s}"
    if s in (401, 403, 405, 429):
        return BLOCKED, f"HTTP {s}: server refused an automated request; verify manually"
    if s >= 500:
        return SERVER_ERROR, f"HTTP {s}: transient? re-check later"
    return UNREACHABLE, f"unexpected HTTP {s}"


@dataclass
class Citation:
    disease_id: str
    name: str
    url: str
    record_last_verified: str | None


def collect_citations(kb: dict) -> list[Citation]:
    out = []
    for d in kb.get("diseases", []):
        for u in d.get("source_urls") or []:
            out.append(Citation(d.get("disease_id", "?"), d.get("preferred_name", "?"), str(u), d.get("last_verified")))
    return out


def structural_issues(kb: dict, today: date | None = None, stale_days: int = 180) -> list[dict]:
    """Offline checks, deterministic and needing no network."""
    today = today or datetime.now(timezone.utc).date()
    issues: list[dict] = []
    seen: dict[tuple, int] = {}
    for d in kb.get("diseases", []):
        did, name = d.get("disease_id", "?"), d.get("preferred_name", "?")
        urls = d.get("source_urls") or []
        if not urls:
            issues.append(_issue(did, name, None, "no_source_url", "high", "Record makes clinical statements with no cited source."))
        for u in urls:
            p = urlparse(str(u))
            if p.scheme != "https" or not p.netloc:
                issues.append(_issue(did, name, u, "invalid_or_non_https_url", "high", "Citation is not a well-formed https URL."))
            seen[(did, u)] = seen.get((did, u), 0) + 1
        lv = d.get("last_verified")
        try:
            age = (today - date.fromisoformat(str(lv))).days
            if age > stale_days:
                issues.append(_issue(did, name, None, "stale_verification", "medium", f"last_verified {lv} is {age} days old."))
        except (TypeError, ValueError):
            issues.append(_issue(did, name, None, "missing_or_unparseable_last_verified", "medium", f"last_verified={lv!r}"))
        # Every record: citations are record-level, not tied to individual claims.
        issues.append(_issue(did, name, None, "no_claim_level_traceability", "info",
                             "Definition/symptoms/red flags are not linked to specific passages of the cited sources; "
                             "a reviewer must confirm each claim against the source."))
        if str(d.get("orpha_code", "")).upper().startswith("PENDING"):
            issues.append(_issue(did, name, None, "identifier_pending_verification", "info", "orpha_code is PENDING_VERIFICATION."))
    issues += [_issue("?", "?", u, "duplicate_citation", "low", f"cited {n} times in one record") for (_, u), n in seen.items() if n > 1]
    return issues


def _issue(did, name, url, kind, priority, detail) -> dict:
    key = hashlib.sha1(f"{did}|{url}|{kind}".encode()).hexdigest()[:12]
    return {"id": key, "disease_id": did, "name": name, "url": url, "kind": kind, "priority": priority, "detail": detail}


# ----------------------------------------------------------------------------------------------- link checking
class RateLimiter:
    """At most one request per host per `min_interval` seconds, plus a global concurrency cap."""

    def __init__(self, min_interval: float = 1.0, max_concurrency: int = 4, clock=time.monotonic, sleep=asyncio.sleep):
        self.min_interval, self._clock, self._sleep = min_interval, clock, sleep
        self._next: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self.sem = asyncio.Semaphore(max_concurrency)

    async def wait(self, host: str) -> None:
        lock = self._locks.setdefault(host, asyncio.Lock())
        async with lock:
            now = self._clock()
            ready = self._next.get(host, 0.0)
            if ready > now:
                await self._sleep(ready - now)
                now = ready
            self._next[host] = now + self.min_interval


async def check_links(citations: list[Citation], *, fetch: Fetch = default_fetch, timeout: float = 10.0,
                      limiter: RateLimiter | None = None, retries: int = 1) -> list[dict]:
    limiter = limiter or RateLimiter()
    cache: dict[str, dict] = {}                 # one request per distinct URL

    async def one(url: str) -> dict:
        p = urlparse(url)
        if p.scheme not in ("http", "https") or not p.netloc:
            return {"url": url, "state": INVALID, "detail": "not an http(s) URL", "attempts": 0}
        attempts, res = 0, FetchResult(error="NotAttempted")
        async with limiter.sem:
            for _ in range(retries + 1):
                await limiter.wait(p.netloc)
                attempts += 1
                try:
                    res = await asyncio.wait_for(fetch(url, timeout), timeout=timeout + 5)
                except asyncio.TimeoutError:
                    res = FetchResult(error="TimeoutError")
                except Exception as exc:
                    res = FetchResult(error=type(exc).__name__)
                state, _ = classify(url, res)
                if state not in (SERVER_ERROR, UNREACHABLE):
                    break
        state, detail = classify(url, res)
        return {"url": url, "state": state, "detail": detail, "attempts": attempts,
                "checked_at": datetime.now(timezone.utc).isoformat()}

    urls = sorted({c.url for c in citations})
    for r in await asyncio.gather(*(one(u) for u in urls)):
        cache[r["url"]] = r
    return [{"disease_id": c.disease_id, "name": c.name, **cache[c.url]} for c in citations]


# ----------------------------------------------------------------------------------------------- review queue
LINK_PRIORITY = {DEAD: "high", REDIRECTED: "medium", INVALID: "high", BLOCKED: "low", SERVER_ERROR: "low", UNREACHABLE: "low"}
LINK_ACTION = {
    DEAD: "Find the current location of the source or choose a replacement; re-verify every claim that relied on it.",
    REDIRECTED: "Confirm the redirect target is the same document, then update the stored URL (reviewer action).",
    INVALID: "Correct the URL.",
    BLOCKED: "Open manually in a browser; the server refuses automated checks.",
    SERVER_ERROR: "Re-run the check later.",
    UNREACHABLE: "Re-run from a network that can reach the site; do not treat as dead.",
}


def build_queue(structural: list[dict], links: list[dict], previous: list[dict] | None = None) -> list[dict]:
    items = list(structural)
    for l in links:
        if l["state"] != LIVE:
            it = _issue(l["disease_id"], l["name"], l["url"], f"link_{l['state']}", LINK_PRIORITY[l["state"]], l["detail"])
            it["suggested_action"] = LINK_ACTION[l["state"]]
            items.append(it)
    prev = {p["id"]: p for p in (previous or [])}
    out = []
    for it in sorted(items, key=lambda x: (["high", "medium", "low", "info"].index(x["priority"]), x["disease_id"], x["kind"])):
        old = prev.get(it["id"], {})
        it["status"] = old.get("status", "open")              # reviewer decisions survive re-runs
        for k in ("reviewed_by", "reviewed_at", "reviewer_note"):
            if k in old:
                it[k] = old[k]
        out.append(it)
    return out


def summarize(links: list[dict]) -> dict:
    out: dict[str, int] = {}
    for l in links:
        out[l["state"]] = out.get(l["state"], 0) + 1
    return out
