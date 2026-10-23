import asyncio
import json
from datetime import date

import pytest

from app.services import kb_integrity as ki
from app.services.kb_integrity import FetchResult

KB = {"diseases": [
    {"disease_id": "D1", "preferred_name": "One", "source_urls": ["https://a.example/x", "https://a.example/x"],
     "last_verified": "2026-08-17", "orpha_code": "PENDING_VERIFICATION"},
    {"disease_id": "D2", "preferred_name": "Two", "source_urls": [], "last_verified": "2025-01-01"},
    {"disease_id": "D3", "preferred_name": "Three", "source_urls": ["http://insecure.example/p", "not a url"], "last_verified": "bad"},
]}


def kinds(issues, did=None):
    return {i["kind"] for i in issues if did is None or i["disease_id"] == did}


def test_structural_checks_find_each_problem_class():
    iss = ki.structural_issues(KB, today=date(2026, 10, 23))
    assert "no_source_url" in kinds(iss, "D2") and "stale_verification" in kinds(iss, "D2")
    assert "invalid_or_non_https_url" in kinds(iss, "D3") and "missing_or_unparseable_last_verified" in kinds(iss, "D3")
    assert "duplicate_citation" in kinds(iss) and "identifier_pending_verification" in kinds(iss, "D1")
    assert all("no_claim_level_traceability" in kinds(iss, d) for d in ("D1", "D2", "D3"))   # honest limitation, every record


@pytest.mark.parametrize("res,state", [
    (FetchResult(status=200), ki.LIVE), (FetchResult(status=301, location="https://b.example/new"), ki.REDIRECTED),
    (FetchResult(status=404), ki.DEAD), (FetchResult(status=410), ki.DEAD), (FetchResult(status=403), ki.BLOCKED),
    (FetchResult(status=429), ki.BLOCKED), (FetchResult(status=503), ki.SERVER_ERROR),
    (FetchResult(error="TimeoutError"), ki.UNREACHABLE), (FetchResult(error="gaierror"), ki.UNREACHABLE),
])
def test_classification_never_calls_unverifiable_links_dead(res, state):
    assert ki.classify("https://a.example/x", res)[0] == state


class Clock:
    def __init__(self): self.t, self.sleeps = 0.0, []
    def now(self): return self.t
    async def sleep(self, d): self.sleeps.append(round(d, 3)); self.t += d


@pytest.mark.asyncio
async def test_rate_limiter_spaces_requests_per_host():
    c = Clock()
    rl = ki.RateLimiter(min_interval=2.0, clock=c.now, sleep=c.sleep)
    for _ in range(3):
        await rl.wait("a.example")
    await rl.wait("b.example")                 # another host is not delayed by a.example
    assert c.sleeps == [2.0, 2.0]


@pytest.mark.asyncio
async def test_each_distinct_url_is_fetched_once_and_retries_are_bounded():
    calls = []

    async def fetch(url, timeout):
        calls.append(url)
        return FetchResult(error="TimeoutError") if "flaky" in url else FetchResult(status=200)

    cites = [ki.Citation("D1", "n", "https://a.example/ok", None), ki.Citation("D2", "n", "https://a.example/ok", None),
             ki.Citation("D3", "n", "https://a.example/flaky", None)]
    c = Clock()
    out = await ki.check_links(cites, fetch=fetch, limiter=ki.RateLimiter(0.0, 2, c.now, c.sleep), retries=2)
    assert calls.count("https://a.example/ok") == 1 and calls.count("https://a.example/flaky") == 3
    st = {(o["disease_id"]): o["state"] for o in out}
    assert st == {"D1": "live", "D2": "live", "D3": "unreachable"}
    assert [o["attempts"] for o in out if o["disease_id"] == "D3"] == [3]


@pytest.mark.asyncio
async def test_fetch_exception_and_invalid_url_are_contained():
    async def fetch(url, timeout):
        raise RuntimeError("secret-bearing message")
    out = await ki.check_links([ki.Citation("D1", "n", "https://a.example/x", None), ki.Citation("D2", "n", "ftp://x", None)],
                               fetch=fetch, limiter=ki.RateLimiter(0.0), retries=0)
    assert out[0]["state"] == "unreachable" and "secret" not in json.dumps(out)
    assert out[1]["state"] == "invalid_url"


def test_queue_orders_by_priority_and_preserves_reviewer_decisions():
    structural = ki.structural_issues(KB, today=date(2026, 10, 23))
    links = [{"disease_id": "D1", "name": "One", "url": "https://a.example/x", "state": "dead", "detail": "HTTP 404"}]
    q1 = ki.build_queue(structural, links)
    assert q1[0]["priority"] == "high" and all(i["status"] == "open" for i in q1)
    dead = [i for i in q1 if i["kind"] == "link_dead"][0]
    assert "re-verify every claim" in dead["suggested_action"]
    dead.update(status="resolved", reviewed_by="Dr X", reviewer_note="replaced")
    q2 = ki.build_queue(structural, links, previous=q1)
    again = [i for i in q2 if i["id"] == dead["id"]][0]
    assert again["status"] == "resolved" and again["reviewed_by"] == "Dr X"


def test_live_links_create_no_queue_item():
    q = ki.build_queue([], [{"disease_id": "D1", "name": "n", "url": "https://a.example/x", "state": "live", "detail": "HTTP 200"}])
    assert q == []


def test_real_kb_offline_scan_runs_and_flags_traceability(tmp_path):
    import app.services.knowledge_base as kbmod
    kb = json.loads(kbmod.DATA_PATH.read_text())
    iss = ki.structural_issues(kb)
    assert sum(1 for i in iss if i["kind"] == "no_claim_level_traceability") == len(kb["diseases"])
    assert not [i for i in iss if i["kind"] in ("no_source_url", "invalid_or_non_https_url")]   # shipped KB is well-formed
