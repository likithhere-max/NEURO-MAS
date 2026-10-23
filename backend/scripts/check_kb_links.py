"""Check knowledge-base citations. Run from backend/:  python -m scripts.check_kb_links [--offline] [--out PATH]

--offline : structural checks only (no network). Default also probes each distinct URL (rate limited, HEAD).
Writes a JSON report + review queue; NEVER modifies data/diseases_curated.json."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.services import kb_integrity as ki

KB = Path(__file__).resolve().parent.parent / "data" / "diseases_curated.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data" / "kb_review_queue.json"))
    ap.add_argument("--interval", type=float, default=1.0)
    a = ap.parse_args()
    kb = json.loads(KB.read_text())
    structural = ki.structural_issues(kb)
    links: list[dict] = []
    if not a.offline:
        links = asyncio.run(ki.check_links(ki.collect_citations(kb), limiter=ki.RateLimiter(min_interval=a.interval)))
    out = Path(a.out)
    prev = json.loads(out.read_text()).get("queue") if out.exists() else None
    queue = ki.build_queue(structural, links, prev)
    report = {"generated": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
              "mode": "offline" if a.offline else "online", "link_summary": ki.summarize(links), "queue": queue,
              "note": "liveness != claim support; unreachable/blocked are unverifiable, not dead."}
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({"mode": report["mode"], "link_summary": report["link_summary"], "queue_items": len(queue), "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
