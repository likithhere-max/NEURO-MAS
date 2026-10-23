"""
Claims register checker (Phase 6, workstream 9).

`claims_register.json` is a hand-written list of every claim this project makes about its models, data and process, each with an
evidence class from the Phase 6 vocabulary and the evidence it rests on.  This module does not decide whether a claim is true; it
enforces that a claim cannot be classed higher than its evidence:

* a class must come from the vocabulary;
* a VERIFIED_* class needs a referenced experiment of that same class (derived from the experiment's output file, see
  `experiment_registry`);
* NOT_RUN / PLANNING_ESTIMATE / SYNTHETIC_ENGINEERING_TEST / METADATA_CONSISTENCY_ONLY claims may only reference experiments of that class;
* referenced experiments, files and tests must exist;
* a claim text may not assert clinical validity, external or prospective validation without negation.

CLI:  PYTHONPATH=. python -m ml.validation.claims_register check
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from ml.validation import experiment_registry as ER

HERE = Path(__file__).resolve().parent
REGISTER_PATH = HERE / "claims_register.json"
BACKEND = ER.BACKEND
REPO = BACKEND.parent
CLASSES = ER.CLASSES
BANNED = re.compile(r"\b(clinically (validated|ready|proven)|externally validated|prospectively validated)\b", re.I)
NEGATIONS = ("not", "no ", "never", "cannot", "neither", "nothing", "without", "none")


def _experiments() -> dict[str, dict]:
    return {r["id"]: r for r in json.loads(ER.REGISTRY_PATH.read_text())["experiments"]}


def check(path: Path = REGISTER_PATH, report: Path | None = None) -> list[str]:
    problems: list[str] = []
    reg = json.loads(path.read_text())
    exps = _experiments()
    seen = set()
    for c in reg["claims"]:
        cid = c["id"]
        if cid in seen:
            problems.append(f"{cid}: duplicate id")
        seen.add(cid)
        cls = c["class"]
        if cls not in CLASSES:
            problems.append(f"{cid}: unknown class {cls}")
            continue
        ev = c.get("evidence", [])
        exp_refs = [e["ref"] for e in ev if e["type"] == "experiment"]
        for ref in exp_refs:
            if ref not in exps:
                problems.append(f"{cid}: experiment {ref} not in the experiment registry")
        classes_of_refs = {exps[r]["evidence_class"] for r in exp_refs if r in exps}
        if cls in ER.VERIFIED:
            if cls not in classes_of_refs:
                problems.append(f"{cid}: class {cls} without a referenced experiment of that class")
        elif cls in ("NOT_RUN", "PLANNING_ESTIMATE", "SYNTHETIC_ENGINEERING_TEST", "METADATA_CONSISTENCY_ONLY"):
            wrong = classes_of_refs - {cls}
            if wrong:
                problems.append(f"{cid}: class {cls} references experiments of class {sorted(wrong)}")
        if any(k in classes_of_refs for k in ER.VERIFIED) and cls not in ER.VERIFIED:
            problems.append(f"{cid}: references VERIFIED experiment evidence but is classed {cls}")
        for e in ev:
            if e["type"] == "file" and not (REPO / e["ref"]).exists():
                problems.append(f"{cid}: file {e['ref']} missing")
            if e["type"] == "test":
                f, _, name = e["ref"].partition("::")
                fp = BACKEND / f
                if not fp.exists() or (name and f"def {name}" not in fp.read_text()):
                    problems.append(f"{cid}: test {e['ref']} not found")
        if BANNED.search(c["claim"]) and not any(w in c["claim"].lower() for w in NEGATIONS):
            problems.append(f"{cid}: asserts clinical/external validity without negation")
        if not c.get("claim") or not c.get("status_note"):
            problems.append(f"{cid}: claim and status_note are required")
    if report is not None:
        text = report.read_text()
        for cid in seen:
            if cid not in text:
                problems.append(f"{cid}: not listed in {report.name}")
    return problems


def main(argv=None) -> int:
    probs = check(report=REPO / "docs/phase6/FINAL_READINESS_REPORT.md")
    for p in probs:
        print("PROBLEM:", p, file=sys.stderr)
    print("claims register consistent" if not probs else f"{len(probs)} problem(s)")
    return 1 if probs else 0


if __name__ == "__main__":
    raise SystemExit(main())
