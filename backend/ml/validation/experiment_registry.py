"""
Machine-readable experiment registry (Phase 6, workstream 3/9).

One record per experiment run in this project, including failed, blocked, synthetic, planning and NOT_RUN runs.  Nothing is
deleted from history: a failed or blocked experiment stays listed with its output file.  The registry is *derived* from the
committed result files (status, evidence label, protocol id/hash, SHA-256 of every output) and checked against them, so it cannot
claim a status the file does not carry.

Evidence classes (the Phase 6 claim vocabulary) -- an experiment's class is derived from its output file, never typed in alone:

    VERIFIED_INTERNAL_EVIDENCE   completed on data authenticated against the shipped artifact       (none exists yet)
    VERIFIED_EXTERNAL_EVIDENCE   completed on an independent cohort with a frozen model              (none exists)
    EXPLORATORY                  completed but outside the registered protocol or on unauthenticated data
    SYNTHETIC_ENGINEERING_TEST   synthetic fixtures; proves the code path only
    PLANNING_ESTIMATE            design-stage arithmetic or simulation
    NOT_RUN                      required input missing; fail-closed record
    BLOCKED                      could not run for a reason outside the code (access)
    FAILED                       ran and failed
    METADATA_CONSISTENCY_ONLY    checks the shipped artifacts' recorded numbers for self-consistency; produces no performance estimate

CLI:  PYTHONPATH=. python -m ml.validation.experiment_registry build   # rewrite experiment_registry.json
      PYTHONPATH=. python -m ml.validation.experiment_registry check    # exit 0 consistent, 1 drift or invalid
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
REGISTRY_PATH = Path(__file__).resolve().parent / "experiment_registry.json"
SCHEMA = "neuro-mas-experiment-registry/1"

CLASSES = ("VERIFIED_INTERNAL_EVIDENCE", "VERIFIED_EXTERNAL_EVIDENCE", "EXPLORATORY", "SYNTHETIC_ENGINEERING_TEST",
           "PLANNING_ESTIMATE", "NOT_RUN", "BLOCKED", "FAILED", "METADATA_CONSISTENCY_ONLY")
VERIFIED = ("VERIFIED_INTERNAL_EVIDENCE", "VERIFIED_EXTERNAL_EVIDENCE")

R = "ml/validation/results"
_WF = "PYTHONPATH=. python -m ml.validation.workflow run"

# (id, phase, title, command, primary output, other outputs, note, allowed derived classes)
EXPERIMENTS: list[dict] = [
    dict(id="E-P2-FETCH", phase="2", title="Download of UCI 174 / 189 through the session proxy",
         command="PYTHONPATH=. python -m ml.validation.validate_models fetch",
         primary=f"{R}/fetch_FAILED.json", outputs=[], allowed=["BLOCKED", "FAILED"],
         note="HTTP 403 from the egress proxy. Retained as the record of the failed access attempt; not retried against policy."),
    dict(id="E-P2-ARTIFACTS", phase="2", title="Inspection of the shipped model artifacts (recorded metadata)",
         command="PYTHONPATH=. python -m ml.validation.validate_models artifacts",
         primary=f"{R}/artifacts.json", outputs=[], allowed=["METADATA_CONSISTENCY_ONLY"],
         note="Reads the shipped artifacts; no dataset used."),
    dict(id="E-P4A-ARTIFACT-AUDIT", phase="4A", title="No-data audit of shipped artifacts (historical figures vs recorded counts)",
         command="PYTHONPATH=. python -m ml.validation.phase4a artifacts --out ml/validation/results/phase4a",
         primary=f"{R}/phase4a/artifact_audit.json", outputs=[], allowed=["METADATA_CONSISTENCY_ONLY"],
         note="Historical metrics are internally consistent only; none is reproduced."),
    dict(id="E-P4A-SELFTEST", phase="4A", title="Harness integrity self-test on synthetic scenarios",
         command="PYTHONPATH=. python -m ml.validation.phase4a selftest --out ml/validation/results/phase4a",
         primary=f"{R}/phase4a/selftest_SYNTHETIC_NOT_EVIDENCE.json", outputs=[], allowed=["SYNTHETIC_ENGINEERING_TEST"],
         note="Proves the validation code detects planted leakage; says nothing about the real data or models."),
    dict(id="E-P4A-DISEASE", phase="4A", title="Registered p4a-1 disease validation",
         command="PYTHONPATH=. python -m ml.validation.phase4a run --kind disease --csv <parkinsons.data> --out ml/validation/results/phase4a",
         primary=f"{R}/phase4a/disease_NOT_RUN.json", outputs=[], allowed=["NOT_RUN"], note="No data file."),
    dict(id="E-P4A-RISK", phase="4A", title="Registered p4a-1 risk validation",
         command="PYTHONPATH=. python -m ml.validation.phase4a run --kind risk --csv <parkinsons_updrs.data> --out ml/validation/results/phase4a",
         primary=f"{R}/phase4a/risk_NOT_RUN.json", outputs=[], allowed=["NOT_RUN"], note="No data file."),
    dict(id="E-P5-POWER", phase="5", title="Design-stage precision and power planning",
         command="PYTHONPATH=. python -m ml.validation.power_planning --reps 20 --out ml/validation/results/phase5/power_planning.json",
         primary=f"{R}/phase5/power_planning.json", outputs=[], allowed=["PLANNING_ESTIMATE"],
         note="Simulated generator with an arbitrary signal unit; not evidence about the real data."),
    dict(id="E-P5-WORKFLOW", phase="5", title="Registered workflow, no data",
         command=f"{_WF} --out ml/validation/results/phase5/workflow_no_data",
         primary=f"{R}/phase5/workflow_no_data/experiment_manifest.json",
         outputs=[f"{R}/phase5/workflow_no_data/disease_NOT_RUN.json", f"{R}/phase5/workflow_no_data/risk_NOT_RUN.json",
                  f"{R}/phase5/workflow_no_data/GENERATED_RESULTS.md", f"{R}/phase5/workflow_no_data/GENERATED_RESULTS.claims.json"],
         allowed=["NOT_RUN"], note="Exit code 2."),
    dict(id="E-P6-BASELINE", phase="6", title="Registered workflow p4a-1, no data (Phase 6 re-run)",
         command=f"{_WF} --out ml/validation/results/phase6/baseline_no_data",
         primary=f"{R}/phase6/baseline_no_data/experiment_manifest.json",
         outputs=[f"{R}/phase6/baseline_no_data/disease_NOT_RUN.json", f"{R}/phase6/baseline_no_data/risk_NOT_RUN.json",
                  f"{R}/phase6/baseline_no_data/GENERATED_RESULTS.md", f"{R}/phase6/baseline_no_data/GENERATED_RESULTS.claims.json"],
         allowed=["NOT_RUN"], note="Exit code 2; manifest records commit 6a9fdf6 and git_dirty false."),
    dict(id="E-P6-EXTENSION", phase="6", title="Prespecified extension p6-1-ext, no data",
         command=f"{_WF} --candidate-set p6-1-ext --out ml/validation/results/phase6/extension_no_data",
         primary=f"{R}/phase6/extension_no_data/experiment_manifest.json",
         outputs=[f"{R}/phase6/extension_no_data/disease_NOT_RUN.json", f"{R}/phase6/extension_no_data/risk_NOT_RUN.json"],
         allowed=["NOT_RUN"], note="Exit code 2. Exploratory by protocol even when data exist."),
    dict(id="E-P6-REPORT", phase="6", title="Phase 6 report pipeline, no data",
         command="PYTHONPATH=. python -m ml.validation.phase6_report --p4a-dir ml/validation/results/phase6/baseline_no_data "
                 "--out ml/validation/results/phase6/report_no_data",
         primary=f"{R}/phase6/report_no_data/phase6_NOT_RUN.json", outputs=[], allowed=["NOT_RUN"], note="Exit code 2."),
    dict(id="E-P6-VOICE-SAMPLE-SIZE", phase="6", title="Voice-value sample-size simulation",
         command="PYTHONPATH=. python -m ml.validation.power_planning --sample-size --out ml/validation/results/phase6/voice_sample_size_planning.json",
         primary=f"{R}/phase6/voice_sample_size_planning.json", outputs=[], allowed=["PLANNING_ESTIMATE"],
         note="12 repetitions per cell, single reference, independent-within-patient voice signal: optimistic planning figure."),
]


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def derive_class(doc: dict, allowed: list[str]) -> str:
    """Class implied by the output file itself.  The declared 'allowed' set only constrains it; it never replaces it."""
    status = doc.get("status") or doc.get("overall_status")
    ev = str(doc.get("evidence_status") or "")
    if status in ("not_run",):
        return "NOT_RUN"
    if status in ("failed", "refused"):
        return "BLOCKED" if "BLOCKED" in allowed else "FAILED"
    if ev.startswith("SYNTHETIC_FIXTURE"):
        return "SYNTHETIC_ENGINEERING_TEST"
    if ev.startswith("DESIGN_STAGE_PLANNING"):
        return "PLANNING_ESTIMATE"
    if ev in ("REAL_ARTIFACTS_NO_DATA_USED", "REAL_RUN_ON_REPO_FILES"):
        return "METADATA_CONSISTENCY_ONLY"
    if status == "completed" and ev.startswith("REAL_DATA_AUTHENTICATED_AGAINST_SHIPPED_ARTIFACT"):
        if any(t in ev for t in ("+FAST_NOT_REGISTERED_PROTOCOL", "+EXPLORATORY_EXTENSION_P6_1", "UNAUTHENTICATED")):
            return "EXPLORATORY"
        return "VERIFIED_INTERNAL_EVIDENCE"
    if status == "completed":
        return "EXPLORATORY"
    return "FAILED"


def _git_commit(rel: str) -> str:
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%h", "--", rel], cwd=BACKEND, capture_output=True, text=True, timeout=30).stdout.strip()
        return out or "UNCOMMITTED"
    except Exception:       # git unavailable: say so, do not invent a value
        return "UNKNOWN"


def build_records(*, with_git: bool = True) -> list[dict]:
    recs = []
    for e in EXPERIMENTS:
        prim = BACKEND / e["primary"]
        if not prim.exists():
            raise FileNotFoundError(f"registry entry {e['id']}: primary output {e['primary']} is missing (history must not be deleted)")
        doc = json.loads(prim.read_text())
        cls = derive_class(doc, e["allowed"])
        if cls not in e["allowed"]:
            raise ValueError(f"{e['id']}: output implies class {cls}, outside the declared set {e['allowed']}")
        outs = []
        for rel in [e["primary"], *e["outputs"]]:
            p = BACKEND / rel
            if not p.exists():
                raise FileNotFoundError(f"registry entry {e['id']}: output {rel} is missing")
            outs.append({"path": rel, "sha256": _sha256(p), "bytes": p.stat().st_size,
                         **({"git_commit": _git_commit(rel)} if with_git else {})})
        recs.append({"id": e["id"], "phase": e["phase"], "title": e["title"], "command": e["command"],
                     "status": doc.get("status") or doc.get("overall_status"), "evidence_status_in_file": doc.get("evidence_status"),
                     "evidence_class": cls, "clinically_validated": bool(doc.get("clinically_validated", False)),
                     "protocol_id": doc.get("protocol_id"), "protocol_sha256": doc.get("protocol_sha256"),
                     "outputs": outs, "note": e["note"]})
    return recs


def build() -> dict:
    return {"registry_schema": SCHEMA, "classes": list(CLASSES),
            "rule": "An experiment's class is derived from its output file. No entry may be VERIFIED_* unless the output carries "
                    "the REAL_DATA_AUTHENTICATED_AGAINST_SHIPPED_ARTIFACT label and status completed. Failed, blocked, synthetic and "
                    "NOT_RUN runs stay listed.",
            "experiments": build_records()}


def check(path: Path = REGISTRY_PATH) -> list[str]:
    """Problems found; empty means the registry matches the files.  git_commit is informational and not compared."""
    problems: list[str] = []
    if not path.exists():
        return [f"{path} does not exist; run `build`"]
    saved = json.loads(path.read_text())
    if saved.get("registry_schema") != SCHEMA:
        problems.append("registry_schema mismatch")
    try:
        fresh = build_records(with_git=False)
    except Exception as e:      # missing file or class outside the declared set
        return [f"{type(e).__name__}: {e}"]
    by_id = {r["id"]: r for r in saved.get("experiments", [])}
    if set(by_id) != {r["id"] for r in fresh}:
        problems.append(f"experiment ids differ: saved {sorted(by_id)} vs code {sorted(r['id'] for r in fresh)}")
    for r in fresh:
        s = by_id.get(r["id"])
        if not s:
            continue
        for k in ("phase", "title", "command", "status", "evidence_status_in_file", "evidence_class", "clinically_validated",
                  "protocol_id", "protocol_sha256", "note"):
            if s.get(k) != r[k]:
                problems.append(f"{r['id']}.{k}: saved {s.get(k)!r} != file {r[k]!r}")
        so = {o["path"]: o for o in s.get("outputs", [])}
        for o in r["outputs"]:
            if o["path"] not in so:
                problems.append(f"{r['id']}: output {o['path']} not recorded")
            elif so[o["path"]]["sha256"] != o["sha256"]:
                problems.append(f"{r['id']}: SHA-256 drift for {o['path']}")
        if set(so) != {o["path"] for o in r["outputs"]}:
            problems.append(f"{r['id']}: output set differs")
    for s in saved.get("experiments", []):
        if s.get("evidence_class") in VERIFIED and "REAL_DATA_AUTHENTICATED_AGAINST_SHIPPED_ARTIFACT" not in str(s.get("evidence_status_in_file")):
            problems.append(f"{s['id']}: VERIFIED class without an authenticated-real-data label")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("CLI:")[0])
    ap.add_argument("action", choices=["build", "check"])
    a = ap.parse_args(argv)
    if a.action == "build":
        REGISTRY_PATH.write_text(json.dumps(build(), indent=1, sort_keys=False) + "\n")
        print(f"wrote {REGISTRY_PATH}")
        return 0
    probs = check()
    for p in probs:
        print("DRIFT:", p, file=sys.stderr)
    print("registry consistent" if not probs else f"{len(probs)} problem(s)")
    return 1 if probs else 0


if __name__ == "__main__":
    raise SystemExit(main())
