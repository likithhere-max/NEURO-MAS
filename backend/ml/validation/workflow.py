"""
One documented workflow for AUTHENTICATED-data validation (Phase 5, Stage 7 / "Phase J").

    cd backend
    PYTHONPATH=. python -m ml.validation.workflow run --disease-csv parkinsons.data --risk-csv parkinsons_updrs.data --out ml/validation/results/workflow
    PYTHONPATH=. python -m ml.validation.workflow report --out ml/validation/results/workflow       # re-render + re-verify the report

Per dataset it: resolves the file -> checks the dataset manifest (hash) -> authenticates the file against the shipped artifact ->
validates schema/counts -> writes the exclusion/data-quality report -> runs the pre-registered protocol (ml.validation.phase4a)
-> saves fold assignments and held-out predictions under <out>/protected/ -> writes the experiment manifest -> renders the results
section FROM the JSON and verifies every displayed number against it.

Statuses are never merged into a success:  completed | not_run | refused | failed.
Exit code:  0 every mandatory experiment completed
            1 an experiment raised                       (status failed)
            2 a mandatory input is missing / manifest unpinned-and-nothing-run   (status not_run)  <- CI-visible, never green
            3 an integrity check refused the data        (hash mismatch, file is not the training data, schema/counts)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

from ml.validation import data as D
from ml.validation import dataset_manifest as DM
from ml.validation import exclusions as E
from ml.validation import fingerprint as F
from ml.validation import phase4a as P
from ml.validation import report_from_artifacts as R

WORKFLOW_VERSION = "1.0.0"
DEFAULT_OUT = Path(__file__).resolve().parent / "results" / "workflow"
NAMES = {"disease": "parkinsons", "risk": "parkinsons_updrs"}
COMPLETED, NOT_RUN, REFUSED, FAILED = "completed", "not_run", "refused", "failed"


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=Path(__file__).resolve().parent, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return None


def _dirty(out_dir: Path | None) -> str | None:
    """`git status --porcelain`, ignoring the run's own output directory (its files are produced by this very run and must not
    make the manifest report an uncommitted *source* tree).  Anything else uncommitted still counts."""
    top = _git("rev-parse", "--show-toplevel")
    spec = ["--", ":(top)."]
    if out_dir is not None and top:
        try:
            rel = Path(out_dir).resolve().relative_to(Path(top).resolve())
            spec.append(f":(top,exclude){rel.as_posix()}")
        except ValueError:
            pass
    return _git("status", "--porcelain", *spec)


def run_context(argv: list[str], out_dir: Path | None = None) -> dict:
    dirty = _dirty(out_dir)
    return {"workflow_version": WORKFLOW_VERSION, "git_commit": _git("rev-parse", "HEAD"), "git_dirty": None if dirty is None else bool(dirty),
            "argv": argv, "python_executable": sys.executable, "environment": P.env_info(), "protocol_id": P.PROTOCOL["id"],
            "protocol_sha256": P.PROTOCOL_SHA256, "seed_base": P.PROTOCOL["seed_base"]}


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Recorder:
    """Collects fold assignments and held-out predictions from the runner (observe-only)."""

    def __init__(self):
        self.rows, self.folds = [], []

    def __call__(self, tag, seed, y, groups, res):
        fold_of, chosen_of = {}, {}
        for f in res.folds:
            for grp in f.test_groups:
                fold_of[grp], chosen_of[grp] = f.fold, f.chosen
        g = pd.Series(groups).reset_index(drop=True)
        self.rows.append(pd.DataFrame({"kind": tag[0], "model_tag": tag[1], "seed": seed, "row": range(len(g)), "subject": g.astype(str),
                                       "y": pd.Series(y).reset_index(drop=True), "p": res.oof,
                                       "outer_fold": g.map(fold_of), "chosen_in_fold": g.map(chosen_of)}))
        self.folds.append({"kind": tag[0], "model_tag": tag[1], "seed": seed,
                           "folds": [{"fold": f.fold, "test_groups": [str(x) for x in f.test_groups], "n_train": f.n_train, "n_test": f.n_test,
                                      "chosen": f.chosen, "selection_status": f.selection_status} for f in res.folds]})

    def save(self, out: Path, kind: str) -> list[Path]:
        prot = out / "protected"
        prot.mkdir(parents=True, exist_ok=True)
        (prot / ".gitignore").write_text("*\n!.gitignore\n!README.md\n")
        (prot / "README.md").write_text("Held-out predictions with subject identifiers. Public research data, but kept out of version control by "
                                        "default as a matter of practice; do not publish alongside any patient-derived data.\n")
        paths = []
        rows = [r for r in self.rows if (r["kind"] == kind).all()]
        if rows:
            p = prot / f"{kind}_heldout_predictions.csv"
            pd.concat(rows).to_csv(p, index=False)
            os.chmod(p, 0o600)
            paths.append(p)
        pf = out / f"{kind}_fold_assignments.json"
        pf.write_text(json.dumps([f for f in self.folds if f["kind"] == kind], indent=1) + "\n")
        return paths + [pf]


def _write(out: Path, name: str, payload: dict) -> Path:
    p = out / name
    p.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return p


def _experiment(kind: str, csv: str | None, out: Path, *, seeds, n_boot, n_perm, fast, models_dir, manifest_path) -> dict:
    """Never raises. Returns {status, exit, reason, ...}."""
    rec = {"kind": kind, "status": None, "reason": None, "outputs": []}
    for stale in (f"{kind}_p4a.json", f"{kind}_NOT_RUN.json", f"{kind}_FAILED.json"):          # a new run must not leave an old verdict behind
        (out / stale).unlink(missing_ok=True)
    try:
        try:
            path = P.resolve_data_path(kind, csv)
        except P.DataUnavailable as e:
            rec.update(status=NOT_RUN, exit=2, reason=str(e))
            rec["outputs"].append(str(_write(out, f"{kind}_NOT_RUN.json", P.envelope(kind, "not_run", "NONE", reason=str(e)))))
            return rec
        man = DM.verify_file(NAMES[kind], path, manifest_path)
        rec["manifest"] = man
        if man["status"] == DM.HASH_MISMATCH:
            rec.update(status=REFUSED, exit=3, reason=f"file SHA-256 {man['sha256']} differs from the pinned {man['pinned_sha256']}")
            return rec
        auth = F.authenticate(kind, path, models_dir)
        rec["authentication"] = auth
        if auth["status"] != F.AUTHENTICATED:
            rec.update(status=REFUSED, exit=3, reason=f"the file does not authenticate as the shipped training matrix ({auth['status']}); "
                                                      "this workflow only runs authenticated data (use ml.validation.phase4a for a labelled exploratory run)")
            return rec
        loader = D.load_disease if kind == "disease" else D.load_risk
        try:
            loaded = loader(path)
        except D.DatasetVerificationError as e:
            rec.update(status=REFUSED, exit=3, reason=f"schema/count verification failed: {e}")
            return rec
        raw = pd.read_csv(path)
        dq = E.data_quality_report(kind, raw if kind == "risk" else loaded[0])
        rec["outputs"].append(str(_write(out, f"{kind}_data_quality.json", dq)))
        evidence = F.evidence_status(auth) + ("" if man["status"] == DM.PINNED_MATCH else "+MANIFEST_UNPINNED") + ("+FAST_NOT_REGISTERED_PROTOCOL" if fast else "")
        recorder = Recorder()
        t0 = time.time()
        if kind == "disease":
            df, info = loaded
            res = P.run_disease(df, {**info, "path": path.name}, seeds=seeds, n_boot=n_boot, n_perm=n_perm, fast=fast, evidence=evidence, recorder=recorder)
        else:
            df, dcols, info = loaded
            res = P.run_risk(df, dcols, {**info, "path": path.name}, seeds=seeds, n_boot=n_boot, fast=fast, evidence=evidence, recorder=recorder)
        res = {**res, "authentication": auth, "manifest_check": man, "data_quality": dq}
        rec["outputs"].append(str(_write(out, f"{kind}_p4a.json", res)))
        rec["outputs"] += [str(x) for x in recorder.save(out, kind)]
        rec.update(status=COMPLETED, exit=0, seconds=round(time.time() - t0, 1), evidence_status=evidence)
        return rec
    except Exception as e:  # noqa: BLE001 - a crash is a reported FAILED, never a partial success
        rec.update(status=FAILED, exit=1, reason=f"{type(e).__name__}: {e}")
        rec["outputs"].append(str(_write(out, f"{kind}_FAILED.json", P.envelope(kind, "failed", "NONE", error=rec["reason"], traceback=traceback.format_exc()[-2000:]))))
        return rec


def run_workflow(*, disease_csv: str | None = None, risk_csv: str | None = None, out_dir: Path = DEFAULT_OUT, seeds: list[int] | None = None,
                 n_boot: int = P.PROTOCOL["n_boot"], n_perm: int = P.PROTOCOL["n_perm"], fast: bool = False,
                 models_dir: Path = F.MODELS, manifest_path: Path = DM.MANIFEST, argv: list[str] | None = None) -> tuple[int, dict]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seeds = seeds or [P.PROTOCOL["seed_base"] + i for i in range(P.PROTOCOL["default_seeds"])]
    exps = {k: _experiment(k, c, out, seeds=seeds, n_boot=n_boot, n_perm=n_perm, fast=fast, models_dir=models_dir, manifest_path=manifest_path)
            for k, c in (("disease", disease_csv), ("risk", risk_csv))}
    st = {e["status"] for e in exps.values()}
    overall = (COMPLETED if st == {COMPLETED} else FAILED if FAILED in st else REFUSED if REFUSED in st else NOT_RUN if st == {NOT_RUN} else "INCOMPLETE")
    code = {COMPLETED: 0, FAILED: 1, REFUSED: 3}.get(overall, 2)
    manifest = {"manifest_schema": "neuro-mas-experiment-manifest/1", **run_context(argv or [], out_dir), "seeds": seeds, "n_boot": n_boot, "n_perm": n_perm,
                "fast": fast, "overall_status": overall, "exit_code": code, "experiments": exps,
                "dataset_manifest": DM.load_manifest(manifest_path)["overall_status"], "clinically_validated": False}
    report_note = None
    try:                                                                                  # the report is generated from, and checked against, the JSON
        report_note = len(R.write_report(out, out / "GENERATED_RESULTS.md"))
    except Exception as e:  # noqa: BLE001
        manifest["report_error"] = f"{type(e).__name__}: {e}"
        manifest["overall_status"], manifest["exit_code"], code = FAILED, 1, 1
    manifest["report_claims_verified"] = report_note
    manifest["output_sha256"] = {Path(p).name: _sha(Path(p)) for e in exps.values() for p in e["outputs"] if Path(p).exists() and "protected" not in str(p)}
    _write(out, "experiment_manifest.json", manifest)
    return code, manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--disease-csv")
    ap.add_argument("--risk-csv")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--seeds", type=int, default=P.PROTOCOL["default_seeds"])
    ap.add_argument("--n-boot", type=int, default=P.PROTOCOL["n_boot"])
    ap.add_argument("--n-perm", type=int, default=P.PROTOCOL["n_perm"])
    ap.add_argument("--fast", action="store_true", help="smoke test only; NOT the registered protocol")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.cmd == "report":
        claims = R.write_report(out, out / "GENERATED_RESULTS.md")
        print(f"report regenerated and verified: {len(claims)} numbers checked against the artifacts")
        return 0
    seeds = [P.PROTOCOL["seed_base"] + i for i in range(a.seeds)]
    code, m = run_workflow(disease_csv=a.disease_csv, risk_csv=a.risk_csv, out_dir=out, seeds=seeds, n_boot=a.n_boot, n_perm=a.n_perm, fast=a.fast,
                           argv=list(sys.argv))
    for k, e in m["experiments"].items():
        print(f"{k:8s} {e['status'].upper():10s} {e.get('reason') or ''}", file=sys.stderr if code else sys.stdout)
    print(f"overall: {m['overall_status'].upper()} (exit {code})", file=sys.stderr if code else sys.stdout)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
