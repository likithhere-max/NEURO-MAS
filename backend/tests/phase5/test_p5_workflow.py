"""Phase 5 / Stage 7: the authenticated-data workflow refuses what it must refuse, never turns a missing input into a success, saves what it
must save, and produces a report whose every number matches the machine-readable outputs.
The 'authenticated' files below are RANDOM NUMBERS made to match a synthetic stand-in artifact (see helpers); nothing here is evidence."""
import json
import os
import shutil
import stat

import pandas as pd
import pytest

from ml.validation import dataset_manifest as DM
from ml.validation import phase4a as P
from ml.validation import phase4a_selftest as ST
from ml.validation import report_from_artifacts as R
from ml.validation import workflow as W
from tests.phase4.helpers_p4 import forged_disease_csv, make_artifact_dir
from tests.phase5.helpers_p5 import forged_risk_csv


@pytest.fixture()
def fast_candidates(monkeypatch):
    monkeypatch.setattr(P.C, "risk_candidates", lambda fast=False: ST.lr_cands())
    monkeypatch.setattr(P.C, "disease_candidates", lambda fast=False: ST.lr_cands())


@pytest.fixture()
def no_env(monkeypatch, tmp_path):
    for v in P.ENV_VARS.values():
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(P.D, "CACHE", tmp_path / "empty_cache")


@pytest.fixture()
def manifest(tmp_path):
    m = tmp_path / "manifest.json"
    shutil.copy(DM.MANIFEST, m)
    return m


# ---------------------------------------------------------------------------------------------- missing prerequisites
def test_no_data_is_not_run_exit_2_and_the_report_shows_no_numbers(tmp_path, no_env, manifest):
    code, m = W.run_workflow(out_dir=tmp_path / "o", manifest_path=manifest)
    assert code == 2 and m["overall_status"] == W.NOT_RUN and m["clinically_validated"] is False
    assert {e["status"] for e in m["experiments"].values()} == {W.NOT_RUN}
    md = (tmp_path / "o" / "GENERATED_RESULTS.md").read_text()
    assert "NOT RUN" in md and "completed" not in md.split(R.BEGIN)[1]
    assert not list((tmp_path / "o").glob("*_p4a.json")) and json.loads((tmp_path / "o" / "experiment_manifest.json").read_text())["report_claims_verified"] == 0


def test_main_returns_the_nonzero_exit_code_for_ci(tmp_path, no_env):
    assert W.main(["run", "--out", str(tmp_path / "o")]) == 2


def test_a_stale_result_from_an_earlier_run_cannot_survive_a_new_not_run(tmp_path, no_env, manifest):
    out = tmp_path / "o"
    out.mkdir()
    (out / "risk_p4a.json").write_text(json.dumps({"status": "completed", "dispersion_across_seeds": {"FULL": {"mean": 0.99}}}))
    code, _ = W.run_workflow(out_dir=out, manifest_path=manifest)
    assert code == 2 and not (out / "risk_p4a.json").exists()


# ---------------------------------------------------------------------------------------------- refusals
def test_unauthenticated_file_is_refused_not_run(tmp_path, no_env, manifest):
    real_stand_in = forged_disease_csv(tmp_path / "train.data", seed=1)
    adir = make_artifact_dir(tmp_path / "models", "disease", pd.read_csv(real_stand_in))
    other = forged_disease_csv(tmp_path / "other.data", seed=2)                       # right shape, different numbers
    code, m = W.run_workflow(disease_csv=str(other), out_dir=tmp_path / "o", models_dir=adir, manifest_path=manifest)
    e = m["experiments"]["disease"]
    assert code == 3 and e["status"] == W.REFUSED and "does not authenticate" in e["reason"]
    assert not (tmp_path / "o" / "disease_p4a.json").exists()


def test_a_pinned_hash_mismatch_is_refused_before_anything_else(tmp_path, no_env, manifest):
    f = forged_disease_csv(tmp_path / "parkinsons.data", seed=3)
    adir = make_artifact_dir(tmp_path / "models", "disease", pd.read_csv(f))
    DM.pin("parkinsons", f, manifest, adir, now="2026-01-01T00:00:00Z")
    f.write_bytes(f.read_bytes() + b"\n")                                             # one extra byte after pinning
    code, m = W.run_workflow(disease_csv=str(f), out_dir=tmp_path / "o", models_dir=adir, manifest_path=manifest)
    assert code == 3 and "differs from the pinned" in m["experiments"]["disease"]["reason"]


def test_schema_violation_is_refused(tmp_path, no_env, manifest):
    f = forged_disease_csv(tmp_path / "parkinsons.data", seed=4)
    df = pd.read_csv(f)
    adir = make_artifact_dir(tmp_path / "models", "disease", df)
    bad = df.copy()
    bad.loc[0, "status"] = 1 - bad.loc[0, "status"]                                   # changes the PD-row count: fails the published structure
    bad.to_csv(f, index=False)
    code, m = W.run_workflow(disease_csv=str(f), out_dir=tmp_path / "o", models_dir=adir, manifest_path=manifest)
    assert code == 3 and m["experiments"]["disease"]["status"] == W.REFUSED


def test_one_dataset_present_one_missing_is_incomplete_exit_2_never_success(tmp_path, no_env, manifest, fast_candidates):
    f = forged_disease_csv(tmp_path / "parkinsons.data", seed=5)
    adir = make_artifact_dir(tmp_path / "models", "disease", pd.read_csv(f))
    code, m = W.run_workflow(disease_csv=str(f), out_dir=tmp_path / "o", models_dir=adir, manifest_path=manifest, seeds=[1], n_boot=20, n_perm=3, fast=True)
    st = {k: e["status"] for k, e in m["experiments"].items()}
    assert st == {"disease": W.COMPLETED, "risk": W.NOT_RUN} and m["overall_status"] == "INCOMPLETE" and code == 2


# ---------------------------------------------------------------------------------------------- the full path on authenticated synthetic stand-ins
@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("wf")
    mp = pytest.MonkeyPatch()
    mp.setattr(P.C, "risk_candidates", lambda fast=False: ST.lr_cands())
    mp.setattr(P.C, "disease_candidates", lambda fast=False: ST.lr_cands())
    for v in P.ENV_VARS.values():
        mp.delenv(v, raising=False)
    d, r = forged_disease_csv(tmp / "parkinsons.data", seed=7), forged_risk_csv(tmp / "parkinsons_updrs.data", seed=8)
    adir = make_artifact_dir(tmp / "models", "disease", pd.read_csv(d))
    make_artifact_dir(tmp / "models", "risk", pd.read_csv(r))
    man = tmp / "manifest.json"
    shutil.copy(DM.MANIFEST, man)
    out = tmp / "out"
    code, m = W.run_workflow(disease_csv=str(d), risk_csv=str(r), out_dir=out, models_dir=adir, manifest_path=man, seeds=[11, 12],
                             n_boot=20, n_perm=3, fast=True, argv=["pytest"])
    yield {"code": code, "m": m, "out": out, "tmp": tmp}
    mp.undo()


def test_full_run_completes_and_labels_the_evidence_honestly(full_run):
    m = full_run["m"]
    assert full_run["code"] == 0 and m["overall_status"] == W.COMPLETED and m["clinically_validated"] is False
    for k in ("disease", "risk"):
        ev = m["experiments"][k]["evidence_status"]
        assert ev.startswith("REAL_DATA_AUTHENTICATED_AGAINST_SHIPPED_ARTIFACT") and "+MANIFEST_UNPINNED" in ev and "+FAST_NOT_REGISTERED_PROTOCOL" in ev
        assert json.loads((full_run["out"] / f"{k}_p4a.json").read_text())["clinically_validated"] is False


def test_experiment_manifest_records_provenance(full_run):
    m = json.loads((full_run["out"] / "experiment_manifest.json").read_text())
    assert m["protocol_sha256"] == P.PROTOCOL_SHA256 and m["seeds"] == [11, 12] and m["argv"] == ["pytest"]
    assert m["git_commit"] is None or len(m["git_commit"]) == 40
    assert m["environment"]["sklearn"] and m["dataset_manifest"] == "UNPINNED_NO_ACCESS"
    for name, digest in m["output_sha256"].items():
        assert W._sha(full_run["out"] / name) == digest


def test_fold_assignments_cover_every_subject_exactly_once_per_model_and_seed(full_run):
    for kind in ("disease", "risk"):
        folds = json.loads((full_run["out"] / f"{kind}_fold_assignments.json").read_text())
        assert {f["seed"] for f in folds} == {11, 12} and folds
        preds = pd.read_csv(full_run["out"] / "protected" / f"{kind}_heldout_predictions.csv")
        for f in folds:
            groups = [g for fold in f["folds"] for g in fold["test_groups"]]
            assert len(groups) == len(set(groups)), (kind, f["model_tag"], f["seed"])
            sub = preds[(preds.model_tag == f["model_tag"]) & (preds.seed == f["seed"])]
            assert set(map(str, sub.subject)) == set(groups) and sub.p.between(0, 1).all() and sub.outer_fold.notna().all()
            for fold in f["folds"]:                                                    # a subject never appears in its own training set
                assert fold["n_train"] + fold["n_test"] == len(sub)


def test_predictions_are_stored_in_protected_storage(full_run):
    prot = full_run["out"] / "protected"
    assert (prot / ".gitignore").read_text().startswith("*")
    mode = stat.S_IMODE(os.stat(prot / "risk_heldout_predictions.csv").st_mode)
    assert mode == 0o600


def test_data_quality_reports_are_saved_and_versioned(full_run):
    rq = json.loads((full_run["out"] / "risk_data_quality.json").read_text())
    dq = json.loads((full_run["out"] / "disease_data_quality.json").read_text())
    assert rq["report_version"] == dq["report_version"] and rq["rows"]["accounting_identity_holds"] is True
    assert dq["subjects"]["parsed"] == 32 and rq["comparison_with_shipped_artifact"]["matches"] is False      # a synthetic file does not reproduce 5,801


def test_generated_report_numbers_match_the_artifacts_and_tampering_is_detected(full_run):
    out = full_run["out"]
    md = (out / "GENERATED_RESULTS.md").read_text()
    claims = json.loads((out / "GENERATED_RESULTS.claims.json").read_text())
    assert len(claims) > 40 and full_run["m"]["report_claims_verified"] == len(claims)
    R.verify(md, claims, out)                                                          # passes untouched
    # (1) edit a displayed number in the markdown
    token = next(c["display"] for c in claims if c["display"] not in ("0.999", "n/a") and md.count(c["display"]) >= 1)
    edited = md.replace(token, "0.999", 1)
    assert edited != md and R.BEGIN in edited
    with pytest.raises(R.ReportMismatch):
        R.verify(edited, claims, out)
    # (2) edit the artifact after the report was rendered
    p = out / "risk_p4a.json"
    orig = p.read_text()
    j = json.loads(orig)
    j["dispersion_across_seeds"]["FULL"]["mean"] += 0.05
    p.write_text(json.dumps(j))
    try:
        with pytest.raises(R.ReportMismatch):
            R.verify(md, claims, out)
    finally:
        p.write_text(orig)
    # (3) numbers claimed from a non-completed experiment
    j2 = json.loads(orig)
    j2["status"] = "failed"
    p.write_text(json.dumps(j2))
    try:
        with pytest.raises(R.ReportMismatch):
            R.verify(md, claims, out)
    finally:
        p.write_text(orig)
    R.verify(md, claims, out)


def test_a_number_that_is_not_a_registered_claim_is_rejected(full_run):
    out = full_run["out"]
    md = (out / "GENERATED_RESULTS.md").read_text()
    claims = json.loads((out / "GENERATED_RESULTS.claims.json").read_text())
    with pytest.raises(R.ReportMismatch):
        R.verify(md.replace(R.END, "| invented | 0.987 |\n" + R.END), claims, out)
