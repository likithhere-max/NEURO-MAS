"""Phase 6 report pipeline end to end.  The 'authenticated' files are RANDOM NUMBERS matched to a stand-in artifact built from the same
file by the test (see tests/phase4/helpers_p4.py).  Nothing here is evidence about any model or data set."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from ml.validation import candidates_p6 as C6
from ml.validation import dataset_manifest as DM
from ml.validation import phase4a as P
from ml.validation import phase4a_selftest as ST
from ml.validation import phase6 as PH6
from ml.validation import phase6_report as R6
from ml.validation import report_from_artifacts as R
from ml.validation import workflow as W
from tests.phase4.helpers_p4 import forged_disease_csv, make_artifact_dir
from tests.phase5.helpers_p5 import forged_risk_csv

SEEDS = [11, 12]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("p6")
    mp = pytest.MonkeyPatch()
    mp.setattr(P.C, "risk_candidates", lambda fast=False: ST.lr_cands())
    mp.setattr(P.C, "disease_candidates", lambda fast=False: ST.lr_cands())
    mp.setattr(C6, "risk_extension_candidates", lambda cols, fast=False: ST.lr_cands())
    mp.setattr(C6, "disease_extension_candidates", lambda fast=False: ST.lr_cands())
    for v in P.ENV_VARS.values():
        mp.delenv(v, raising=False)
    d, r = forged_disease_csv(tmp / "parkinsons.data", seed=7), forged_risk_csv(tmp / "parkinsons_updrs.data", seed=8)
    adir = make_artifact_dir(tmp / "models", "disease", pd.read_csv(d))
    make_artifact_dir(tmp / "models", "risk", pd.read_csv(r))
    man = tmp / "manifest.json"
    shutil.copy(DM.MANIFEST, man)
    kw = dict(disease_csv=str(d), risk_csv=str(r), models_dir=adir, manifest_path=man, seeds=SEEDS, n_boot=20, n_perm=3, fast=True, argv=["pytest"])
    c1, m1 = W.run_workflow(out_dir=tmp / "p4a", **kw)
    c2, m2 = W.run_workflow(out_dir=tmp / "ext", candidate_set=PH6.EXT_ID, **kw)
    code, res = R6.build(tmp / "p4a", tmp / "report", ext_dir=tmp / "ext", risk_csv=r, n_boot=30)
    yield {"tmp": tmp, "c": (c1, c2, code), "m": (m1, m2), "res": res, "report": tmp / "report", "risk_csv": r, "kw": kw}
    mp.undo()


def test_both_runs_and_the_report_complete(runs):
    assert runs["c"] == (0, 0, 0)
    assert runs["res"]["status"] == "completed" and runs["res"]["clinically_validated"] is False


def test_extension_run_is_labelled_exploratory_and_records_both_protocols(runs):
    m = runs["m"][1]
    assert m["candidate_set"] == "p6-1-ext" and m["extension_protocol_id"] == "p6-1" and m["extension_protocol_sha256"] == PH6.PROTOCOL_P6_SHA256
    for k in ("disease", "risk"):
        assert m["experiments"][k]["evidence_status"].endswith("+EXPLORATORY_EXTENSION_P6_1")
    j = json.loads((runs["tmp"] / "ext" / "risk_p6ext.json").read_text())
    assert j["protocol_id"] == "p6-1" and j["parent_protocol_id"] == "p4a-1" and j["parent_protocol_sha256"] == P.PROTOCOL_SHA256
    assert not (runs["tmp"] / "ext" / "risk_p4a.json").exists()


def test_the_p4a_run_is_not_touched_by_the_extension_run(runs):
    assert runs["m"][0]["candidate_set"] == "p4a-1" and (runs["tmp"] / "p4a" / "risk_p4a.json").exists()
    assert (runs["tmp"] / "p4a" / "protected" / "risk_heldout_predictions.csv").exists()
    assert (runs["tmp"] / "ext" / "protected" / "p6ext_risk_heldout_predictions.csv").exists()


def test_a_different_candidate_set_cannot_overwrite_an_existing_run(runs):
    with pytest.raises(ValueError, match="separate --out"):
        W.run_workflow(out_dir=runs["tmp"] / "p4a", candidate_set=PH6.EXT_ID, **runs["kw"])
    with pytest.raises(ValueError, match="separate --out"):
        W.run_workflow(out_dir=runs["tmp"] / "ext", **runs["kw"])


def test_evidence_label_is_copied_from_the_run_and_never_upgraded(runs):
    for k in ("risk", "disease"):
        assert runs["res"][k]["evidence_status"] == runs["m"][0]["experiments"][k]["evidence_status"]
        assert "+FAST_NOT_REGISTERED_PROTOCOL" in runs["res"][k]["evidence_status"]


def test_every_registered_model_has_every_metric_and_an_interval(runs):
    r = runs["res"]["risk"]
    assert set(r["models"]) == set(R6.RISK_P4A_SETS)
    for t, blk in r["models"].items():
        assert set(blk["metrics"]) == set(R6.CM.METRICS)
        assert blk["n_clusters"] == 42 and blk["confusion"]["tp"] + blk["confusion"]["fn"] + blk["confusion"]["fp"] + blk["confusion"]["tn"] == blk["n_rows"]
        assert len(blk["decision_curve"]) == len(R6.DCA_GRID) and blk["reliability"]
        assert blk["metrics"]["roc_auc"]["lo"] is not None


def test_point_metrics_are_recomputable_from_the_saved_predictions(runs):
    preds = pd.read_csv(runs["tmp"] / "p4a" / "protected" / "risk_heldout_predictions.csv", dtype={"subject": str})
    s = preds[(preds["model_tag"] == "FULL") & (preds["seed"] == SEEDS[0])]
    assert runs["res"]["risk"]["models"]["FULL"]["metrics"]["roc_auc"]["point"] == pytest.approx(roc_auc_score(s["y"], s["p"]))
    assert s["outer_fold"].notna().all()


def test_voice_value_verdict_is_never_stated_as_no_value_and_matches_the_rule(runs):
    vv = runs["res"]["risk"]["voice_value"]
    assert vv["verdict"] in ("VOICE_ADDS_MEASURABLE_INFORMATION_ON_THIS_DATA", "INCONCLUSIVE")
    assert set(vv["by_reference"]) == set(P.PROTOCOL["reference_for_voice_value"])
    met = all(all(v["excludes_zero_by_seed"]) for v in vv["by_reference"].values())
    assert vv["rule_met_recomputed"] == met and len(next(iter(vv["by_reference"].values()))["point_by_seed"]) == len(SEEDS)
    assert vv["p4a_recorded_verdict"]["registered_rule_met"] is not None


def test_pairs_are_computed_for_every_seed_and_every_metric(runs):
    pr = runs["res"]["risk"]["paired"]["FULL_minus_TIME_BASELINE"]
    assert set(pr) == {str(s) for s in SEEDS} and set(pr[str(SEEDS[0])]) == set(R6.PAIR_METRICS)


def test_extension_is_paired_with_p4a_on_identical_folds(runs):
    ext = runs["res"]["risk"]["extension"]
    assert ext["status"] == "completed" and ext["label"] == "EXPLORATORY" and set(ext["sets"]) == set(PH6.RISK_SETS)
    for t, blk in ext["sets"].items():
        assert blk["status"] == "completed" and "roc_auc" in blk["paired_extension_minus_p4a1"]
    assert runs["res"]["disease"]["extension"]["status"] == "completed"


def test_extension_with_different_folds_is_refused_not_compared(runs, tmp_path):
    shutil.copytree(runs["tmp"] / "ext", tmp_path / "ext2")
    f = tmp_path / "ext2" / "protected" / "p6ext_risk_heldout_predictions.csv"
    d = pd.read_csv(f, dtype={"subject": str})
    d["outer_fold"] = (d["outer_fold"] + 1) % 10
    d.to_csv(f, index=False)
    code, res = R6.build(runs["tmp"] / "p4a", tmp_path / "rep", ext_dir=tmp_path / "ext2", risk_csv=runs["risk_csv"], n_boot=20)
    s = res["risk"]["extension"]["sets"]
    assert all(v["status"] == "refused" and "outer-fold" in v["reason"] for v in s.values())


def test_subgroups_use_covariates_only_when_the_file_reproduces_the_saved_rows(runs, tmp_path):
    sg = runs["res"]["risk"]["subgroups"]
    assert sg["status"] == "completed" and set(k.split("|")[1] for k in sg["groups"]) == {"sex", "age_tertile", "baseline_severity_tertile"}
    code, res = R6.build(runs["tmp"] / "p4a", tmp_path / "rep_nocsv", n_boot=20)
    assert res["risk"]["subgroups"]["status"] == "not_run" and "not supplied" in res["risk"]["subgroups"]["reason"]
    other = forged_risk_csv(tmp_path / "other.data", seed=99)
    code, res = R6.build(runs["tmp"] / "p4a", tmp_path / "rep_wrong", risk_csv=other, n_boot=20)
    assert res["risk"]["subgroups"]["status"] == "not_run" and "row by row" in res["risk"]["subgroups"]["reason"]


def test_disease_report_has_recording_and_subject_level_blocks_and_errors(runs):
    d = runs["res"]["disease"]
    assert {"candidates", "dummy_prior", "classic3_LR"} <= set(d["models"])
    sl = d["models"]["candidates"]["subject_level"]
    assert sl["n_subjects"] == sl["n_pd"] + sl["n_healthy"] and "note" in sl
    assert "candidates_minus_classic3_LR" in d["paired"] and d["error_analysis"]["n_subjects"] == sl["n_subjects"]


def test_generated_markdown_numbers_match_the_json_and_tampering_is_detected(runs):
    rep = runs["report"]
    for name in ("MODEL_COMPARISON.generated.md", "ERROR_ANALYSIS.generated.md"):
        md = (rep / name).read_text()
        claims = json.loads((rep / name.replace(".md", ".claims.json")).read_text())
        assert claims
        R.verify(md, claims, rep)
        disp = claims[0]["display"]
        bad = md.replace(f" {disp} ", " 9.999 ", 1) if f" {disp} " in md else md.replace(disp, "9.999", 1)
        assert bad != md
        with pytest.raises(R.ReportMismatch):
            R.verify(bad, claims, rep)


def test_a_run_that_is_not_completed_gives_not_run_and_no_numbers(tmp_path, monkeypatch):
    monkeypatch.setattr(P.D, "CACHE", tmp_path / "empty")
    for v in P.ENV_VARS.values():
        monkeypatch.delenv(v, raising=False)
    W.run_workflow(out_dir=tmp_path / "nodata")
    code, res = R6.build(tmp_path / "nodata", tmp_path / "rep")
    assert code == 2 and res["status"] == "not_run"
    assert (tmp_path / "rep" / "phase6_NOT_RUN.json").exists() and not (tmp_path / "rep" / "phase6_metrics.json").exists()
    assert not (tmp_path / "rep" / "MODEL_COMPARISON.generated.md").exists()


def test_missing_manifest_or_predictions_fail_closed(tmp_path, runs):
    code, res = R6.build(tmp_path / "nothing", tmp_path / "rep1")
    assert code == 2 and "experiment_manifest.json" in res["risk"]["reason"]
    shutil.copytree(runs["tmp"] / "p4a", tmp_path / "np4a")
    (tmp_path / "np4a" / "protected" / "risk_heldout_predictions.csv").unlink()
    code, res = R6.build(tmp_path / "np4a", tmp_path / "rep2", n_boot=20)
    assert res["risk"]["status"] == "not_run" and "prediction file is missing" in res["risk"]["reason"] and res["disease"]["status"] == "completed" and code == 2


def test_wrong_candidate_set_directory_is_rejected(runs, tmp_path):
    code, res = R6.build(runs["tmp"] / "ext", tmp_path / "rep", n_boot=20)
    assert code == 2 and "expected 'p4a-1'" in res["risk"]["reason"]


def test_scores_outside_unit_interval_or_missing_values_are_rejected(runs, tmp_path):
    shutil.copytree(runs["tmp"] / "p4a", tmp_path / "bad")
    f = tmp_path / "bad" / "protected" / "risk_heldout_predictions.csv"
    d = pd.read_csv(f, dtype={"subject": str})
    d.loc[0, "p"] = 1.5
    d.to_csv(f, index=False)
    code, res = R6.build(tmp_path / "bad", tmp_path / "rep", n_boot=20)
    assert "outside [0, 1]" in res["risk"]["reason"]
