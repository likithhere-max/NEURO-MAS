"""Phase 5 / Stage 2: row-count and subject-count discrepancies are characterised by tests, not forced.
Everything here is synthetic. Where a test mentions 5,875 / 5,801 / 32 it demonstrates a MECHANISM; it says nothing about the real file."""
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.validation import data as D
from ml.validation import dataset_manifest as DM
from ml.validation import exclusions as E
from ml.validation import fingerprint as F
from tests.phase4.helpers_p4 import forged_disease_csv, make_artifact_dir

MANIFEST_SRC = Path(DM.MANIFEST)


def _raw(rows_by_subject: dict[int, list[float]], total=lambda s, t: 20.0 + t) -> pd.DataFrame:
    rows = []
    for s, times in rows_by_subject.items():
        for i, t in enumerate(times):
            rows.append({"subject#": s, "age": 60, "sex": 0, "test_time": float(t), "motor_UPDRS": 1.0, "total_UPDRS": total(s, t), "Jitter(%)": 0.1 * i})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- risk row accounting
def test_hand_checked_exclusion_accounting():
    raw = _raw({1: [0, 0, 0, 5, 9], 2: [0, 0, 0, 5, 9], 3: [0, 5, 9]})         # 13 rows; subjects 1,2 each have 3 rows tied at the minimum
    rep = E.reconcile_risk_rows(raw)
    r = rep["rows"]
    assert (r["raw"], r["subjects"], r["tied_with_baseline_dropped"], r["kept_for_training"]) == (13, 3, 4, 6)
    assert r["expected_if_exactly_one_baseline_row_per_subject"] == 10 and r["kept_minus_expected"] == -4
    assert rep["ties_at_minimum_test_time"]["subjects"] == {"1": 3, "2": 3}


def test_without_ties_exactly_one_row_per_subject_is_dropped():
    raw = _raw({s: [0, 3, 7, 11] for s in range(1, 6)})
    rep = E.reconcile_risk_rows(raw)
    assert rep["rows"]["kept_for_training"] == 20 - 5 and rep["rows"]["kept_minus_expected"] == 0
    assert rep["ties_at_minimum_test_time"]["subjects_with_ties"] == 0


def test_accounting_agrees_with_build_risk_frame_on_random_data():
    rng = np.random.default_rng(0)
    for _ in range(25):
        n_sub = int(rng.integers(2, 9))
        raw = _raw({s: sorted(rng.integers(0, 6, int(rng.integers(2, 12))).tolist()) for s in range(1, n_sub + 1)})   # many ties by design
        rep = E.reconcile_risk_rows(raw)                                            # raises if the two disagree
        built, _ = D.build_risk_frame(raw)
        assert rep["rows"]["kept_for_training"] == len(built)


def test_mechanism_32_ties_turn_5875_rows_into_5801_which_is_the_shipped_count():
    """DEMONSTRATES the mechanism proposed for the 32 unexplained rows. Whether the real file has 32 such ties is unknown until it is supplied."""
    sizes = [140] * 37 + [139] * 5                                                  # 42 subjects, 5,875 rows
    assert sum(sizes) == 5875
    by = {}
    for i, n in enumerate(sizes, start=1):
        times = list(np.arange(n, dtype=float))
        if i <= 32:
            times[1] = times[0]                                                     # exactly one extra row tied at the minimum
        by[i] = times
    raw = _raw(by)
    E.reconcile_risk_rows(raw)                                                      # accounting agrees with build_risk_frame
    rep = E.risk_exclusion_report(raw)                                              # (this one carries the shipped-count comparison)
    assert rep["rows"]["raw"] == 5875 and rep["rows"]["subjects"] == 42
    assert rep["rows"]["tied_with_baseline_dropped"] == 32 and rep["rows"]["kept_for_training"] == 5801
    assert rep["comparison_with_shipped_artifact"]["matches"] is True


def test_a_file_that_does_not_reproduce_the_shipped_count_says_so():
    rep = E.risk_exclusion_report(_raw({s: [0, 3, 7, 11] for s in range(1, 6)}), shipped_rows=5801)
    cmp = rep["comparison_with_shipped_artifact"]
    assert cmp["matches"] is False and "NOT reproduce" in cmp["reading"]


def test_subject_with_only_minimum_time_rows_is_reported_as_lost():
    rep = E.risk_exclusion_report(_raw({1: [0, 0], 2: [0, 4, 8]}))
    assert rep["subjects_left_with_no_training_rows"] == ["1"]


def test_data_quality_counters():
    raw = _raw({1: [0, 2, 1, 4], 2: [0, 3, 6]})
    raw.loc[5, "test_time"] = -1.0
    raw = pd.concat([raw, raw.iloc[[1]]], ignore_index=True)                         # an exact duplicate row
    dq = E.risk_exclusion_report(raw)["data_quality"]
    assert dq["exact_duplicate_rows"] == 1 and dq["negative_test_time_rows"] == 1
    assert dq["subjects_with_rows_out_of_time_order_in_file"] >= 1 and dq["non_finite_cells"] == 0


def test_shipped_baseline_row_is_the_first_in_file_order_among_ties():
    """The row used as baseline is the first tied row in FILE ORDER (sort_values on two keys is stable); a different tie order changes the
    deltas of every later row. Pins the rule documented in app.ml.contract."""
    raw = _raw({1: [0, 0, 5, 9]})
    raw["Jitter(%)"] = [10.0, 99.0, 11.0, 12.0]
    built, _ = D.build_risk_frame(raw)
    assert built["delta_Jitter(%)"].tolist() == [1.0, 2.0]                          # 11-10, 12-10: baseline is the FIRST tied row
    assert len(built) == 2


# ---------------------------------------------------------------------------------------------- disease subject parsing
def test_both_repository_subject_parsers_agree_on_all_well_formed_names():
    names = [f"phon_R{r:02d}_S{s:02d}_{k}" for r in (1, 2) for s in (1, 9, 10, 35, 50, 99) for k in (1, 6, 7)]
    assert all(E.subject_id_training_script(n) == E.subject_id_loader(n) for n in names)


def test_the_parsers_disagree_only_on_malformed_names_and_the_report_surfaces_it():
    df = pd.DataFrame({"name": ["phon_R01_S01_1", "phon_R01_S01", "weird"], "status": [1, 1, 0]})
    assert E.subject_id_training_script("phon_R01_S01") == "S01" and E.subject_id_loader("phon_R01_S01") is None
    rep = E.disease_subject_report(df)
    assert rep["parsers_agree_on_every_row"] is False and rep["unparsed_names"] == 2


def test_disease_report_32_subjects_matches_shipped_metadata_but_not_the_publisher(tmp_path):
    df = pd.read_csv(forged_disease_csv(tmp_path / "d.csv"))
    rep = E.disease_subject_report(df)
    s = rep["subjects"]
    assert (s["parsed"], s["pd"], s["healthy"]) == (32, 24, 8) and rep["parsers_agree_on_every_row"]
    assert rep["comparison"]["matches_shipped_metadata"] is True and rep["comparison"]["matches_publisher"] is False
    assert "no count is forced" in rep["comparison"]["reading"]


def test_report_is_json_serialisable_and_versioned(tmp_path):
    rep = E.disease_subject_report(pd.read_csv(forged_disease_csv(tmp_path / "d.csv")))
    assert json.loads(json.dumps(rep))["report_version"] == E.REPORT_VERSION


# ---------------------------------------------------------------------------------------------- dataset manifest
def test_shipped_manifest_is_honestly_unpinned():
    m = DM.load_manifest()
    assert m["overall_status"] == "UNPINNED_NO_ACCESS"
    for name, ds in m["datasets"].items():
        assert ds["sha256"] is None and ds["pin_status"] == "NOT_PINNED" and ds["retrieved_utc"] is None, name
        assert any("403" in a["outcome"] or "not retrieved" in a["outcome"] for a in ds["access_attempts"]), name
        assert re.fullmatch(r"https://archive\.ics\.uci\.edu/.*", ds["url"])


def test_unpinned_manifest_never_returns_a_success_status(tmp_path):
    f = forged_disease_csv(tmp_path / "parkinsons.data")
    r = DM.verify_file("parkinsons", f)
    assert r["status"] == DM.UNPINNED and r["status"] != DM.PINNED_MATCH
    assert DM.verify_file("parkinsons", tmp_path / "absent.data")["status"] == DM.MISSING_FILE


def test_pin_requires_authentication_then_verify_and_tamper_detection(tmp_path):
    man = tmp_path / "manifest.json"
    shutil.copy(MANIFEST_SRC, man)
    f = forged_disease_csv(tmp_path / "parkinsons.data", seed=5)
    adir = make_artifact_dir(tmp_path / "models", "disease", pd.read_csv(f))          # stands in for the shipped artifact
    # a different file of the right shape must NOT be pinnable
    other = forged_disease_csv(tmp_path / "other.data", seed=6)
    with pytest.raises(DM.ManifestError, match="does not authenticate"):
        DM.pin("parkinsons", other, man, adir)
    assert DM.load_manifest(man)["datasets"]["parkinsons"]["sha256"] is None
    # the authenticated file can be pinned
    ds = DM.pin("parkinsons", f, man, adir, now="2026-01-01T00:00:00Z")
    assert ds["sha256"] == D.sha256_file(f) and DM.load_manifest(man)["overall_status"] == "PARTIALLY_PINNED"
    assert DM.verify_file("parkinsons", f, man)["status"] == DM.PINNED_MATCH
    # one byte changed -> mismatch
    tampered = tmp_path / "t.data"
    tampered.write_bytes(f.read_bytes() + b" ")
    assert DM.verify_file("parkinsons", tampered, man)["status"] == DM.HASH_MISMATCH
    # re-pinning to different bytes is refused
    with pytest.raises(DM.ManifestError):
        DM.pin("parkinsons", other, man, adir)


def test_manifest_cli_exit_codes(tmp_path, capsys):
    assert DM.main(["status"]) == 2                              # unpinned: not a success
    assert DM.main(["verify", "--name", "parkinsons", "--csv", str(tmp_path / "nope.csv")]) == 2
