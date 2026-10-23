"""
Exclusion and data-quality accounting (Phase 5, Stage 2).  Machine-readable, no forcing of counts.

Two discrepancies were open after Phase 4A:
  * risk:    5,801 rows stored in the shipped scaler vs 5,833 = 5,875 - 42 expected "if only each patient's first visit is dropped";
  * disease: 32 subjects in the shipped metadata vs 31 on the publisher's page.

What this module can establish WITHOUT the data (from code, proven by tests):
  * the risk filter removes EVERY row tied at a patient's minimum test_time, not one row per patient.  Each tie beyond the first
    therefore costs one extra row.  The shipped 32 extra dropped rows are exactly what 32 such ties would produce; whether the real file
    has them is decided by `risk_exclusion_report(raw)["rows"]["tied_with_baseline_dropped"]`, which only the data can supply;
  * the two disease subject-ID parsers in the repository (training script and Phase 2 loader) are equivalent on all well-formed
    names, so a 31/32 difference is NOT a parser difference.
What it cannot establish without the data: the actual tie count, and which S-codes the file contains.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ml.validation import data as D

REPORT_VERSION = "1.0.0"
SHIPPED = {"risk_rows": 5801, "risk_subjects": 42, "disease_subjects": 32, "disease_rows": 195,
           "publisher_risk_rows": 5875, "publisher_disease_subjects": 31, "publisher_disease_pd_subjects": 23}


# ----------------------------------------------------------------------------- risk
def risk_exclusion_report(raw: pd.DataFrame, *, shipped_rows: int | None = SHIPPED["risk_rows"]) -> dict:
    df = raw.rename(columns=lambda c: c.strip()).reset_index(drop=True)
    for c in ("subject#", "test_time", "total_UPDRS"):
        if c not in df.columns:
            raise D.DatasetVerificationError(f"risk: column {c!r} is required for the exclusion report")
    g = df.groupby("subject#")["test_time"]
    min_t = g.transform("min")
    at_min = df["test_time"] == min_t
    per = pd.DataFrame({"n_rows": g.size(), "n_at_min_test_time": at_min.groupby(df["subject#"]).sum().astype(int)})
    per["n_kept"] = per["n_rows"] - per["n_at_min_test_time"]
    n_raw, n_subj = len(df), int(df["subject#"].nunique())
    tied_extra = int((per["n_at_min_test_time"] - 1).clip(lower=0).sum())
    kept = int(per["n_kept"].sum())
    expected_one_per_subject = n_raw - n_subj
    ties = per[per["n_at_min_test_time"] > 1]
    # do tied rows disagree on the clinical score? (interpolated scores should not)
    sub = df[at_min]
    score_disagree = int((sub.groupby("subject#")["total_UPDRS"].nunique() > 1).sum())
    out = {
        "report_version": REPORT_VERSION, "dataset": "parkinsons_updrs",
        "rows": {"raw": n_raw, "subjects": n_subj, "baseline_rows_one_per_subject": n_subj,
                 "tied_with_baseline_dropped": tied_extra, "kept_for_training": kept,
                 "expected_if_exactly_one_baseline_row_per_subject": expected_one_per_subject,
                 "kept_minus_expected": kept - expected_one_per_subject,
                 "accounting_identity_holds": kept == n_raw - n_subj - tied_extra},
        "ties_at_minimum_test_time": {"subjects_with_ties": int(len(ties)), "max_rows_tied_in_one_subject": int(per["n_at_min_test_time"].max()),
                                      "subjects": {str(k): int(v) for k, v in ties["n_at_min_test_time"].items()},
                                      "subjects_whose_tied_rows_disagree_on_total_UPDRS": score_disagree},
        "subjects_left_with_no_training_rows": [str(k) for k in per.index[per["n_kept"] == 0]],
        "data_quality": {
            "non_finite_cells": int((~np.isfinite(df.select_dtypes("number").to_numpy(float))).sum()),
            "exact_duplicate_rows": int(df.duplicated().sum()),
            "duplicate_subject_test_time_pairs": int(df.duplicated(subset=["subject#", "test_time"]).sum()),
            "negative_test_time_rows": int((df["test_time"] < 0).sum()),
            "subjects_with_rows_out_of_time_order_in_file": int((df.groupby("subject#")["test_time"].apply(
                lambda s: bool((s.diff().dropna() < 0).any()))).sum()),
            "rows_per_subject_min_max": [int(per["n_rows"].min()), int(per["n_rows"].max())],
        },
    }
    if shipped_rows is not None:
        out["comparison_with_shipped_artifact"] = {
            "shipped_training_rows": shipped_rows, "kept_for_training": kept, "matches": kept == shipped_rows,
            "difference": kept - shipped_rows,
            "reading": ("the shipped row count is reproduced by this filter on this file" if kept == shipped_rows else
                        "this file does NOT reproduce the shipped row count; do not describe it as the training data")}
    return out


def reconcile_risk_rows(raw: pd.DataFrame) -> dict:
    """The independent accounting above must equal what build_risk_frame actually keeps. Raises if not (a code/accounting bug)."""
    rep = risk_exclusion_report(raw, shipped_rows=None)
    built, _ = D.build_risk_frame(raw)
    if len(built) != rep["rows"]["kept_for_training"]:
        raise AssertionError(f"build_risk_frame kept {len(built)} rows but the exclusion accounting expects {rep['rows']['kept_for_training']}")
    if not rep["rows"]["accounting_identity_holds"]:
        raise AssertionError("exclusion accounting identity failed")
    return rep


# ----------------------------------------------------------------------------- disease
_LOADER_RE = re.compile(r"(S\d+)_\d+$")


def subject_id_training_script(name: str) -> str | None:
    """ml/training/train_parkinsons_classifier.py: name.str.split('_').str[2]"""
    parts = str(name).split("_")
    return parts[2] if len(parts) > 2 else None


def subject_id_loader(name: str) -> str | None:
    """ml/validation/data.py: name.str.extract(r'(S\\d+)_\\d+$')"""
    m = _LOADER_RE.search(str(name))
    return m.group(1) if m else None


def disease_subject_report(df: pd.DataFrame, *, shipped_subjects: int | None = SHIPPED["disease_subjects"]) -> dict:
    if "name" not in df.columns or "status" not in df.columns:
        raise D.DatasetVerificationError("disease: 'name' and 'status' columns are required")
    a = df["name"].map(subject_id_training_script)
    b = df["name"].map(subject_id_loader)
    disagree = int((a != b).sum())
    sid = b.fillna("UNPARSED")
    per = df.assign(subject=sid).groupby("subject")["status"].agg(["nunique", "first", "size"])
    n = int(per.shape[0])
    pd_n = int((per["first"] == 1).sum())
    rec = re.compile(r"phon_(R\d+)_")
    r_codes = sorted({m.group(1) for m in (rec.match(str(x)) for x in df["name"]) if m})
    numbers = sorted(int(s[1:]) for s in per.index if re.fullmatch(r"S\d+", s))
    return {
        "report_version": REPORT_VERSION, "dataset": "parkinsons",
        "parsers_agree_on_every_row": disagree == 0, "rows_where_parsers_disagree": disagree,
        "unparsed_names": int((b.isna()).sum()),
        "subjects": {"parsed": n, "pd": pd_n, "healthy": n - pd_n, "mixed_label_subjects": [str(k) for k in per.index[per["nunique"] > 1]],
                     "recordings_per_subject_min_max": [int(per["size"].min()), int(per["size"].max())],
                     "recording_session_codes_present": r_codes, "s_code_numbers_min_max": [numbers[0], numbers[-1]] if numbers else None,
                     "s_code_gaps_inside_range": int((numbers[-1] - numbers[0] + 1) - len(numbers)) if numbers else None},
        "comparison": {"shipped_metadata_subjects": shipped_subjects, "publisher_subjects": SHIPPED["publisher_disease_subjects"],
                       "publisher_pd_subjects": SHIPPED["publisher_disease_pd_subjects"],
                       "matches_shipped_metadata": None if shipped_subjects is None else n == shipped_subjects,
                       "matches_publisher": n == SHIPPED["publisher_disease_subjects"],
                       "reading": ("the file parses to the publisher's subject count" if n == SHIPPED["publisher_disease_subjects"] else
                                   "the file parses to a different subject count than the publisher's page; the S-codes present must be inspected "
                                   "(no count is forced)")},
        "data_quality": {"exact_duplicate_rows": int(df.duplicated().sum()), "non_finite_cells": int((~np.isfinite(df.select_dtypes("number").to_numpy(float))).sum())},
    }


def data_quality_report(kind: str, raw: pd.DataFrame) -> dict:
    if kind == "risk":
        return risk_exclusion_report(raw)
    if kind == "disease":
        return disease_subject_report(raw)
    raise ValueError(kind)
