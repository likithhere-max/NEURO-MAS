"""Sample-size arithmetic checked against hand calculations and independent library routines.  Planning arithmetic, not evidence."""
import math

import pytest
from scipy.stats import norm

from ml.validation import power_planning as PP
from ml.validation import sample_size as SS


def test_wald_sample_size_matches_hand_calculation():
    assert SS.n_for_proportion_wald(0.5, 0.05) == 385          # 1.959964^2 * .25 / .0025 = 384.15
    assert SS.n_for_proportion_wald(0.85, 0.05) == 196         # 1.959964^2 * .1275 / .0025 = 195.9
    assert SS.n_for_proportion_wald(0.9, 0.1) == math.ceil(1.959964 ** 2 * .09 / .01)


def test_exact_interval_half_width_agrees_with_wald_in_the_middle_of_the_range_and_is_wider_near_the_edge():
    n = 400
    assert SS.clopper_pearson_half_width(200, n) == pytest.approx(1.959964 * math.sqrt(.25 / n), rel=0.03)      # the exact interval is slightly wider than Wald
    assert SS.clopper_pearson_half_width(396, 400) > 1.959964 * math.sqrt(.99 * .01 / 400)
    assert SS.clopper_pearson_half_width(0, 10) == pytest.approx(0.3085 / 2, abs=1e-3)       # exact interval for 0/10 is [0, 0.3085]


def test_exact_n_is_not_smaller_than_needed_and_is_stable_just_above():
    n = SS.n_for_proportion_exact(0.5, 0.05)
    assert SS.clopper_pearson_half_width(round(.5 * n), n) <= 0.05 and 395 <= n <= 410          # exact needs more than Wald's 385
    assert SS.clopper_pearson_half_width(round(.5 * (n - 1)), n - 1) > 0.05 or n == 5


def test_sens_spec_total_is_decided_by_the_binding_class():
    r = SS.subjects_for_sens_spec(0.10, 0.85, 0.80, 0.05)
    assert r["positives_needed"] == 196 and r["negatives_needed"] == SS.n_for_proportion_wald(0.80, 0.05)
    assert r["binding_class"] == "positive" and r["total_subjects"] == math.ceil(196 / 0.10)
    r2 = SS.subjects_for_sens_spec(0.90, 0.85, 0.80, 0.05)
    assert r2["binding_class"] == "negative" and r2["total_subjects"] == 2460


def test_lower_prevalence_needs_more_subjects_and_wider_tolerance_needs_fewer():
    a = SS.subjects_for_sens_spec(0.10, 0.85, 0.80, 0.05)["total_subjects"]
    b = SS.subjects_for_sens_spec(0.50, 0.85, 0.80, 0.05)["total_subjects"]
    c = SS.subjects_for_sens_spec(0.50, 0.85, 0.80, 0.10)["total_subjects"]
    assert a > b > c


def test_hanley_mcneil_agrees_with_the_phase5_planning_module():
    for auc in (0.6, 0.8, 0.9):
        assert SS.hanley_mcneil_se(auc, 24, 8) == pytest.approx(PP.hanley_mcneil_se(auc, 24, 8))


def test_auc_precision_result_actually_achieves_the_half_width_and_one_fewer_does_not():
    r = SS.subjects_for_auc_precision(0.80, 0.5, 0.05)
    assert 1.959964 * SS.hanley_mcneil_se(0.80, r["positives"], r["negatives"]) <= 0.05
    n = r["total_subjects"] - 1
    npos = max(2, round(n * 0.5))
    assert 1.959964 * SS.hanley_mcneil_se(0.80, npos, max(2, n - npos)) > 0.05
    assert SS.subjects_for_auc_precision(0.9, 0.5, 0.05)["total_subjects"] < r["total_subjects"] < SS.subjects_for_auc_precision(0.7, 0.5, 0.05)["total_subjects"]


def test_design_effect_and_effective_sample_size():
    assert SS.design_effect(1, 0.9) == 1 and SS.design_effect(6, 0.5) == 3.5
    assert SS.effective_sample_size(195, 6, 0.5) == pytest.approx(195 / 3.5)
    with pytest.raises(ValueError):
        SS.design_effect(0.5, 0.1)
    with pytest.raises(ValueError):
        SS.design_effect(5, 1.5)


def test_invalid_inputs_are_rejected():
    for bad in (lambda: SS.n_for_proportion_wald(0, .05), lambda: SS.n_for_proportion_wald(.5, 0), lambda: SS.subjects_for_auc_precision(.5, .5, .05),
                lambda: SS.subjects_for_sens_spec(1.0, .8, .8, .05)):
        with pytest.raises(ValueError):
            bad()


def test_illustrative_tables_are_labelled_planning_not_evidence():
    t = SS.illustrative_tables()
    assert t["label"] == "PLANNING_ARITHMETIC_NOT_EVIDENCE" and len(t["sens_spec"]) == 6 and len(t["auc_precision"]) == 6
    assert norm.ppf(0.975) == pytest.approx(SS.Z, abs=1e-5)


def test_published_planning_table_values():
    """The numbers quoted in docs/phase6/CLINICAL_VALIDATION_PROTOCOL.md are exactly what the module produces."""
    t = SS.illustrative_tables()
    got = {(r["prevalence"], r["half_width"]): (r["positives_needed"], r["negatives_needed"], r["total_subjects"]) for r in t["sens_spec"]}
    assert got[(0.1, 0.1)] == (49, 62, 490) and got[(0.1, 0.05)] == (196, 246, 1960)
    assert got[(0.25, 0.1)] == (49, 62, 196) and got[(0.25, 0.05)] == (196, 246, 784)
    assert got[(0.5, 0.1)] == (49, 62, 124) and got[(0.5, 0.05)] == (196, 246, 492)
    auc = {(r["assumed_auc"], r["prevalence"]): r["total_subjects"] for r in t["auc_precision"]}
    assert auc[(0.7, 0.25)] == 631 and auc[(0.8, 0.25)] == 494 and auc[(0.9, 0.25)] == 280
    assert auc[(0.7, 0.5)] == 416 and auc[(0.8, 0.5)] == 302 and auc[(0.9, 0.5)] == 158
