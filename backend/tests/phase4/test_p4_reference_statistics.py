"""Phase 4A: hand-derived references for the helper statistics, and consistency of the numbers the shipped artifacts report.
Expected values come from closed forms or arithmetic done by hand, not from calling the function under test."""
import json
import math

import pytest

from ml.validation import fingerprint as F
from ml.validation import phase4a as P


# ---------------------------------------------------------------------------------------------- Clopper-Pearson
def test_clopper_pearson_closed_forms():
    # k = 0:  upper = 1 - (alpha/2)^(1/n);  k = n:  lower = (alpha/2)^(1/n)   (exact, from the binomial tail)
    n = 8
    assert P.clopper_pearson(0, n) == pytest.approx((0.0, 1 - 0.025 ** (1 / n)), abs=1e-12)
    assert P.clopper_pearson(n, n) == pytest.approx((0.025 ** (1 / n), 1.0), abs=1e-12)
    assert 1 - 0.025 ** (1 / 8) == pytest.approx(0.36942, abs=1e-5)       # hand: 0.025^(0.125) = exp(ln(.025)/8) = exp(-0.46111) = .63058


def test_clopper_pearson_defining_property():
    # the interval endpoints solve  P(X >= k | p=lo) = alpha/2  and  P(X <= k | p=hi) = alpha/2  (computed with plain sums)
    k, n = 3, 8
    lo, hi = P.clopper_pearson(k, n)
    upper_tail = sum(math.comb(n, i) * lo ** i * (1 - lo) ** (n - i) for i in range(k, n + 1))
    lower_tail = sum(math.comb(n, i) * hi ** i * (1 - hi) ** (n - i) for i in range(0, k + 1))
    assert upper_tail == pytest.approx(0.025, abs=1e-9) and lower_tail == pytest.approx(0.025, abs=1e-9)
    assert P.clopper_pearson(0, 0) is None


def test_eight_healthy_subjects_cannot_give_a_narrow_specificity_interval():
    widths = [hi - lo for lo, hi in (P.clopper_pearson(k, 8) for k in range(9))]
    assert min(widths) > 0.36          # even a perfect 8/8 leaves an interval wider than 36 points


# ---------------------------------------------------------------------------------------------- confusion reconstruction
def test_reconstruct_confusion_hand_example():
    # TP=8 FN=2 FP=3 TN=7  (n=20, 10 positives):  accuracy 15/20, precision 8/11, recall 8/10
    r = P.reconstruct_confusion(0.75, 8 / 11, 0.8, 20, 10)
    assert (r["TP"], r["FN"], r["FP"], r["TN"]) == (8, 2, 3, 7) and r["consistent"]
    assert r["sensitivity"] == 0.8 and r["specificity"] == 0.7 and r["majority_class_accuracy"] == 0.5
    assert r["balanced_accuracy"] == pytest.approx(0.75)


def test_reconstruct_confusion_flags_impossible_figures():
    assert not P.reconstruct_confusion(0.9, 8 / 11, 0.8, 20, 10)["consistent"]       # accuracy contradicts the other two
    assert not P.reconstruct_confusion(0.75, 0.0, 0.8, 20, 10)["consistent"]          # precision 0 cannot come with TP > 0


# ---------------------------------------------------------------------------------------------- shipped artifacts
@pytest.fixture(scope="module")
def audit():
    return P.artifact_audit()


def test_every_candidate_in_both_metadata_files_reconstructs_to_an_integer_confusion_matrix(audit):
    for kind, entry in audit["models"].items():
        assert entry["reconstructed_confusion_matrices"], kind
        for name, rc in entry["reconstructed_confusion_matrices"].items():
            assert rc["consistent"], (kind, name)


def test_artifact_scalers_record_the_row_counts_the_metadata_claims(audit):
    assert audit["models"]["disease"]["training_rows_recorded_in_scaler"] == 195 == audit["models"]["disease"]["n_recordings_claimed"]
    assert audit["models"]["risk"]["training_rows_recorded_in_scaler"] == 5801 == audit["models"]["risk"]["n_recordings_claimed"]


def test_reported_gain_over_the_majority_class_is_small_in_both_models(audit):
    # hand arithmetic from the confusion matrices: disease (151-147)/195 = 4/195 ; risk (3657-3586)/5801 = 71/5801
    assert audit["models"]["disease"]["winner_accuracy_minus_majority_accuracy"] == pytest.approx(4 / 195, abs=1e-12)
    assert audit["models"]["risk"]["winner_accuracy_minus_majority_accuracy"] == pytest.approx(71 / 5801, abs=1e-12)
    rc = audit["models"]["disease"]["reconstructed_confusion_matrices"]["RandomForest_Top10Features"]
    assert (rc["TP"], rc["FN"], rc["FP"], rc["TN"]) == (130, 17, 27, 21)               # specificity 21/48 = 0.4375
    rr = audit["models"]["risk"]["reconstructed_confusion_matrices"]["LogisticRegression"]
    assert (rr["TP"], rr["FN"], rr["FP"], rr["TN"]) == (1325, 890, 1254, 2332)


def test_documented_discrepancies_are_surfaced_not_hidden(audit):
    d, r = audit["models"]["disease"], audit["models"]["risk"]
    assert d["metadata_n_subjects"] == 32 and d["publisher_n_subjects"] == 31            # one-subject disagreement is reported
    assert r["rows_dropped_beyond_first_visit_per_subject"] == 32                         # 5875 - 42 - 5801
    assert r["jitter_coefficients_have_mixed_signs"] is True                              # collinear predictors with cancelling signs
    assert not r["records_dataset_sha256"] and not r["records_sklearn_version"] and not r["records_seed"]


def test_audit_never_claims_clinical_validation(audit):
    env = P.envelope("artifact_audit", "completed", "REAL_ARTIFACTS_NO_DATA_USED", **audit)
    assert env["clinically_validated"] is False and "NO_DATA" in env["evidence_status"]
    json.dumps(env, default=str)       # serialisable
