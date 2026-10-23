"""Input-domain / OOD abstention tests (Phase 2, C4).

The provisional-domain tests use the REAL training moments stored inside the shipped StandardScaler (so the
mean vector is a genuine in-distribution point, with no dataset needed). Fitted-domain tests use SYNTHETIC
Gaussian data only to check the code's behaviour; they are not evidence about acoustic data.
"""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.ml.ood import InputDomain, evaluate_domain_detector, fit_domain

MODELS = Path(__file__).resolve().parents[1] / "ml" / "models"


@pytest.fixture(scope="module")
def voice_domain():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = joblib.load(MODELS / "parkinsons_demo_classifier.joblib")
    feats = json.loads((MODELS / "parkinsons_demo_classifier_metadata.json").read_text())["feature_columns_in_order"]
    return InputDomain.from_scaler("voice", feats, m.named_steps["scaler"]), feats, m.named_steps["scaler"]


@pytest.fixture(scope="module")
def risk_domain():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = joblib.load(MODELS / "parkinsons_risk_classifier.joblib")
    feats = json.loads((MODELS / "parkinsons_risk_classifier_metadata.json").read_text())["feature_columns_in_order"]
    return InputDomain.from_scaler("risk", feats, m.named_steps["scaler"]), feats, m.named_steps["scaler"]


def mean_row(feats, scaler):
    return dict(zip(feats, map(float, scaler.mean_)))


def test_training_mean_vector_is_in_domain_for_both_shipped_models(voice_domain, risk_domain):
    for dom, feats, sc in (voice_domain, risk_domain):
        r = dom.check(mean_row(feats, sc))
        assert r.status == "in_domain", r.reasons
        assert r.domain_source == "scaler_moments_provisional" and "L1_contract" in r.checks_run
        assert r.max_abs_z == pytest.approx(0.0, abs=1e-9)


def test_shipped_scaler_moments_satisfy_the_linear_identities(voice_domain):
    # Because the identities are linear, the TRAINING MEAN of DDP must equal 3 x the training mean of RAP.
    dom, feats, sc = voice_domain
    m = dict(zip(feats, sc.mean_))
    assert m["Jitter:DDP"] == pytest.approx(3 * m["MDVP:RAP"], rel=1e-2)
    assert m["Shimmer:DDA"] == pytest.approx(3 * m["Shimmer:APQ3"], rel=1e-2)


@pytest.mark.parametrize("mutate, code", [
    (lambda r: r.pop("HNR"), "missing_feature"),
    (lambda r: r.update({"bogus": 1.0}), "unexpected_feature"),
    (lambda r: r.update({"HNR": "21.0"}), "not_numeric"),
    (lambda r: r.update({"HNR": True}), "not_numeric"),
    (lambda r: r.update({"HNR": float("nan")}), "non_finite"),
    (lambda r: r.update({"NHR": float("inf")}), "non_finite"),
])
def test_contract_violations_abstain_as_invalid_input(voice_domain, mutate, code):
    dom, feats, sc = voice_domain
    row = mean_row(feats, sc)
    mutate(row)
    r = dom.check(row)
    assert r.status == "abstain_invalid_input" and code in {x.code for x in r.reasons}


def test_unit_error_on_fundamental_frequency_is_rejected(voice_domain):
    dom, feats, sc = voice_domain
    row = mean_row(feats, sc)
    for f in ("MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)"):
        row[f] /= 1000.0  # Hz entered as kHz
    r = dom.check(row)
    assert r.status == "abstain_invalid_input" and "physical_bound" in {x.code for x in r.reasons}


def test_negative_jitter_and_broken_ordering_and_broken_identity(voice_domain):
    dom, feats, sc = voice_domain
    base = mean_row(feats, sc)
    neg = {**base, "MDVP:Jitter(%)": -0.01}
    assert dom.check(neg).status == "abstain_invalid_input"
    order = {**base, "MDVP:Flo(Hz)": base["MDVP:Fhi(Hz)"] + 10}
    assert "ordering" in {x.code for x in dom.check(order).reasons}
    ident = {**base, "Jitter:DDP": base["Jitter:DDP"] * 5}
    rr = dom.check(ident)
    assert rr.status == "abstain_invalid_input" and "identity" in {x.code for x in rr.reasons}


def test_percent_vs_fraction_error_is_out_of_domain(voice_domain):
    dom, feats, sc = voice_domain
    row = mean_row(feats, sc)
    row["MDVP:Jitter(%)"] *= 100  # identities untouched, so only the marginal layer can notice
    r = dom.check(row)
    assert r.status == "abstain_out_of_domain" and any(x.code == "range" and x.feature == "MDVP:Jitter(%)" for x in r.reasons)


def test_risk_model_hard_bounds(risk_domain):
    dom, feats, sc = risk_domain
    base = mean_row(feats, sc)
    assert dom.check({**base, "test_time": -3.0}).status == "abstain_invalid_input"
    assert dom.check({**base, "baseline_total_UPDRS": 300.0}).status == "abstain_invalid_input"
    assert dom.check({**base, "delta_Shimmer:DDA": base["delta_Shimmer:APQ3"] * 3 + 1.0}).status == "abstain_invalid_input"


def test_invalid_input_outranks_out_of_domain(voice_domain):
    dom, feats, sc = voice_domain
    row = mean_row(feats, sc)
    row["MDVP:Jitter(%)"] *= 100
    row["MDVP:Fo(Hz)"] = 0.1
    assert dom.check(row).status == "abstain_invalid_input"


def test_non_mapping_input(voice_domain):
    assert voice_domain[0].check([1, 2, 3]).status == "abstain_invalid_input"


# ----------------------------------------------------------------------------- fitted domain (synthetic Gaussian)
def correlated(n, p=6, rho=0.95, seed=0):
    rng = np.random.default_rng(seed)
    cov = np.full((p, p), rho) + (1 - rho) * np.eye(p)
    return pd.DataFrame(rng.multivariate_normal(np.zeros(p), cov, size=n), columns=[f"x{i}" for i in range(p)])


def test_joint_detector_catches_correlation_breaking_that_range_checks_miss():
    train, test = correlated(600, seed=1), correlated(300, seed=2)
    dom = fit_domain("generic", train, groups=np.arange(len(train)) // 3, range_margin=0.0)
    assert dom.source == "fitted" and dom.d2_calibration["calibration"] == "leave-one-group-out"

    def break_corr(r):  # opposite signs on strongly positively-correlated features; each marginal stays in range
        r["x0"], r["x1"] = 1.5, -1.5
        return r

    def far_shift(r):
        r["x0"] += 30
        return r

    rows = test.to_dict("records")
    ev = evaluate_domain_detector(dom, rows, {"correlation_break": break_corr, "far_shift_one_feature": far_shift})
    assert ev["false_rejection_rate"] <= 0.06           # nominal ~1% (+ sampling noise) on UNSEEN rows
    assert ev["perturbations"]["correlation_break"]["missed_ood_rate"] <= 0.05
    assert ev["perturbations"]["far_shift_one_feature"]["missed_ood_rate"] == 0.0
    # and the marginal layer alone really would have missed the correlation break:
    marginal_only = InputDomain(kind="generic", feature_order=dom.feature_order, source="fitted", mean=dom.mean, sd=dom.sd,
                                train_min=dom.train_min, train_max=dom.train_max, range_margin=0.0)
    assert evaluate_domain_detector(marginal_only, rows, {"correlation_break": break_corr})["perturbations"]["correlation_break"]["missed_ood_rate"] > 0.9


def test_fitted_domain_json_roundtrip_is_lossless_and_has_no_pickle():
    dom = fit_domain("generic", correlated(200, seed=3))
    text = dom.to_json()
    assert "NaN" not in text
    back = InputDomain.from_json(text)
    row = {c: 0.0 for c in dom.feature_order}
    assert back.check(row).status == dom.check(row).status == "in_domain"
    assert back.d2_threshold == pytest.approx(dom.d2_threshold)


def test_fit_rejects_non_finite_training_data_and_handles_constant_columns():
    bad = correlated(50)
    bad.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        fit_domain("generic", bad)
    const = correlated(80)
    const["x0"] = 1.0
    dom = fit_domain("generic", const)
    assert dom.check({**{c: 0.0 for c in dom.feature_order}, "x0": 1.0}).status == "in_domain"
    json.loads(dom.to_json())  # valid JSON despite the zero-variance column


@settings(max_examples=60, deadline=None)
@given(k=st.floats(min_value=-7.9, max_value=7.9, allow_nan=False), idx=st.integers(0, 5))
def test_property_inside_z_limit_never_flagged_outside_always_flagged(k, idx):
    mean = {f"x{i}": float(i) for i in range(6)}
    sd = {f"x{i}": 2.0 for i in range(6)}
    dom = InputDomain(kind="generic", feature_order=list(mean), source="scaler_moments_provisional", mean=mean, sd=sd, z_limit=8.0)
    row = dict(mean)
    row[f"x{idx}"] = mean[f"x{idx}"] + k * 2.0
    assert dom.check(row).status == "in_domain"
    row[f"x{idx}"] = mean[f"x{idx}"] + (8.1 + abs(k)) * 2.0
    assert dom.check(row).status == "abstain_out_of_domain"
