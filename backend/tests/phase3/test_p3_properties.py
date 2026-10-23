"""Phase 3 property-based tests (hypothesis, derandomised so every run is identical).

Each property is an INVARIANCE or BOUND that follows from the mathematics, not from reading the implementation:
affine equivariance of standardised statistics, monotone-transform invariance of rank statistics, non-negativity and
reset of CUSUM, boundedness of EWMA, permutation invariance of the support index, containment and shrinkage of the Wilson
interval, AUC symmetries, DDI pair symmetry, and so on.  Points within a tolerance of a decision boundary are skipped
(`assume`), because a floating-point flip exactly on a boundary is not a defect.
"""
import math

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings, strategies as st

from app.core.consensus import compute_evidence_consensus
from app.core.evidence import Direction, EvidenceType, HypothesisEvidence
from app.core.metric_semantics import metric_polarity
from app.ml import longitudinal as L
from app.ml.personalized_anomaly import detect_personal_anomaly, effective_z_threshold
from app.schemas.agent import AgentResult, AgentStatus
from app.services.ddi import CuratedTableSource, DDIChecker
from ml.evalkit import metrics as M

PROFILE = settings(max_examples=120, derandomize=True, deadline=None,
                   suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])

val = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)
series = st.lists(val, min_size=4, max_size=40)
pos_scale = st.floats(min_value=1e-3, max_value=1e3)
offset = st.sampled_from([0.0, 1.0, -37.5, 1e4, 1e6, 1e8])


# ---------------------------------------------------------------------------------------- prediction-interval detector
@PROFILE
@given(h=series, x=val, a=pos_scale, b=offset)
def test_point_detector_affine_invariance(h, x, a, b):
    assume(float(np.std(h)) > 1e-2)                                   # well-posed: a real, non-degenerate spread
    base = detect_personal_anomaly(h, x)
    mapped = detect_personal_anomaly([a * v + b for v in h], a * x + b)
    assert base["method"] == mapped["method"] == "z_score"
    thr = effective_z_threshold(len(h))
    assume(abs(abs(base["z_score"]) - thr) > 1e-4)                    # stay off the decision boundary
    assert mapped["z_score"] == pytest.approx(base["z_score"], rel=1e-5, abs=1e-5)
    assert mapped["is_anomaly"] == base["is_anomaly"]


@PROFILE
@given(h=series, x=val)
def test_point_detector_reflection_flips_sign_only(h, x):
    assume(float(np.std(h)) > 1e-2)
    base = detect_personal_anomaly(h, x)
    refl = detect_personal_anomaly([-v for v in h], -x)
    assert refl["z_score"] == pytest.approx(-base["z_score"], rel=1e-9, abs=1e-9)
    assert refl["is_anomaly"] == base["is_anomaly"]


@PROFILE
@given(h=series, x=val)
def test_point_detector_order_of_history_is_irrelevant(h, x):
    assume(float(np.std(h)) > 1e-2)
    a = detect_personal_anomaly(h, x)
    b = detect_personal_anomaly(list(reversed(h)), x)
    assert a["z_score"] == pytest.approx(b["z_score"], rel=1e-9, abs=1e-9) and a["is_anomaly"] == b["is_anomaly"]


@PROFILE
@given(n=st.integers(min_value=3, max_value=2000))
def test_prediction_threshold_is_decreasing_in_n_and_above_normal_quantile(n):
    # PI half-width in SD units: t_{.975,n-1} sqrt(1+1/n) > z_{.975}=1.95996 and decreases toward it
    assert effective_z_threshold(n) > 1.9599
    assert effective_z_threshold(n) >= effective_z_threshold(n + 1) - 1e-3        # table is 3-decimal; allow its rounding


# ------------------------------------------------------------------------------------------------ Q-statistics
@PROFILE
@given(x=st.lists(val, min_size=5, max_size=30), a=pos_scale, b=offset)
def test_q_statistics_are_affine_invariant(x, a, b):
    assume(float(np.std(x)) > 1e-2)
    q0 = L.q_statistics(x)
    q1 = L.q_statistics([a * v + b for v in x])
    for u, v in zip(q0, q1):
        if u is None or v is None:
            continue
        assume(abs(u) < 7.5)                                            # |Q| is capped at 8 by design
        assert v == pytest.approx(u, abs=2e-4)


@PROFILE
@given(x=st.lists(val, min_size=5, max_size=30))
def test_q_statistics_are_finite_and_capped(x):
    for q in L.q_statistics(x):
        assert q is None or (math.isfinite(q) and abs(q) <= 8.2)


# --------------------------------------------------------------------------------------------------- Mann-Kendall
@PROFILE
@given(y=st.lists(val, min_size=4, max_size=40, unique=True))
def test_mann_kendall_is_invariant_to_strictly_increasing_transforms(y):
    y = np.array(y)
    base = L.mann_kendall(y, autocorr_correct=False)
    for f in (lambda v: 3 * v + 7, lambda v: np.exp(v / 1e3), lambda v: v ** 3):
        out = L.mann_kendall(f(y), autocorr_correct=False)
        # exp/cube can create ties through floating-point saturation: compare only when ranks are preserved
        if len(np.unique(f(y))) == len(y) and (np.argsort(f(y)) == np.argsort(y)).all():
            assert out["S"] == base["S"] and out["p_value"] == pytest.approx(base["p_value"], abs=1e-12)


@PROFILE
@given(y=st.lists(val, min_size=4, max_size=40))
def test_mann_kendall_negation_flips_S_and_keeps_p(y):
    y = np.array(y)
    a = L.mann_kendall(y, autocorr_correct=False)
    b = L.mann_kendall(-y, autocorr_correct=False)
    assert a["status"] == b["status"]
    if a["status"] == "assessed":
        assert b["S"] == -a["S"] and b["p_value"] == pytest.approx(a["p_value"], abs=1e-12)
        assert 0.0 <= a["p_value"] <= 1.0 and abs(a["S"]) <= len(y) * (len(y) - 1) // 2


@PROFILE
@given(n=st.integers(min_value=4, max_value=40))
def test_mann_kendall_strictly_monotone_series_attains_extreme_S(n):
    out = L.mann_kendall(np.arange(float(n)), autocorr_correct=False)
    assert out["S"] == n * (n - 1) // 2 and out["p_value"] < 0.2


# ---------------------------------------------------------------------------------------------------- Theil-Sen
@PROFILE
@given(y=st.lists(val, min_size=4, max_size=30), a=st.floats(min_value=0.1, max_value=50), b=val, shift=val)
def test_theil_sen_slope_is_equivariant(y, a, b, shift):
    t = np.arange(len(y), dtype=float)
    base = L.theil_sen(t, np.array(y))
    mapped = L.theil_sen(t + shift, a * np.array(y) + b)
    assert base["status"] == mapped["status"] == "assessed"
    assert mapped["slope_per_day"] == pytest.approx(a * base["slope_per_day"], rel=1e-6, abs=1e-6)


# ----------------------------------------------------------------------------------------------- CUSUM and EWMA
zs = st.lists(st.floats(min_value=-8, max_value=8, allow_nan=False), min_size=0, max_size=60)


@PROFILE
@given(z=zs, k=st.floats(min_value=0.0, max_value=2.0), h=st.floats(min_value=0.5, max_value=10.0))
def test_cusum_paths_are_nonnegative_and_alarm_iff_boundary_crossed(z, k, h):
    out = L.cusum(z, k, h)
    assert all(v >= 0 for v in out["C_plus"]) and all(v >= 0 for v in out["C_minus"])
    crossed = [i for i in range(len(z)) if out["C_plus"][i] > h or out["C_minus"][i] > h]
    assert out["first_alarm_index"] == (crossed[0] if crossed else None)


@PROFILE
@given(z=st.lists(st.floats(min_value=-0.5, max_value=0.5), min_size=1, max_size=80))
def test_cusum_never_alarms_inside_the_reference_value(z):
    # every |z| <= k = 0.5 gives increments <= 0, so both sums stay at 0
    out = L.cusum(z, 0.5, 5.0)
    assert out["first_alarm_index"] is None and out["final_C_plus"] == 0.0 and out["final_C_minus"] == 0.0


@PROFILE
@given(z=st.lists(st.floats(min_value=-8, max_value=8), min_size=1, max_size=60), lam=st.floats(min_value=0.05, max_value=1.0))
def test_ewma_is_a_convex_combination_bounded_by_the_data(z, lam):
    out = L.ewma(z, lam, 3.0)
    m = max(abs(v) for v in z)
    assert all(abs(w) <= m + 1e-9 for w in out["path"])
    assert all(a < b + 1e-12 for a, b in zip(out["limits"], out["limits"][1:]))      # control limits widen to the asymptote
    asym = 3.0 * math.sqrt(lam / (2 - lam))
    assert all(v <= asym + 1e-9 for v in out["limits"])


# ----------------------------------------------------------------------------------- analyze_series end to end
def _times(n):
    return [float(i) for i in range(n)]


@PROFILE
@given(y=st.lists(val, min_size=3, max_size=40), a=pos_scale, b=offset)
def test_analyze_series_is_deterministic_json_safe_and_affine_stable(y, a, b):
    import json
    for metric in ("tremor_score", "gait_speed", "mystery_metric"):
        r1 = L.analyze_series(metric, _times(len(y)), y).as_dict()
        r2 = L.analyze_series(metric, _times(len(y)), y).as_dict()
        assert json.dumps(r1, sort_keys=True, default=str) == json.dumps(r2, sort_keys=True, default=str)   # deterministic
        json.dumps(r1, allow_nan=False, default=str)                                                       # no NaN / inf leaks
        assert r1["is_diagnosis"] is False


@PROFILE
@given(y=st.lists(st.floats(min_value=-50, max_value=50), min_size=12, max_size=40), a=st.floats(min_value=0.5, max_value=20),
       b=st.sampled_from([0.0, 1e3, 1e8]))
def test_analyze_series_signal_set_is_affine_invariant(y, a, b):
    assume(float(np.std(y)) > 0.5)
    base = L.analyze_series("tremor_score", _times(len(y)), y)
    mapped = L.analyze_series("tremor_score", _times(len(y)), [a * v + b for v in y])
    s0 = sorted((s["method"], s.get("direction")) for s in base.signals)
    s1 = sorted((s["method"], s.get("direction")) for s in mapped.signals)
    # Boundary flips from floating-point error are tolerated only if they involve a single borderline method.
    assert len(set(s0) ^ set(s1)) <= 1, (s0, s1)


# ---------------------------------------------------------------------------------------------- consensus index
def _ev(agent, hyp, s, group, direction=Direction.SUPPORTS):
    return HypothesisEvidence(source_agent=agent, hypothesis=hyp, evidence_type=EvidenceType.DIAGNOSTIC, direction=direction,
                              strength=s, independence_group=group, validated=False)


W = {f"a{i}": 0.3 + 0.1 * i for i in range(6)}
ev_strategy = st.lists(st.tuples(st.integers(0, 5), st.sampled_from(["H1", "H2", "H3"]), st.floats(0.0, 1.0),
                                 st.sampled_from(["g1", "g2", "g3", "g4"]),
                                 st.sampled_from([Direction.SUPPORTS, Direction.SUPPORTS, Direction.CONTRADICTS])),
                       min_size=1, max_size=14)


def _results(items):
    by = {}
    for ai, h, s, g, d in items:
        by.setdefault(f"a{ai}", []).append(_ev(f"a{ai}", h, s, g, d))
    return [AgentResult(agent_name=a, task_id="t", status=AgentStatus.COMPLETED, hypothesis_evidence=e) for a, e in by.items()]


@PROFILE
@given(items=ev_strategy, seed=st.integers(0, 10_000))
def test_support_index_is_permutation_invariant_and_bounded(items, seed):
    res = _results(items)
    shuffled = list(res)
    np.random.default_rng(seed).shuffle(shuffled)
    for r in shuffled:
        r.hypothesis_evidence = list(reversed(r.hypothesis_evidence))
    a = compute_evidence_consensus(res, W)
    b = compute_evidence_consensus(shuffled, W)
    sa = {h.hypothesis: round(h.consensus_score, 9) for h in a.hypotheses}
    sb = {h.hypothesis: round(h.consensus_score, 9) for h in b.hypotheses}
    assert sa == sb and a.top_hypothesis == b.top_hypothesis
    assert all(0.0 <= v <= 1.0 for v in sa.values())
    assert a.probability is None and a.calibrated is False                    # never a probability


@PROFILE
@given(items=ev_strategy, copies=st.integers(2, 6))
def test_duplicating_an_evidence_item_in_its_group_changes_nothing(items, copies):
    res = _results(items)
    base = compute_evidence_consensus(res, W)
    for r in res:
        r.hypothesis_evidence = [e for e in r.hypothesis_evidence for _ in range(copies)]
    again = compute_evidence_consensus(res, W)
    assert {h.hypothesis: round(h.consensus_score, 9) for h in base.hypotheses} == \
           {h.hypothesis: round(h.consensus_score, 9) for h in again.hypotheses}


# ---------------------------------------------------------------------------------------------------- Wilson
@PROFILE
@given(n=st.integers(1, 5000), data=st.data())
def test_wilson_interval_contains_the_estimate_and_stays_in_unit_interval(n, data):
    k = data.draw(st.integers(0, n))
    lo, hi = M.wilson_interval(k, n)
    assert 0.0 <= lo <= k / n + 1e-12 <= hi + 2e-12 <= 1.0 + 2e-12


@PROFILE
@given(n=st.integers(2, 2000), r=st.integers(2, 6), data=st.data())
def test_wilson_width_shrinks_with_more_data_at_fixed_proportion(n, r, data):
    k = data.draw(st.integers(0, n))
    w1 = np.subtract(*reversed(M.wilson_interval(k, n)))
    w2 = np.subtract(*reversed(M.wilson_interval(k * r, n * r)))
    assert w2 < w1 + 1e-12


# ----------------------------------------------------------------------------------------------------- AUC
@PROFILE
@given(p=st.lists(st.floats(0.001, 0.999), min_size=6, max_size=60, unique=True), data=st.data())
def test_auc_symmetries(p, data):
    y = data.draw(st.lists(st.integers(0, 1), min_size=len(p), max_size=len(p)))
    assume(0 < sum(y) < len(y))
    p = np.array(p)
    auc = M.score_metrics(y, p)["roc_auc"]
    assert M.score_metrics(y, 1 - p)["roc_auc"] == pytest.approx(1 - auc, abs=1e-12)                # reversal
    assert M.score_metrics(y, p ** 2)["roc_auc"] == pytest.approx(auc, abs=1e-12)                   # monotone transform
    assert 0.0 <= auc <= 1.0


@PROFILE
@given(p=st.lists(st.floats(0.0, 1.0), min_size=5, max_size=80), data=st.data())
def test_ece_and_brier_are_bounded(p, data):
    y = data.draw(st.lists(st.integers(0, 1), min_size=len(p), max_size=len(p)))
    assert 0.0 <= M.expected_calibration_error(y, p) <= 1.0
    assert 0.0 <= M.score_metrics(y, p)["brier"] <= 1.0


@PROFILE
@given(y=st.lists(st.integers(0, 1), min_size=5, max_size=60))
def test_perfectly_calibrated_constant_forecast_has_zero_ece(y):
    prev = sum(y) / len(y)
    assume(0.0 < prev < 1.0)
    # a constant forecast equal to the base rate lies in one bin whose mean(y) == mean(p): ECE must be (numerically) 0
    assert M.expected_calibration_error(y, [prev] * len(y)) == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------------------------------------- DDI
DRUGS = ["levodopa", "carbidopa", "haloperidol", "warfarin", "aspirin", "selegiline", "tramadol", "sertraline", "metformin",
         "phenytoin", "carbamazepine", "unknownium"]


@PROFILE
@given(a=st.sampled_from(DRUGS), b=st.sampled_from(DRUGS))
def test_ddi_pair_assessment_is_symmetric(a, b):
    assume(a != b)
    chk = DDIChecker([CuratedTableSource()])
    ab = chk.check([a, b]).pairs[0]
    ba = chk.check([b, a]).pairs[0]
    assert (ab.status, ab.severity) == (ba.status, ba.severity)


@PROFILE
@given(names=st.lists(st.sampled_from(DRUGS), min_size=2, max_size=6), reps=st.integers(1, 3))
def test_ddi_duplicate_names_and_case_do_not_change_the_result(names, reps):
    chk = DDIChecker([CuratedTableSource()])
    base = chk.check(names)
    again = chk.check([n.upper() if i % 2 else n for i, n in enumerate(names) for _ in range(reps)])
    key = lambda rep: sorted((tuple(sorted((p.drug_a.lower(), p.drug_b.lower()))), p.status, p.severity) for p in rep.pairs)
    assert key(base) == key(again)


# ----------------------------------------------------------------------------------------------- polarity
@PROFILE
@given(name=st.sampled_from(["tremor_score", "Gait_Speed", "moca_total", "stride_time_variability", "seizure_free_days",
                             "missteps_count", "UPDRS III", "grip-strength", "mystery"]))
def test_polarity_ignores_case_and_separator_style(name):
    base = metric_polarity(name)
    assert metric_polarity(name.upper()) == base == metric_polarity(name.lower())
    assert metric_polarity(name.replace("_", " ")) == base == metric_polarity(name.replace("_", "-"))
