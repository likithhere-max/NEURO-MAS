"""Phase 3 independent reference tests (part 2): risk heuristic, consensus, change classification, evaluation metrics,
input-domain checks.  Expected values are hand-derived (derivation in the comments) or come from a different
implementation (scipy / numpy / stdlib)."""
import math

import numpy as np
import pandas as pd
import pytest

from app.agents.risk import RiskStratificationAgent, _bucket
from app.core.consensus import compute_evidence_consensus
from app.core.evidence import Direction, EvidenceType, HypothesisEvidence
from app.core.metric_semantics import classify_change
from app.schemas.agent import AgentResult, AgentStatus, AgentTask
from app.schemas.patient_state import AdherenceState, MedicationState, PatientState
from ml.evalkit import metrics as M


# =============================================================================================== risk heuristic
@pytest.fixture(autouse=True)
def _no_llm_narrative(monkeypatch):
    async def none(*a, **k):
        return None
    monkeypatch.setattr("app.agents.risk._explain_risk_with_gemini", none)


def _risk(state=None, **upstream):
    state = state or PatientState(patient_id="p")
    task = AgentTask(workflow_id="w", patient_id="p", agent_name="risk_stratification_agent", reason="t",
                     input_payload={"upstream_results": upstream})
    import asyncio
    return asyncio.run(RiskStratificationAgent().execute(task, state))


@pytest.mark.parametrize("score,bucket", [(0.0, "low"), (0.2499, "low"), (0.25, "medium"), (0.4999, "medium"),
                                          (0.5, "high"), (0.7499, "high"), (0.75, "critical"), (1.0, "critical")])
def test_bucket_edges_are_lower_inclusive(score, bucket):
    assert _bucket(score) == bucket


def test_fall_score_hand_sum_and_bucket():
    # gait_slowing .40 + deteriorating-longitudinal .25 + >=3 missed check-ins .10 = 0.75 -> 'critical' (>= .75)
    st = PatientState(patient_id="p", adherence_state=AdherenceState(missed_checkins_7d=3))
    r = _risk(st, signal_analysis_agent={"findings": [{"label": "gait_slowing"}]},
              longitudinal_agent={"raw_payload": {"deteriorating_metrics": ["x"]}, "findings": [1]})
    assert r.raw_payload["heuristic_scores"]["fall_risk"] == 0.75
    assert r.raw_payload["overall_bucket"] == "critical" and r.requires_human_review is True


def test_fall_score_is_capped_at_one():
    st = PatientState(patient_id="p", adherence_state=AdherenceState(missed_checkins_7d=9))
    r = _risk(st, signal_analysis_agent={"findings": [{"label": "gait_slowing"}, {"label": "tremor_increase"}]},
              longitudinal_agent={"raw_payload": {"deteriorating_metrics": ["x"]}, "findings": [1]})
    assert r.raw_payload["heuristic_scores"]["fall_risk"] == 1.0          # .4+.25+.25+.1 = 1.0 (cap not exceeded)


def test_seizure_score_hand_cases_and_adherence_boundary():
    def seizure(adh, recent=0, urgency=None):
        st = PatientState(patient_id="p", medication_state=MedicationState(adherence_pct_30d=adh),
                          seizure_history=[{"recent": True}] * recent)
        up = {"data_intake_agent": {"raw_payload": {"urgency": urgency}}} if urgency else {}
        return _risk(st, **up).raw_payload["heuristic_scores"]["seizure_risk"]
    assert seizure(0.79) == 0.35                       # below 0.8
    assert seizure(0.80) == 0.0                        # exactly 0.8 is NOT below the threshold
    assert seizure(0.79, recent=2) == 0.55             # .35 + .1*2
    assert seizure(0.79, recent=9) == 0.75             # recent-seizure term capped at .4 -> .35 + .4
    assert seizure(0.79, recent=2, urgency="high") == 0.85   # + .3
    assert seizure(1.0, urgency="critical") == 0.3


def test_deterioration_score_hand_cases():
    def det(conf, deteriorating=True):
        lg = {"raw_payload": {"deteriorating_metrics": ["m"] if deteriorating else []}, "findings": [1]}
        dg = {"raw_payload": {"candidate_conditions": [{"confidence": conf}]}}
        return _risk(longitudinal_agent=lg, neuro_diagnostic_agent=dg).raw_payload["heuristic_scores"]["deterioration_risk"]
    assert det(0.51) == 0.75 and det(0.50) == 0.5 and det(0.9, deteriorating=False) == 0.25


def test_risk_confidence_is_input_coverage_not_probability():
    r = _risk()                                         # nothing available
    assert r.confidence == 0.0
    assert "risk_scores_low_may_reflect_missing_inputs_not_low_risk" in r.warnings
    assert "confidence_is_input_coverage_not_probability" in r.warnings


# =============================================================================================== consensus index
def _res(agent, evid, status=AgentStatus.COMPLETED):
    return AgentResult(agent_name=agent, task_id="t", status=status, hypothesis_evidence=evid)


def _ev(agent, hyp, s, direction=Direction.SUPPORTS, group="", typ=EvidenceType.DIAGNOSTIC, **kw):
    return HypothesisEvidence(source_agent=agent, hypothesis=hyp, evidence_type=typ, direction=direction, strength=s,
                              independence_group=group, validated=kw.pop("validated", False), **kw)


W = {"a": 0.65, "b": 0.9, "c": 0.5}


def test_support_index_is_reliability_weighted_mean_over_independent_groups():
    # (0.6*0.65 + 0.8*0.90) / (0.65 + 0.90) = 1.11/1.55 = 0.716129
    out = compute_evidence_consensus([_res("a", [_ev("a", "H", 0.6, group="g1")]), _res("b", [_ev("b", "H", 0.8, group="g2")])], W)
    h = out.hypotheses[0]
    assert h.consensus_score == pytest.approx(0.716, abs=5e-4) and h.independent_sources == 2
    assert out.probability is None and out.calibrated is False and out.score_kind == "uncalibrated_support_index"


def test_correlated_items_in_one_group_count_once_with_strongest_strength():
    one = compute_evidence_consensus([_res("a", [_ev("a", "H", 0.6, group="shared"), _ev("a", "H", 0.8, group="shared")])], W)
    assert one.hypotheses[0].independent_sources == 1 and one.hypotheses[0].consensus_score == pytest.approx(0.8)
    dup = compute_evidence_consensus([_res("a", [_ev("a", "H", 0.8, group="g")] * 5)], W)
    assert dup.hypotheses[0].consensus_score == pytest.approx(0.8)          # repeating an item adds nothing


def test_contradiction_subtracts_and_can_veto():
    # support .8 (single group) ; strongest independent contradiction .3 -> rank .5
    r = compute_evidence_consensus([_res("a", [_ev("a", "H", 0.8, group="g"), _ev("b", "H", 0.3, Direction.CONTRADICTS, group="k")])], W)
    assert r.hypotheses[0].consensus_score == pytest.approx(0.5) and r.hypotheses[0].status == "supported"
    v = compute_evidence_consensus([_res("a", [_ev("a", "H", 0.5, group="g"), _ev("b", "H", 0.5, Direction.CONTRADICTS, group="k")])], W)
    assert v.hypotheses[0].status == "contradicted" and v.top_hypothesis is None and not v.action_permitted


def test_non_completed_agents_never_vote_and_are_reported():
    out = compute_evidence_consensus([_res("a", [_ev("a", "H", 0.9, group="g")], AgentStatus.TIMED_OUT),
                                      _res("b", [_ev("b", "H", 0.4, group="h")])], W)
    assert out.non_voting_agents == {"a": "timed_out"} and out.hypotheses[0].consensus_score == pytest.approx(0.4)


def test_ties_leave_no_unique_leader_and_block_action():
    out = compute_evidence_consensus([_res("a", [_ev("a", "H1", 0.80, group="g"), _ev("a", "H2", 0.79, group="g")])], W)
    assert out.top_hypothesis is None and set(out.ties) == {"H1", "H2"} and out.action_permitted is False


def test_type_policy_knowledge_retrieval_and_unvalidated_models_never_rank():
    out = compute_evidence_consensus([_res("a", [_ev("a", "H", 0.99, typ=EvidenceType.KNOWLEDGE_RETRIEVAL),
                                                 _ev("b", "H", 0.99, typ=EvidenceType.VALIDATED_MODEL, validated=False),
                                                 _ev("c", "H", 0.99, typ=EvidenceType.DATA_QUALITY)])], W)
    h = out.hypotheses[0]
    assert h.status == "unsupported" and out.top_hypothesis is None and len(h.excluded_evidence) == 3


def test_independence_is_declared_not_verified__known_limitation():
    """PINS A KNOWN LIMITATION (docs/phase3/STATISTICAL_VALIDATION.md, consensus section): independence comes from the
    emitter's `independence_group`.  Two agents that secretly read the same input but use their default (per-agent)
    group are counted as two independent sources, so shared errors can look like corroboration.  Closing this needs
    input lineage, which the evidence contract does not carry."""
    out = compute_evidence_consensus([_res("a", [_ev("a", "H", 0.7)]), _res("b", [_ev("b", "H", 0.7)])], W)
    assert out.hypotheses[0].independent_sources == 2           # <- cannot detect the shared upstream


def test_single_source_is_labelled_uncorroborated():
    out = compute_evidence_consensus([_res("a", [_ev("a", "H", 0.7, group="g")])], W)
    assert any("single independent source" in n for n in out.hypotheses[0].notes)


# =============================================================================================== change classification
@pytest.mark.parametrize("metric,b,c,cls,pct", [
    ("tremor_amplitude", 10.0, 12.0, "deteriorating", 0.2),          # +20.0% exactly (inclusive)
    ("tremor_amplitude", 10.0, 11.9, "stable", 0.19),
    ("tremor_amplitude", 10.0, 10.5, "stable", 0.05),                # +5%: outside the <5% band but adverse change < 20% -> still stable
    ("gait_speed_m_s", 1.0, 0.8, "deteriorating", -0.2),             # -20% on a lower-is-worse metric
    ("gait_speed_m_s", 1.0, 1.2, "improving", 0.2),
])
def test_classification_hand_cases(metric, b, c, cls, pct):
    r = classify_change(metric, b, c)
    assert r["classification"] == cls and r["delta_pct"] == pytest.approx(pct, abs=1e-9)


def test_negative_baseline_uses_absolute_value_in_denominator_hand_case():
    # -4 -> -2 on a higher-is-worse metric: delta = +2 ; denominator |baseline| = 4 ; +50% -> deteriorating
    r = classify_change("tremor_amplitude", -4.0, -2.0)
    assert r["classification"] == "deteriorating" and r["delta_pct"] == pytest.approx(0.5)


# =============================================================================================== evaluation metrics
def test_wilson_matches_scipy_and_analytic_edges():
    from scipy.stats import binomtest
    for k, n in [(8, 10), (1, 30), (50, 50), (0, 10), (13, 200)]:
        ref = binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
        lo, hi = M.wilson_interval(k, n)
        assert lo == pytest.approx(ref.low, abs=1e-9) and hi == pytest.approx(ref.high, abs=1e-9)
    z2 = 1.959963984540054 ** 2
    assert M.wilson_interval(0, 10)[1] == pytest.approx(z2 / (10 + z2))      # upper bound for 0/10 = z^2/(n+z^2)
    assert M.wilson_interval(10, 10)[0] == pytest.approx(10 / (10 + z2))
    assert M.wilson_interval(0, 0) is None


def test_ece_and_reliability_hand_values():
    y = [1, 0, 1, 1, 0, 0]; p = [0.9, 0.2, 0.8, 0.6, 0.4, 0.1]          # six separate bins: ECE = mean |y - p| = 1.4/6
    assert M.expected_calibration_error(y, p) == pytest.approx(1.4 / 6)
    assert M.expected_calibration_error([1, 0], [0.61, 0.69]) == pytest.approx(0.15)   # one bin: |0.5 - 0.65|
    assert M.expected_calibration_error([1, 1, 0, 0], [1.0, 1.0, 0.0, 0.0]) == 0.0       # p=1.0 lands in the last bin


def test_ppv_npv_bayes_hand_values():
    r = M.ppv_npv_at_prevalence(0.9, 0.8, 0.1)       # TP .09 FP .18 TN .72 FN .01
    assert r["ppv"] == pytest.approx(0.09 / 0.27) and r["npv"] == pytest.approx(0.72 / 0.73)
    with pytest.raises(ValueError):
        M.ppv_npv_at_prevalence(0.9, 0.8, 0.0)


def test_threshold_metrics_hand_values_and_undefined_handling():
    y = [1, 1, 1, 0, 0, 0, 0, 0]; pred = [1, 1, 0, 1, 0, 0, 0, 0]     # tp2 fn1 fp1 tn4
    r = M.threshold_metrics(y, pred)
    assert (r["sensitivity"], r["specificity"], r["precision"]) == (pytest.approx(2 / 3), pytest.approx(4 / 5), pytest.approx(2 / 3))
    assert r["f1"] == pytest.approx(4 / 6) and r["balanced_accuracy"] == pytest.approx((2 / 3 + 4 / 5) / 2)
    one = M.threshold_metrics([0, 0, 0], [0, 0, 0])
    assert one["sensitivity"] is None and one["f1"] is None and "sensitivity" in one["undefined"]    # never 0 or NaN


def test_auc_equals_mann_whitney_probability_with_ties():
    y = np.array([1, 1, 0, 0, 1, 0]); p = np.array([0.8, 0.5, 0.5, 0.2, 0.9, 0.1])
    pos, neg = p[y == 1], p[y == 0]
    mw = np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])
    assert M.score_metrics(y, p)["roc_auc"] == pytest.approx(mw)
    assert M.score_metrics([1, 1, 1], [0.2, 0.4, 0.6])["roc_auc"] is None          # one class -> undefined, not 0.5


def test_brier_hand_value_and_constant_baseline():
    y = [1, 0, 1, 0]; p = [0.9, 0.2, 0.6, 0.4]                      # (0.01 + 0.04 + 0.16 + 0.16)/4 = 0.0925
    r = M.score_metrics(y, p)
    assert r["brier"] == pytest.approx(0.0925) and r["brier_constant_prevalence_baseline"] == pytest.approx(0.25)


def test_probability_inputs_are_validated():
    for bad in ([0.1, 1.1], [float("nan"), 0.2], [-0.1, 0.5]):
        with pytest.raises(ValueError):
            M.score_metrics([0, 1], bad)
    with pytest.raises(ValueError):
        M.score_metrics([0, 2], [0.1, 0.2])


def test_regression_metrics_hand_values():
    r = M.regression_metrics([1, 2, 3, 4], [1, 2, 3, 6])             # err 0,0,0,2 : MAE .5 RMSE 1 R2 = 1 - 4/5 = .2
    assert (r["mae"], r["rmse"], r["r2"]) == (pytest.approx(0.5), pytest.approx(1.0), pytest.approx(0.2))
    assert M.regression_metrics([3, 3, 3], [3, 3, 4])["r2"] is None


def test_cluster_bootstrap_is_deterministic_and_resamples_clusters_not_rows():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"g": np.repeat(np.arange(12), 5), "y": np.repeat(rng.integers(0, 2, 12), 5)})
    df["p"] = rng.random(len(df)) * 0.3 + df["y"] * 0.4
    a = M.cluster_bootstrap(df, "g", M.auc_stat, n_boot=200, seed=5); b = M.cluster_bootstrap(df, "g", M.auc_stat, n_boot=200, seed=5)
    assert a == b and a["n_clusters"] == 12 and a["lo"] <= a["point"] <= a["hi"]


# =============================================================================================== input-domain checks
def test_domain_identity_and_ordering_hand_cases():
    from app.ml.ood import InputDomain
    feats = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:RAP", "Jitter:DDP"]
    class S: mean_ = np.array([150., 200., 100., 0.005, 0.015]); scale_ = np.array([40., 60., 30., 0.003, 0.009])
    dom = InputDomain.from_scaler("voice", feats, S())
    ok = {"MDVP:Fo(Hz)": 150., "MDVP:Fhi(Hz)": 200., "MDVP:Flo(Hz)": 100., "MDVP:RAP": 0.01, "Jitter:DDP": 0.03}
    assert dom.check(ok).in_domain
    # DDP must equal 3*RAP within abs 1e-4 + 5%: 3*0.01 = 0.03 -> tolerance 0.0016 ; 0.0315 passes, 0.0320 fails
    assert dom.check({**ok, "Jitter:DDP": 0.0315}).in_domain
    assert dom.check({**ok, "Jitter:DDP": 0.0320}).status == "abstain_invalid_input"
    assert dom.check({**ok, "MDVP:Flo(Hz)": 210.}).status == "abstain_invalid_input"       # Flo <= Fo <= Fhi violated
    assert dom.check({**ok, "MDVP:RAP": -0.001}).status == "abstain_invalid_input"          # non-negative measure
    assert dom.check({**ok, "MDVP:Fo(Hz)": float("nan")}).status == "abstain_invalid_input"
    assert dom.check({k: v for k, v in ok.items() if k != "MDVP:RAP"}).status == "abstain_invalid_input"


def test_domain_provisional_z_limit_boundary():
    from app.ml.ood import InputDomain
    class S: mean_ = np.array([0.0]); scale_ = np.array([2.0])
    dom = InputDomain.from_scaler("generic", ["x"], S(), z_limit=8.0)
    assert dom.check({"x": 15.9}).in_domain                                      # |z| = 7.95
    assert dom.check({"x": 16.1}).status == "abstain_out_of_domain"              # |z| = 8.05


def test_mahalanobis_distance_matches_independent_numpy_computation():
    from sklearn.covariance import LedoitWolf
    from app.ml.ood import fit_domain
    rng = np.random.default_rng(1)
    A = rng.normal(size=(3, 3)); X = pd.DataFrame(rng.normal(size=(300, 3)) @ A, columns=["a", "b", "c"])
    dom = fit_domain("generic", X, range_margin=10.0)                           # wide range so only L5 can fire
    Z = ((X - X.mean()) / X.std(ddof=1)).to_numpy()
    cov = LedoitWolf().fit(Z).covariance_
    row = {"a": 1.0, "b": -2.0, "c": 0.5}
    z = (np.array([row[c] for c in X.columns]) - X.mean().to_numpy()) / X.std(ddof=1).to_numpy()
    mu = LedoitWolf().fit(Z).location_
    expect = float((z - mu) @ np.linalg.inv(cov) @ (z - mu))
    assert dom.check(row).mahalanobis_d2 == pytest.approx(expect, rel=1e-6)
