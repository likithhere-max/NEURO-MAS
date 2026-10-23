"""Property-based mathematical regression tests (Phase 2, C). Only properties that hold BY DESIGN are asserted;
known non-properties are pinned explicitly so a future change must be deliberate."""
import math

from hypothesis import HealthCheck, given, settings, strategies as st

from app.core import metric_semantics as ms
from app.core.config import Settings
from app.core.consensus import compute_evidence_consensus
from app.core.evidence import Direction, EvidenceType, HypothesisEvidence
from app.schemas.agent import AgentResult, AgentStatus

W = {"a": 0.65, "b": 0.8, "c": 0.75, "d": 0.5}
S = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
HYPS = ["PD", "ET", "MS"]


def ev(agent, hyp, strength, group, direction=Direction.SUPPORTS):
    return HypothesisEvidence(source_agent=agent, hypothesis=hyp, evidence_type=EvidenceType.DIAGNOSTIC,
                              direction=direction, strength=strength, independence_group=group)


def run(items, weights=W):
    by_agent: dict[str, list] = {}
    for e in items:
        by_agent.setdefault(e.source_agent, []).append(e)
    results = [AgentResult(agent_name=a, task_id="t", status=AgentStatus.COMPLETED, hypothesis_evidence=es)
               for a, es in by_agent.items()]
    return compute_evidence_consensus(results, weights, 0.9)


def get(o, h):
    return next((x for x in o.hypotheses if x.hypothesis == h), None)


item = st.tuples(st.sampled_from(list(W)), st.sampled_from(HYPS), S, st.sampled_from(["g1", "g2", "g3", "g4"]),
                 st.sampled_from([Direction.SUPPORTS, Direction.CONTRADICTS]))
items = st.lists(item, min_size=1, max_size=8)
cfg = settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def build(raw):
    # one group per (agent, group) so groups are independent across agents
    return [ev(a, h, s, f"{a}:{g}", d) for a, h, s, g, d in raw]


@cfg
@given(items)
def test_scores_are_bounded_and_never_a_probability(raw):
    o = run(build(raw))
    assert o.probability is None and o.calibrated is False
    assert 0.0 <= o.disagreement <= 1.0
    for h in o.hypotheses:
        assert 0.0 <= h.consensus_score <= 1.0 and 0.0 <= h.contradiction_index <= 1.0
        assert math.isfinite(h.consensus_score)


@cfg
@given(items, st.sampled_from(["a", "b", "c", "d"]), S)
def test_added_contradiction_never_raises_a_hypothesis_rank(raw, agent, strength):
    base = build(raw)
    before = get(run(base), "PD")
    after = get(run(base + [ev(agent, "PD", strength, f"{agent}:new_contra", Direction.CONTRADICTS)]), "PD")
    if before is not None:
        assert after.consensus_score <= before.consensus_score + 1e-12


@cfg
@given(items, S, st.sampled_from(["a", "b", "c", "d"]))
def test_evidence_about_one_hypothesis_does_not_change_another_hypotheses_own_score(raw, strength, agent):
    base = [e for e in build(raw) if e.hypothesis != "MS"]
    before = get(run(base), "PD")
    after = get(run(base + [ev(agent, "MS", strength, f"{agent}:ms")]), "PD")
    if before is not None:
        assert after.consensus_score == before.consensus_score
        assert after.contradiction_index == before.contradiction_index


@cfg
@given(items, st.floats(min_value=0.1, max_value=1.0))
def test_uniform_down_scaling_of_reliability_weights_changes_nothing(raw, c):
    """Weights are reliabilities clipped to [0,1] (so up-scaling is NOT invariant by design); within range the
    index is a weighted mean and must be scale-free. Tolerance covers the 3-decimal rounding of published scores."""
    base = build(raw)
    o1, o2 = run(base), run(base, {k: v * c for k, v in W.items()})
    for h in o1.hypotheses:
        assert math.isclose(h.consensus_score, get(o2, h.hypothesis).consensus_score, abs_tol=1.1e-3)


def test_known_non_property_adding_weaker_independent_support_lowers_the_index():
    """DESIGN FACT, pinned on purpose: the support index is a reliability-weighted MEAN, so corroboration by a weaker
    independent source lowers it. It is an ordering aid, not accumulating belief; see CONSENSUS_DESIGN.md."""
    strong = get(run([ev("a", "PD", 0.9, "a:g")]), "PD").consensus_score
    mixed = get(run([ev("a", "PD", 0.9, "a:g"), ev("b", "PD", 0.3, "b:g")]), "PD").consensus_score
    assert mixed < strong


# ---- change classification ----
@cfg
@given(st.floats(min_value=0.01, max_value=1e4, allow_nan=False), st.floats(min_value=0.01, max_value=1e4, allow_nan=False),
       st.sampled_from(["updrs_total", "tremor_amplitude", "gait_speed", "moca"]))
def test_polarity_antisymmetry_for_deteriorating_and_improving(b, c, metric):
    """Swapping baseline and current (equal-magnitude relative change in the opposite direction) swaps
    deteriorating<->improving only when the relative changes are symmetric; we assert the weaker, always-true
    statement: a change can never be BOTH deteriorating in one direction and in the reverse direction."""
    s = Settings(_env_file=None)
    orig = ms.get_settings
    ms.get_settings = lambda: s
    try:
        fwd = ms.classify_change(metric, b, c)["classification"]
        rev = ms.classify_change(metric, c, b)["classification"]
    finally:
        ms.get_settings = orig
    assert not (fwd == "deteriorating" and rev == "deteriorating")
    assert not (fwd == "improving" and rev == "improving")


@cfg
@given(st.floats(min_value=0.01, max_value=1e3, allow_nan=False), st.sampled_from(["updrs_total", "gait_speed"]))
def test_identity_is_always_stable(b, metric):
    s = Settings(_env_file=None)
    orig = ms.get_settings
    ms.get_settings = lambda: s
    try:
        assert ms.classify_change(metric, b, b)["classification"] == "stable"
    finally:
        ms.get_settings = orig


# ---- determinism of the simulation harness ----
def test_anomaly_simulation_is_seed_reproducible():
    from ml.simulation.anomaly_study import study
    a = study(reps=4, seed=123, N=30)
    b = study(reps=4, seed=123, N=30)
    assert a == b
    c = study(reps=4, seed=124, N=30)
    assert a != c
