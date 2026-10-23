"""Evidence-typed ranking (app/core/consensus.compute_evidence_consensus) -- Phase 2, D2."""
import math

import pytest
from hypothesis import given, settings, strategies as st

from app.core.consensus import SCORE_KIND, compute_evidence_consensus
from app.core.evidence import Direction, EvidenceType, HypothesisEvidence
from app.schemas.agent import AgentResult, AgentStatus

W = {"diag": 0.65, "model": 0.8, "long": 0.75, "kb": 0.8, "intake": 0.9, "other": 0.5}


def ev(agent, hyp, strength, typ=EvidenceType.DIAGNOSTIC, direction=Direction.SUPPORTS, group="", **kw):
    return HypothesisEvidence(source_agent=agent, hypothesis=hyp, evidence_type=typ, direction=direction,
                              strength=strength, independence_group=group, **kw)


def res(agent, evidence, status=AgentStatus.COMPLETED):
    return AgentResult(agent_name=agent, task_id="t", status=status, hypothesis_evidence=evidence)


def cons(*results, dq=0.9):
    return compute_evidence_consensus(list(results), W, dq)


def test_output_is_labelled_uncalibrated_and_has_no_probability():
    o = cons(res("diag", [ev("diag", "PD", 0.7), ev("diag", "ET", 0.3)]))
    assert o.score_kind == SCORE_KIND and o.calibrated is False and o.probability is None
    assert o.top_hypothesis == "PD" and o.margin == 0.4
    assert any("NOT a probability" in c for c in o.caveats)


def test_knowledge_retrieval_never_votes_even_with_a_strength():
    o = cons(res("diag", [ev("diag", "PD", 0.5)]),
             res("kb", [ev("kb", "ET", 0.99, typ=EvidenceType.KNOWLEDGE_RETRIEVAL)]))
    assert o.top_hypothesis == "PD"
    et = [h for h in o.hypotheses if h.hypothesis == "ET"][0]
    assert et.status == "unsupported" and et.consensus_score == 0.0
    assert "circular" in et.excluded_evidence[0]["reason"]


def test_agent_without_hypothesis_evidence_does_not_vote():
    o = cons(res("diag", [ev("diag", "PD", 0.6)]), res("long", []), res("other", []))
    assert o.hypotheses[0].contributing_agents == ["diag"]


def test_no_evidence_at_all_gives_no_leader_and_blocks_action():
    o = cons(res("long", []))
    assert o.top_hypothesis is None and o.action_permitted is False and o.disagreement == 1.0


def test_correlated_evidence_in_one_group_is_not_double_counted():
    one = cons(res("diag", [ev("diag", "PD", 0.6, group="symptom_text")]))
    dup = cons(res("diag", [ev("diag", "PD", 0.6, group="symptom_text"), ev("diag", "PD", 0.55, group="symptom_text")]))
    assert one.hypotheses[0].consensus_score == dup.hypotheses[0].consensus_score == 0.6
    assert dup.hypotheses[0].independent_sources == 1


def test_independent_groups_are_averaged_by_reliability_weight():
    # diag w=.65 s=.8 ; model w=.8 s=.4  ->  (0.52 + 0.32)/1.45 = 0.84/1.45 = 0.57931
    o = cons(res("diag", [ev("diag", "PD", 0.8, group="a")]),
             res("model", [ev("model", "PD", 0.4, typ=EvidenceType.VALIDATED_MODEL, group="b", validated=True)]))
    assert o.hypotheses[0].consensus_score == pytest.approx(0.84 / 1.45, abs=1e-3)
    assert o.hypotheses[0].independent_sources == 2


def test_single_source_is_flagged_as_not_corroborated():
    o = cons(res("diag", [ev("diag", "PD", 0.8)]))
    assert any("not corroborated" in n for n in o.hypotheses[0].notes)


def test_contradicting_evidence_demotes_and_blocks_top():
    o = cons(res("diag", [ev("diag", "PD", 0.8), ev("diag", "ET", 0.5)]),
             res("model", [ev("model", "PD", 0.9, typ=EvidenceType.VALIDATED_MODEL, direction=Direction.CONTRADICTS,
                              group="m", validated=True)]))
    pd = [h for h in o.hypotheses if h.hypothesis == "PD"][0]
    assert pd.status == "contradicted" and pd.consensus_score == 0.0
    assert o.top_hypothesis == "ET"


def test_partial_contradiction_reduces_rank_score():
    o = cons(res("diag", [ev("diag", "PD", 0.8)]),
             res("model", [ev("model", "PD", 0.3, typ=EvidenceType.DIAGNOSTIC, direction=Direction.CONTRADICTS, group="m")]))
    assert o.hypotheses[0].consensus_score == pytest.approx(0.5)


def test_tie_leaves_no_leader_and_blocks_action():
    o = cons(res("diag", [ev("diag", "PD", 0.70), ev("diag", "ET", 0.69)]))
    assert o.top_hypothesis is None and sorted(o.ties) == ["ET", "PD"]
    assert o.action_permitted is False and o.disagreement == 1.0 and "tie" in o.reason


def test_failed_timed_out_and_abstained_agents_are_listed_and_excluded():
    for status in (AgentStatus.FAILED, AgentStatus.TIMED_OUT, AgentStatus.ABSTAINED, AgentStatus.INSUFFICIENT_DATA):
        o = cons(res("diag", [ev("diag", "PD", 0.9)], status=status))
        assert o.top_hypothesis is None
        assert o.non_voting_agents == {"diag": status.value}
        assert o.hypotheses[0].excluded_evidence[0]["reason"] == f"agent_status:{status.value}"


def test_out_of_domain_model_output_is_excluded():
    o = cons(res("model", [ev("model", "PD", 0.95, typ=EvidenceType.VALIDATED_MODEL, validated=True, out_of_domain=True)]))
    assert o.top_hypothesis is None
    assert "out_of_domain" in o.hypotheses[0].excluded_evidence[0]["reason"]


def test_unvalidated_model_is_reported_but_not_ranked():
    o = cons(res("model", [ev("model", "PD", 0.95, typ=EvidenceType.VALIDATED_MODEL, validated=False)]))
    assert o.top_hypothesis is None
    assert "validation" in o.hypotheses[0].excluded_evidence[0]["reason"]


def test_data_quality_evidence_can_contradict_but_never_support():
    sup = ev("intake", "PD", 0.9, typ=EvidenceType.DATA_QUALITY)
    assert sup.votes_for_ranking()[0] is False
    con = ev("intake", "PD", 0.9, typ=EvidenceType.DATA_QUALITY, direction=Direction.CONTRADICTS)
    assert con.votes_for_ranking()[0] is True


def test_unscored_and_nonfinite_strength_never_votes():
    assert ev("diag", "PD", None).votes_for_ranking()[0] is False
    assert ev("diag", "PD", float("nan")).strength is None
    assert ev("diag", "PD", 7.0).strength == 1.0 and ev("diag", "PD", -3.0).strength == 0.0


def test_low_data_quality_blocks_autonomous_action_but_keeps_the_ranking():
    o = cons(res("diag", [ev("diag", "PD", 0.95)]), dq=0.1)
    assert o.top_hypothesis == "PD" and o.action_permitted is False
    assert any("low data quality" in c for c in o.caveats)


def test_hypothesis_names_are_matched_case_and_space_insensitively():
    o = cons(res("diag", [ev("diag", "Parkinson  Disease", 0.6, group="a")]),
             res("model", [ev("model", "parkinson disease", 0.6, typ=EvidenceType.VALIDATED_MODEL, group="b", validated=True)]))
    assert len(o.hypotheses) == 1 and o.hypotheses[0].independent_sources == 2


_strength = st.floats(0, 1, allow_nan=False)


@settings(max_examples=60, deadline=None)
@given(a=_strength, b=_strength, c=_strength, order=st.permutations([0, 1, 2]))
def test_result_does_not_depend_on_agent_order(a, b, c, order):
    rs = [res("diag", [ev("diag", "PD", a, group="g1"), ev("diag", "ET", c, group="g1")]),
          res("model", [ev("model", "PD", b, typ=EvidenceType.VALIDATED_MODEL, group="g2", validated=True)]),
          res("long", [ev("long", "ET", b, typ=EvidenceType.LONGITUDINAL, group="g3")])]
    base = compute_evidence_consensus(rs, W, 0.9)
    perm = compute_evidence_consensus([rs[i] for i in order], W, 0.9)
    assert [(h.hypothesis, h.consensus_score, h.status) for h in base.hypotheses] == \
           [(h.hypothesis, h.consensus_score, h.status) for h in perm.hypotheses]
    assert base.top_hypothesis == perm.top_hypothesis


@settings(max_examples=60, deadline=None)
@given(s1=_strength, s2=_strength, other=_strength)
def test_raising_sole_support_never_lowers_the_hypothesis_score(s1, s2, other):
    lo, hi = sorted([s1, s2])
    def score(s):
        o = cons(res("diag", [ev("diag", "PD", s), ev("diag", "ET", other, group="x")]))
        return [h for h in o.hypotheses if h.hypothesis == "PD"][0].consensus_score
    assert score(hi) >= score(lo)


@settings(max_examples=60, deadline=None)
@given(vals=st.lists(_strength, min_size=1, max_size=5), cvals=st.lists(_strength, max_size=3))
def test_scores_stay_in_unit_interval_and_disagreement_consistent(vals, cvals):
    items = [ev("diag", "PD", v, group=f"g{i}") for i, v in enumerate(vals)] + \
            [ev("model", "PD", v, typ=EvidenceType.DIAGNOSTIC, direction=Direction.CONTRADICTS, group=f"c{i}") for i, v in enumerate(cvals)]
    o = cons(res("diag", items))
    for h in o.hypotheses:
        assert 0.0 <= h.consensus_score <= 1.0 and 0.0 <= h.contradiction_index <= 1.0
    assert 0.0 <= o.disagreement <= 1.0 and math.isfinite(o.disagreement)
    if o.top_hypothesis is None:
        assert o.disagreement == 1.0 and o.action_permitted is False
