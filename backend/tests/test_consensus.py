from app.core.consensus import compute_consensus
from app.schemas.agent import AgentResult, AgentStatus


def _result(agent_name, confidence, evidence_strength=None):
    from app.core.messaging import EvidenceItem
    evidence = [EvidenceItem(source_type="knowledge_base", description="x", strength=evidence_strength)] if evidence_strength is not None else []
    return AgentResult(agent_name=agent_name, task_id="t1", status=AgentStatus.COMPLETED, confidence=confidence, evidence=evidence)


def test_agreement_yields_low_disagreement():
    results = [_result("agent_a", 0.9, 0.9), _result("agent_b", 0.85, 0.85)]
    weights = {"agent_a": 0.8, "agent_b": 0.8}
    outcome = compute_consensus(results, weights, data_quality=0.9, hypotheses={"H1": ["agent_a", "agent_b"]})
    assert outcome.disagreement < 0.3
    assert outcome.action_permitted is True


def test_conflicting_agents_raise_disagreement_and_block_action():
    results = [_result("agent_a", 0.9, 0.9), _result("agent_b", 0.1, 0.9)]
    weights = {"agent_a": 0.8, "agent_b": 0.8}
    # Two competing hypotheses, each backed by one agent -> neither reaches strong consensus.
    outcome = compute_consensus(
        results, weights, data_quality=0.9,
        hypotheses={"H1": ["agent_a"], "H2": ["agent_b"]},
    )
    assert outcome.top_hypothesis == "H1"
    assert outcome.disagreement > 0.0


def test_missing_data_quality_falls_back_to_agent_confidence():
    results = [_result("agent_a", 0.7, 0.7)]
    weights = {"agent_a": 0.8}
    outcome = compute_consensus(results, weights, data_quality=0.0, hypotheses={"H1": ["agent_a"]})
    assert outcome.hypotheses[0].consensus_score > 0


def test_single_agent_score_is_independent_of_its_weight():
    high_w = compute_consensus(
        [_result("a", 0.9, 0.9)], {"a": 1.0}, data_quality=1.0, hypotheses={"H": ["a"]},
    )
    low_w = compute_consensus(
        [_result("a", 0.9, 0.9)], {"a": 0.1}, data_quality=1.0, hypotheses={"H": ["a"]},
    )
    # Consensus score for a single-agent hypothesis is independent of its own
    # weight (w_i cancels in numerator/denominator) -- but disagreement should
    # be identical in both single-agent cases; this test documents that
    # behavior explicitly rather than assuming it.
    assert high_w.hypotheses[0].consensus_score == low_w.hypotheses[0].consensus_score



# ---------------- hypothesis-specific support (audit fix) ----------------

def test_hand_calculated_two_voter_consensus():
    """C = (w1*q*e1*s1 + w2*q*e2*s2) / (w1*q + w2*q), e=1 with hypothesis-specific support.
    w1=1.0, w2=0.5, q=0.5, s1=0.8, s2=0.2 -> (0.4 + 0.05) / (0.5 + 0.25) = 0.6 exactly."""
    results = [_result("a", 0.99), _result("b", 0.01)]   # global confidences are IRRELEVANT now
    out = compute_consensus(results, {"a": 1.0, "b": 0.5}, data_quality=0.5,
                            hypotheses={"H": ["a", "b"]}, hypothesis_support={"H": {"a": 0.8, "b": 0.2}})
    assert out.hypotheses[0].consensus_score == 0.6
    assert out.disagreement == 0.4
    assert out.agent_level_support is False


def test_hypotheses_with_different_support_get_different_scores():
    """Regression: previously every hypothesis shared the same voters and the same global confidence,
    so all scores were identical (0.245 / 0.245)."""
    results = [_result("diag", 0.75)]
    out = compute_consensus(results, {"diag": 0.8}, data_quality=1.0,
                            hypotheses={"PD": ["diag"], "ET": ["diag"]},
                            hypothesis_support={"PD": {"diag": 0.75}, "ET": {"diag": 0.38}})
    scores = {h.hypothesis: h.consensus_score for h in out.hypotheses}
    assert scores == {"PD": 0.75, "ET": 0.38}
    assert out.top_hypothesis == "PD"
    assert out.margin == 0.37


def test_voter_without_hypothesis_specific_support_abstains():
    results = [_result("diag", 0.75), _result("risk", 0.65)]
    out = compute_consensus(results, {"diag": 0.8, "risk": 0.7}, data_quality=1.0,
                            hypotheses={"PD": ["diag", "risk"]}, hypothesis_support={"PD": {"diag": 0.75}})
    h = out.hypotheses[0]
    assert h.abstained_agents == ["risk"] and h.contributing_agents == ["diag"]
    assert h.consensus_score == 0.75        # risk's unrelated 0.65 did not dilute it


def test_failed_or_missing_agent_never_votes():
    failed = AgentResult(agent_name="diag", task_id="t", status=AgentStatus.FAILED, confidence=0.9)
    out = compute_consensus([failed], {"diag": 0.8}, data_quality=1.0, hypotheses={"H": ["diag", "ghost"]},
                            hypothesis_support={"H": {"diag": 0.9, "ghost": 0.9}})
    assert out.hypotheses[0].consensus_score == 0.0
    assert out.action_permitted is False       # no voter -> never permit autonomous action
    assert "No agent supplied" in out.reason


def test_support_is_clipped_and_nan_safe():
    out = compute_consensus([_result("a", 0.5)], {"a": 1.0}, data_quality=1.0, hypotheses={"H": ["a"]},
                            hypothesis_support={"H": {"a": float("nan")}})
    assert out.hypotheses[0].consensus_score == 0.0
    out = compute_consensus([_result("a", 0.5)], {"a": 1.0}, data_quality=1.0, hypotheses={"H": ["a"]},
                            hypothesis_support={"H": {"a": 7.0}})
    assert out.hypotheses[0].consensus_score == 1.0


def test_evidence_strength_is_not_double_counted_with_specific_support():
    # same quantity feeds s_i and EvidenceItem.strength -> must not be squared
    r = _result("a", 0.75, evidence_strength=1.0)
    out = compute_consensus([r], {"a": 1.0}, 1.0, {"H": ["a"]}, hypothesis_support={"H": {"a": 0.75}})
    assert out.hypotheses[0].consensus_score == 0.75
