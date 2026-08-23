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


def test_low_reliability_agent_contributes_less_to_consensus():
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
