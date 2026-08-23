"""
Consensus engine (spec section 11).

Implements the weighted-consensus / disagreement model:

    C(H) = sum(w_i * q_i * e_i * s_i) / sum(w_i * q_i)
    D = 1 - max_H C(H)

    w_i = agent reliability weight (BaseNeuroAgent.reliability_weight)
    q_i = input/data quality for that agent's finding (state.data_completeness,
          falls back to the agent's own confidence when unavailable)
    e_i = evidence strength (mean of the agent's evidence[].strength, or its
          own confidence if it reported no discrete evidence items)
    s_i = agent's support for the hypothesis H, taken from AgentResult.confidence

Explicitly NOT claimed to be clinically validated -- it is a configurable,
auditable aggregation function, matching the spec's own caveat.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.core.config import get_settings
from app.schemas.agent import AgentResult


@dataclass
class HypothesisConsensus:
    hypothesis: str
    contributing_agents: list[str]
    consensus_score: float
    per_agent_support: dict[str, float] = field(default_factory=dict)


@dataclass
class ConsensusOutcome:
    hypotheses: list[HypothesisConsensus]
    disagreement: float
    top_hypothesis: str | None
    action_permitted: bool
    reason: str


def _evidence_strength(result: AgentResult) -> float:
    if result.evidence:
        return sum(e.strength for e in result.evidence) / len(result.evidence)
    return result.confidence


def compute_consensus(
    results: list[AgentResult],
    reliability_weights: dict[str, float],
    data_quality: float,
    hypotheses: dict[str, list[str]],
    # hypotheses: {hypothesis_label: [agent_name, ...] that "vote" on it}
) -> ConsensusOutcome:
    settings = get_settings()
    results_by_agent = {r.agent_name: r for r in results}

    hypothesis_scores: list[HypothesisConsensus] = []
    for label, voting_agents in hypotheses.items():
        numerator = 0.0
        denominator = 0.0
        per_agent: dict[str, float] = {}
        for agent_name in voting_agents:
            result = results_by_agent.get(agent_name)
            if result is None:
                continue
            w_i = reliability_weights.get(agent_name, 0.5)
            q_i = data_quality if data_quality > 0 else result.confidence
            e_i = _evidence_strength(result)
            s_i = result.confidence
            numerator += w_i * q_i * e_i * s_i
            denominator += w_i * q_i
            per_agent[agent_name] = round(s_i, 3)
        score = round(numerator / denominator, 3) if denominator > 0 else 0.0
        hypothesis_scores.append(HypothesisConsensus(
            hypothesis=label,
            contributing_agents=voting_agents,
            consensus_score=score,
            per_agent_support=per_agent,
        ))

    hypothesis_scores.sort(key=lambda h: h.consensus_score, reverse=True)
    top = hypothesis_scores[0] if hypothesis_scores else None
    max_c = top.consensus_score if top else 0.0
    disagreement = round(1 - max_c, 3)

    action_permitted = disagreement <= settings.consensus_disagreement_threshold
    reason = (
        f"Top hypothesis '{top.hypothesis if top else 'none'}' consensus={max_c}, "
        f"disagreement={disagreement} "
        + (
            f"<= threshold {settings.consensus_disagreement_threshold}: autonomous action permitted."
            if action_permitted
            else f"> threshold {settings.consensus_disagreement_threshold}: additional analysis required, "
                 f"autonomous action NOT permitted."
        )
    )

    return ConsensusOutcome(
        hypotheses=hypothesis_scores,
        disagreement=disagreement,
        top_hypothesis=top.hypothesis if top else None,
        action_permitted=action_permitted,
        reason=reason,
    )
