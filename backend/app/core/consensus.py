"""
Consensus engine (spec section 11).

    C(H) = sum_i(w_i * q_i * e_i * s_i(H)) / sum_i(w_i * q_i)        over voters i of H
    D    = 1 - max_H C(H)

    w_i      agent reliability weight (BaseNeuroAgent.reliability_weight; informal, NOT validated)
    q_i      input data quality (state completeness; falls back to the agent's confidence if 0)
    e_i      evidence strength. 1.0 (neutral) whenever s_i is hypothesis-specific, because the
             evidence is already embodied in s_i [audit fix: previously e_i defaulted to the
             agent's confidence AND s_i was the same confidence, i.e. squared]; in legacy
             agent-level mode, mean EvidenceItem.strength or 1.0.
    s_i(H)   agent i's support for hypothesis H, in [0, 1].

Audit fixes (see docs/audit/AUDIT_REPORT.md):
  * Critical/High: s_i was the agent's single global confidence and every
    hypothesis had the same voters, so all hypotheses received an IDENTICAL
    score (measured: 0.245 vs 0.245) and the "consensus" could not
    discriminate between diseases. s_i is now HYPOTHESIS-SPECIFIC and supplied by the
    caller via `hypothesis_support[H][agent]`. A listed voter with no
    hypothesis-specific support ABSTAINS (reported in `abstained_agents`) instead of
    voting with an unrelated global number.
  * When `hypothesis_support` is omitted entirely the legacy agent-level behaviour is
    kept for backwards compatibility but is flagged `agent_level_support=True`.
  * `margin` (top1 - top2) is reported: D alone says nothing about how clearly the top
    hypothesis beats the runner-up.

Explicitly NOT clinically validated: an auditable aggregation function, not a
probability of disease.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.core.config import get_settings
from app.schemas.agent import AgentResult


@dataclass
class HypothesisConsensus:
    hypothesis: str
    contributing_agents: list[str]
    consensus_score: float
    per_agent_support: dict[str, float] = field(default_factory=dict)
    abstained_agents: list[str] = field(default_factory=list)


@dataclass
class ConsensusOutcome:
    hypotheses: list[HypothesisConsensus]
    disagreement: float
    top_hypothesis: str | None
    action_permitted: bool
    reason: str
    margin: float | None = None
    agent_level_support: bool = False


def _evidence_strength(result: AgentResult) -> float:
    strengths = [e.strength for e in result.evidence if e.strength is not None]
    return sum(strengths) / len(strengths) if strengths else 1.0


def _clip01(x: float) -> float:
    if x is None or not math.isfinite(x):
        return 0.0
    return min(1.0, max(0.0, x))


def compute_consensus(
    results: list[AgentResult],
    reliability_weights: dict[str, float],
    data_quality: float,
    hypotheses: dict[str, list[str]],
    hypothesis_support: dict[str, dict[str, float]] | None = None,
) -> ConsensusOutcome:
    settings = get_settings()
    results_by_agent = {r.agent_name: r for r in results}
    agent_level = hypothesis_support is None

    hypothesis_scores: list[HypothesisConsensus] = []
    for label, voting_agents in hypotheses.items():
        numerator = denominator = 0.0
        per_agent: dict[str, float] = {}
        abstained: list[str] = []
        for agent_name in voting_agents:
            result = results_by_agent.get(agent_name)
            if result is None or str(result.status) not in ("completed", "AgentStatus.COMPLETED"):
                abstained.append(agent_name)  # missing / failed / skipped agents never vote
                continue
            if agent_level:
                s_i = result.confidence
            else:
                s_i = (hypothesis_support or {}).get(label, {}).get(agent_name)
                if s_i is None:
                    abstained.append(agent_name)
                    continue
            s_i = _clip01(s_i)
            w_i = reliability_weights.get(agent_name, 0.5)
            q_i = _clip01(data_quality) if data_quality and data_quality > 0 else _clip01(result.confidence)
            # With hypothesis-specific support, the evidence is already embodied in s_i (the
            # diagnostic's EvidenceItem.strength IS its retrieval score), so multiplying again
            # would double-count it. e_i only applies in the legacy agent-level mode.
            e_i = 1.0 if not agent_level else _clip01(_evidence_strength(result))
            numerator += w_i * q_i * e_i * s_i
            denominator += w_i * q_i
            per_agent[agent_name] = round(s_i, 3)
        score = round(numerator / denominator, 3) if denominator > 0 else 0.0
        hypothesis_scores.append(HypothesisConsensus(
            hypothesis=label,
            contributing_agents=[a for a in voting_agents if a not in abstained],
            consensus_score=score,
            per_agent_support=per_agent,
            abstained_agents=abstained,
        ))

    hypothesis_scores.sort(key=lambda h: h.consensus_score, reverse=True)
    top = hypothesis_scores[0] if hypothesis_scores else None
    max_c = top.consensus_score if top else 0.0
    disagreement = round(1 - max_c, 3)
    margin = round(max_c - hypothesis_scores[1].consensus_score, 3) if len(hypothesis_scores) > 1 else None

    # With no voter at all the "consensus" is not evidence of anything: never permit action.
    any_voter = any(h.per_agent_support for h in hypothesis_scores)
    action_permitted = any_voter and disagreement <= settings.consensus_disagreement_threshold
    if not any_voter:
        reason = "No agent supplied hypothesis-specific support: no consensus can be formed; autonomous action NOT permitted."
    else:
        reason = (
            f"Top hypothesis '{top.hypothesis if top else 'none'}' consensus={max_c}, disagreement={disagreement}, "
            f"margin over runner-up={margin}; voters={sorted({a for h in hypothesis_scores for a in h.per_agent_support})}. "
            + (f"disagreement <= threshold {settings.consensus_disagreement_threshold}: autonomous action permitted."
               if action_permitted else
               f"disagreement > threshold {settings.consensus_disagreement_threshold}: additional analysis required, autonomous action NOT permitted.")
        )
    return ConsensusOutcome(hypotheses=hypothesis_scores, disagreement=disagreement,
                            top_hypothesis=top.hypothesis if top else None, action_permitted=action_permitted,
                            reason=reason, margin=margin, agent_level_support=agent_level)
