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
from app.schemas.agent import EVIDENCE_ELIGIBLE_STATUSES, AgentResult


@dataclass
class HypothesisConsensus:
    hypothesis: str
    contributing_agents: list[str]
    consensus_score: float          # legacy name. In evidence mode this is the UNCALIBRATED support index, not a probability
    per_agent_support: dict[str, float] = field(default_factory=dict)
    abstained_agents: list[str] = field(default_factory=list)
    # --- evidence mode (compute_evidence_consensus) ---
    status: str = "legacy"          # supported | contradicted | unsupported | legacy
    contradiction_index: float = 0.0
    independent_sources: int = 0
    excluded_evidence: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class ConsensusOutcome:
    hypotheses: list[HypothesisConsensus]
    disagreement: float
    top_hypothesis: str | None
    action_permitted: bool
    reason: str
    margin: float | None = None
    agent_level_support: bool = False
    # Honest labelling: no probability is produced because voter reliability has never been validated.
    score_kind: str = "legacy_weighted_mean"
    calibrated: bool = False
    probability: float | None = None
    ties: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    non_voting_agents: dict[str, str] = field(default_factory=dict)


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


# =============================================================================================================
# Evidence-typed ranking (Phase 2, D2).  See docs/phase2/CONSENSUS_DESIGN.md.
# =============================================================================================================
SCORE_KIND = "uncalibrated_support_index"
TIE_EPSILON = 0.02
MIN_DATA_QUALITY_FOR_ACTION = 0.4


def _key(name: str) -> str:
    return " ".join(str(name).split()).casefold()


def compute_evidence_consensus(
    results: list[AgentResult],
    reliability_weights: dict[str, float],
    data_quality: float | None = None,
    *,
    tie_epsilon: float = TIE_EPSILON,
) -> ConsensusOutcome:
    """
    Rank hypotheses from TYPED evidence. The output is an ordering plus an evidence summary, never a probability.

      1. Only evidence from COMPLETED agents is considered; other statuses are listed in `non_voting_agents`.
      2. Each item must pass `HypothesisEvidence.votes_for_ranking()` (type policy, scored, in-domain, validated if a
         model); rejected items are kept in `excluded_evidence` with the reason.
      3. Items sharing an `independence_group` are ONE observation (strongest kept): correlated evidence is not
         double counted.
      4. support_index(H)       = reliability-weighted mean over independent supporting groups, in [0, 1].
         contradiction_index(H) = strongest independent contradicting group, in [0, 1].
         rank_score(H)          = max(0, support_index - contradiction_index).
         Weights are informal and unvalidated, so the number orders hypotheses; it does not quantify belief.
      5. A hypothesis with no eligible support is "unsupported"; one whose contradiction >= support is "contradicted";
         neither can be the top hypothesis.
      6. Ties (|difference| <= tie_epsilon among the leaders) and a lone independent source are reported, and a tie
         leaves `top_hypothesis=None`.
      7. `disagreement` (kept for the safety gate) is 1 - top rank_score, and 1.0 whenever no unique supported leader
         exists. `action_permitted` additionally requires adequate data quality.
    """
    settings = get_settings()
    caveats: list[str] = []
    non_voting: dict[str, str] = {}
    names: dict[str, str] = {}
    sup: dict[str, dict[str, list[tuple[float, float, str]]]] = {}   # key -> group -> [(strength, weight, agent)]
    con: dict[str, dict[str, list[tuple[float, float, str]]]] = {}
    excluded: dict[str, list[dict]] = {}

    for r in results:
        status = str(getattr(r.status, "value", r.status))
        if status not in EVIDENCE_ELIGIBLE_STATUSES:
            non_voting[r.agent_name] = status
        for ev in r.hypothesis_evidence:
            k = _key(ev.hypothesis)
            names.setdefault(k, ev.hypothesis)
            sup.setdefault(k, {})
            con.setdefault(k, {})
            excluded.setdefault(k, [])
            if status not in EVIDENCE_ELIGIBLE_STATUSES:
                excluded[k].append({"agent": ev.source_agent, "type": ev.evidence_type.value,
                                    "reason": f"agent_status:{status}"})
                continue
            ok, why = ev.votes_for_ranking()
            if not ok:
                excluded[k].append({"agent": ev.source_agent, "type": ev.evidence_type.value, "reason": why})
                continue
            w = _clip01(reliability_weights.get(ev.source_agent, 0.5))
            bucket = sup if ev.direction.value == "supports" else con
            bucket[k].setdefault(ev.group, []).append((float(ev.strength), w, ev.source_agent))

    hyps: list[HypothesisConsensus] = []
    for k, name in names.items():
        groups = sup.get(k, {})
        sg = [(max(i[0] for i in items), max(i[1] for i in items), sorted({i[2] for i in items}))
              for items in groups.values()]
        wsum = sum(w for _, w, _ in sg)
        support = (sum(s * w for s, w, _ in sg) / wsum) if wsum > 0 else 0.0
        cgroups = con.get(k, {})
        contra = max((max(i[0] for i in items) for items in cgroups.values()), default=0.0)
        notes: list[str] = []
        if not sg:
            status = "unsupported"
            notes.append("no eligible supporting evidence")
        elif contra >= support:
            status = "contradicted"
            notes.append(f"contradicting evidence ({contra:.2f}) >= support index ({support:.2f})")
        else:
            status = "supported"
        if len(sg) == 1:
            notes.append("single independent source: not corroborated")
        rank = max(0.0, support - contra) if status == "supported" else 0.0
        agents = sorted({a for _, _, ags in sg for a in ags})
        hyps.append(HypothesisConsensus(
            hypothesis=name, contributing_agents=agents, consensus_score=round(rank, 3),
            per_agent_support={a: round(max(i[0] for items in groups.values() for i in items if i[2] == a), 3) for a in agents},
            abstained_agents=sorted(non_voting), status=status, contradiction_index=round(contra, 3),
            independent_sources=len(sg), excluded_evidence=excluded.get(k, []), notes=notes,
        ))

    order = {"supported": 0, "contradicted": 1, "unsupported": 2}
    hyps.sort(key=lambda h: (order.get(h.status, 3), -h.consensus_score, h.hypothesis.casefold()))
    supported = [h for h in hyps if h.status == "supported"]
    top = supported[0] if supported else None
    ties: list[str] = []
    if len(supported) > 1 and supported[0].consensus_score - supported[1].consensus_score <= tie_epsilon:
        ties = [h.hypothesis for h in supported if supported[0].consensus_score - h.consensus_score <= tie_epsilon]
        top = None
        caveats.append(f"tie among {ties}: no unique leader (|difference| <= {tie_epsilon})")
    margin = (round(supported[0].consensus_score - supported[1].consensus_score, 3) if len(supported) > 1 else None)

    dq = None if data_quality is None else _clip01(data_quality)
    low_dq = dq is not None and dq < MIN_DATA_QUALITY_FOR_ACTION
    if low_dq:
        caveats.append(f"low data quality ({dq}): ranking is weak evidence")
    if non_voting:
        caveats.append(f"agents not contributing evidence: {non_voting}")
    if not hyps:
        caveats.append("no agent supplied hypothesis-specific evidence")
    elif top is None and not ties:
        caveats.append("no hypothesis has eligible supporting evidence")
    caveats.append("voter reliability is unvalidated: the index orders hypotheses, it is NOT a probability")

    disagreement = round(1 - top.consensus_score, 3) if top else 1.0
    action_permitted = bool(top) and not low_dq and disagreement <= settings.consensus_disagreement_threshold
    if top is None:
        reason = ("; ".join(c for c in caveats if "tie" in c or "no " in c)
                  or "no unique supported hypothesis") + ": autonomous action NOT permitted."
    else:
        reason = (f"Leading hypothesis '{top.hypothesis}' (uncalibrated support index {top.consensus_score}, "
                  f"{top.independent_sources} independent source(s), margin {margin}); "
                  + ("within the disagreement threshold" if action_permitted else
                     "autonomous action NOT permitted (disagreement/data-quality gate)"))
    return ConsensusOutcome(
        hypotheses=hyps, disagreement=disagreement, top_hypothesis=top.hypothesis if top else None,
        action_permitted=action_permitted, reason=reason, margin=margin, agent_level_support=False,
        score_kind=SCORE_KIND, calibrated=False, probability=None, ties=ties, caveats=caveats,
        non_voting_agents=non_voting,
    )
