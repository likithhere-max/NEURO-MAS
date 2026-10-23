# Consensus Design (Phase 2, D)

## 1. Problem with the Phase 1 design
`compute_consensus` took each agent's *scalar confidence* as "support" for the hypotheses and reported a weighted
percentage. Only the diagnostic agent named a hypothesis, so "consensus" was a single agent's retrieval score,
displayed as a percentage, and "agreement" among agents was not measured at all. Several agents' outputs were
correlated (the knowledge agent re-retrieved from the same symptom text as the diagnostic agent), so combining them would
have double counted.

## 2. Design principles
1. **No probability without validated voters.** No agent in this system has been validated as a probabilistic predictor of any
   condition, so the consensus emits an *uncalibrated support index*: `score_kind = "uncalibrated_support_index"`,
   `calibrated = false`, `probability = null`. The UI never shows a percent sign for it (`frontend/lib/labels.ts`,
   enforced by a unit test and a backend contract test).
2. **Evidence is typed and hypothesis-specific.** An agent contributes only `HypothesisEvidence(hypothesis, evidence_type,
   direction, strength, independence_group, …)`. A scalar confidence is never reinterpreted as support.
3. **Only COMPLETED agents vote.** `abstained`, `insufficient_data`, `timed_out`, `failed`, `skipped`,
   `model_not_available` results are listed under `non_voting_agents` and their evidence is recorded in
   `excluded_evidence` with the reason.
4. **Type policy.** `knowledge_retrieval` evidence is *context only* (it is circular: it re-reads the same text). A
   `validated_model` item must be flagged validated and in-domain. Items with `strength = None` never vote.
5. **No double counting.** Items sharing an `independence_group` are one observation (the strongest is kept).
6. **Contradiction is first-class.**

## 3. Computation (`app/core/consensus.compute_evidence_consensus`)
For hypothesis *H*, over independent groups *g*:

* `support(H)       = Σ_g w_g · s_g / Σ_g w_g` — reliability-weighted mean of the strongest supporting item per group (`w_g` = max weight in the group, weights clipped to [0, 1]).
* `contradiction(H) = max_g s_g` over independent contradicting groups.
* `rank(H)          = max(0, support − contradiction)` if `H` is *supported*; else 0.
* Status: `unsupported` (no eligible support), `contradicted` (`contradiction ≥ support`), otherwise `supported`.
* Leader: highest `rank` among supported hypotheses; **ties** (difference ≤ ε = 0.02) ⇒ *no leader*.
* `disagreement = 1 − rank(leader)`; `1.0` if there is no unique supported leader.
* `action_permitted` requires a leader, data quality ≥ 0.4 **and** `disagreement ≤ CONSENSUS_DISAGREEMENT_THRESHOLD`.
* A lone independent source is reported as "not corroborated".

## 4. Properties (tested)
Tested in `tests/test_consensus_evidence.py` and, with Hypothesis, `tests/test_math_properties.py`:
scores bounded in [0,1]; order of agents irrelevant; duplicate evidence within a group changes nothing; adding
contradiction never raises a rank; evidence about another hypothesis never changes this hypothesis' score; uniform
down-scaling of reliability weights changes nothing (within the 3-decimal rounding).

## 5. Known non-properties (deliberate, pinned by a test)
* **Adding weaker independent support lowers the index** (it is a weighted mean). It orders hypotheses; it does not
  accumulate belief. A Bayesian or calibrated stacking model would behave differently but needs labelled outcomes.
* **Weights are informal engineering defaults** (`reliability_weight` per agent). They are not estimated from data.
* **Up-scaling weights beyond 1 is clipped**, so it is not scale-free outside [0, 1].
* With one voting agent the "consensus" is that agent's ranking; the output says so (`independent_sources = 1`,
  caveat "single independent source: not corroborated").

## 6. What would make it a probability
(a) a labelled cohort per condition, (b) out-of-fold predictions from each voter, (c) a stacking/calibration model fitted
on those predictions with a group-aware split, (d) calibration curves and a pre-specified acceptance criterion. None of these
exist; therefore none of this is claimed.

## 7. Orchestration interplay (D4)
The orchestrator runs agents as a DAG (`AGENT_DEPENDENCIES` in `app/agents/registry.py`). Dependencies are *ordering*
edges, not success requirements: a failed upstream agent does not prevent a downstream agent from running on whatever
is available (it is told the upstream status). The deterministic emergency pre-scan runs first and its alert is
checkpointed to storage before slower agents start. Any emergency-agent result other than `completed` fails closed
(the run is treated as an emergency pending human review).
