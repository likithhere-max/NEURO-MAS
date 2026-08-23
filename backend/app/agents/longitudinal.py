"""
Longitudinal / Biomarker Agent (spec 6.5).

Answers "how is this patient changing over time?" by comparing the
patient's existing `longitudinal_trends` (persisted state, populated from
prior workflow runs) against newly observed values. Deviation detection uses
a simple percentage-change heuristic -- explicitly not a validated
statistical change-point model.
"""
from __future__ import annotations

from app.agents.base import BaseNeuroAgent
from app.schemas.agent import AgentFinding, AgentResult, AgentStatus, AgentTask
from app.schemas.patient_state import PatientState

DETERIORATION_THRESHOLD = 0.20  # 20% adverse change vs baseline triggers a flag


class LongitudinalAgent(BaseNeuroAgent):
    name = "longitudinal_agent"
    capabilities = ["trend_detection", "baseline_comparison"]
    reliability_weight = 0.75

    async def execute(self, task: AgentTask, state: PatientState) -> AgentResult:
        trends = state.longitudinal_trends or []
        if not trends:
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                status=AgentStatus.COMPLETED,
                confidence=0.0,
                warnings=["no_baseline_history_available"],
                reasoning_summary="No longitudinal history exists yet for this patient; "
                                   "this workflow run establishes the first data point.",
                raw_payload={"trend_summary": []},
            )

        findings = []
        deteriorating, improving, stable = [], [], []
        for t in trends:
            if t.baseline_value is None or t.current_value is None:
                continue
            delta_pct = (t.current_value - t.baseline_value) / (abs(t.baseline_value) or 1)
            if abs(delta_pct) < 0.05:
                stable.append(t.metric)
                sev = "info"
            elif delta_pct >= DETERIORATION_THRESHOLD:
                deteriorating.append(t.metric)
                sev = "high"
            elif delta_pct <= -DETERIORATION_THRESHOLD:
                improving.append(t.metric)
                sev = "info"
            else:
                stable.append(t.metric)
                sev = "info"

            findings.append(AgentFinding(
                label=f"trend:{t.metric}",
                detail=f"{t.baseline_value} -> {t.current_value} ({round(delta_pct*100,1)}%) over {t.window}",
                severity=sev,
                value=round(delta_pct, 3),
            ))

        confidence = 0.7 if findings else 0.0

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            status=AgentStatus.COMPLETED,
            findings=findings,
            confidence=confidence,
            warnings=[],
            recommended_next_agents=["risk_stratification_agent", "personalization_agent"],
            requires_human_review=bool(deteriorating),
            reasoning_summary=(
                f"Compared {len(trends)} tracked metric(s) to patient baseline. "
                f"Deteriorating: {deteriorating or 'none'}. Improving: {improving or 'none'}. "
                f"Stable: {stable or 'none'}."
            ),
            raw_payload={
                "deteriorating_metrics": deteriorating,
                "improving_metrics": improving,
                "stable_metrics": stable,
            },
        )
