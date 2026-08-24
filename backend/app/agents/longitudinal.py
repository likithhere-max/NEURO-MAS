"""
Longitudinal / Biomarker Agent (spec 6.5).
Answers "how is this patient changing over time?" by comparing the
patient's existing `longitudinal_trends` (persisted state, populated from
prior workflow runs) against newly observed values. Deviation detection uses
a simple percentage-change heuristic -- explicitly not a validated
statistical change-point model.

Gemini narrates ONLY the already-computed classification (which metrics are
deteriorating/improving/stable, and by how much) -- it never reclassifies a
metric or changes a delta. Skipped entirely when there's nothing to explain
(no trend history, or no trend with both a baseline and current value).
"""
from __future__ import annotations

import asyncio

from pydantic import BaseModel

from app.agents.base import BaseNeuroAgent
from app.ai.gemini_service import GeminiServiceError, generate_structured
from app.schemas.agent import AgentFinding, AgentResult, AgentStatus, AgentTask
from app.schemas.patient_state import PatientState

DETERIORATION_THRESHOLD = 0.20  # 20% adverse change vs baseline triggers a flag


class _TrendNarrative(BaseModel):
    explanation: str


async def _explain_trends_with_gemini(trend_summaries: list[dict]) -> str | None:
    """
    Gives Gemini ONLY the already-computed classification per metric
    (metric name, baseline, current, delta %, window, deteriorating/
    improving/stable bucket) -- nothing else -- so it has nothing to
    reclassify or fabricate. Returns None on any failure so the caller
    falls back to the template.
    """
    lines = [
        f"- {t['metric']}: {t['baseline_value']} -> {t['current_value']} "
        f"({t['delta_pct']}% change over {t['window']}), classified as {t['classification']}"
        for t in trend_summaries
    ]
    prompt = (
        "Here is an already-computed longitudinal trend comparison for a patient:\n\n"
        + "\n".join(lines) + "\n\n"
        "Write a short (2-4 sentence) plain-language summary of these trends for "
        "a clinician reviewing this patient's progress over time."
    )
    system_instruction = (
        "You summarize already-computed longitudinal trend data for a clinical "
        "documentation assistant. Only describe what is explicitly listed above "
        "-- never invent a metric, a value, or a classification not given to "
        "you. Do not make or imply a diagnosis. Output only the summary text."
    )
    try:
        result = await asyncio.to_thread(
            generate_structured,
            prompt,
            _TrendNarrative,
            system_instruction=system_instruction,
        )
        return result.explanation
    except GeminiServiceError:
        return None


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
        trend_summaries = []
        for t in trends:
            if t.baseline_value is None or t.current_value is None:
                continue
            delta_pct = (t.current_value - t.baseline_value) / (abs(t.baseline_value) or 1)
            if abs(delta_pct) < 0.05:
                stable.append(t.metric)
                sev = "info"
                classification = "stable"
            elif delta_pct >= DETERIORATION_THRESHOLD:
                deteriorating.append(t.metric)
                sev = "high"
                classification = "deteriorating"
            elif delta_pct <= -DETERIORATION_THRESHOLD:
                improving.append(t.metric)
                sev = "info"
                classification = "improving"
            else:
                stable.append(t.metric)
                sev = "info"
                classification = "stable"
            findings.append(AgentFinding(
                label=f"trend:{t.metric}",
                detail=f"{t.baseline_value} -> {t.current_value} ({round(delta_pct*100,1)}%) over {t.window}",
                severity=sev,
                value=round(delta_pct, 3),
            ))
            trend_summaries.append({
                "metric": t.metric,
                "baseline_value": t.baseline_value,
                "current_value": t.current_value,
                "delta_pct": round(delta_pct * 100, 1),
                "window": t.window,
                "classification": classification,
            })

        confidence = 0.7 if findings else 0.0
        template_summary = (
            f"Compared {len(trends)} tracked metric(s) to patient baseline. "
            f"Deteriorating: {deteriorating or 'none'}. Improving: {improving or 'none'}. "
            f"Stable: {stable or 'none'}."
        )

        warnings = []
        if findings:
            narrative = await _explain_trends_with_gemini(trend_summaries)
            if narrative:
                reasoning_summary = narrative
                narrative_source = "gemini"
            else:
                reasoning_summary = template_summary
                narrative_source = "template_fallback"
                warnings.append("gemini_narrative_unavailable_used_template_summary")
        else:
            reasoning_summary = template_summary
            narrative_source = "not_applicable_no_valid_trends"

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            status=AgentStatus.COMPLETED,
            findings=findings,
            confidence=confidence,
            warnings=warnings,
            recommended_next_agents=["risk_stratification_agent", "personalization_agent"],
            requires_human_review=bool(deteriorating),
            reasoning_summary=reasoning_summary,
            raw_payload={
                "deteriorating_metrics": deteriorating,
                "improving_metrics": improving,
                "stable_metrics": stable,
                "narrative_source": narrative_source,
            },
        )