"""
Longitudinal / Biomarker Agent (spec 6.5).

Answers "how is this patient changing over time?" using two independent
deterministic signals:

  1. Baseline-vs-current percentage-change heuristic (unchanged from before --
     explicitly not a validated statistical change-point model), computed
     from `longitudinal_metrics` (one manually-set baseline + current value
     per metric).
  2. Personalized within-patient anomaly detection (Roadmap item 6, Track 1):
     real z-score statistics over THIS patient's own historical
     `metric_readings`, computed in repository.build_patient_state() via
     app/ml/personalized_anomaly.py and attached to each LongitudinalTrend.
     Never a population model -- only ever compares a patient's own history
     against their own history.

Gemini narrates ONLY the already-computed signals -- it never reclassifies a
metric, changes a delta, or invents a z-score. Skipped entirely when there's
nothing to explain (no trend history at all).
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
    Gives Gemini ONLY the already-computed classification and/or personal-
    anomaly result per metric -- nothing else -- so it has nothing to
    reclassify or fabricate. Returns None on any failure so the caller falls
    back to the template.
    """
    lines = []
    for t in trend_summaries:
        if t["classification"] is not None:
            lines.append(
                f"- {t['metric']}: {t['baseline_value']} -> {t['current_value']} "
                f"({t['delta_pct']}% change over {t['window']}), classified as {t['classification']}"
            )
        if t["is_personal_anomaly"]:
            lines.append(
                f"- {t['metric']}: latest reading is a statistical outlier compared to "
                f"this patient's own historical readings (z-score {t['anomaly_z_score']}, "
                f"based on {t['readings_available']} prior reading(s) from this same patient)"
            )
    prompt = (
        "Here is an already-computed longitudinal trend/anomaly comparison for a patient:\n\n"
        + "\n".join(lines) + "\n\n"
        "Write a short (2-4 sentence) plain-language summary of these trends for "
        "a clinician reviewing this patient's progress over time."
    )
    system_instruction = (
        "You summarize already-computed longitudinal trend data for a clinical "
        "documentation assistant. Only describe what is explicitly listed above "
        "-- never invent a metric, a value, a classification, or a z-score not "
        "given to you. Do not make or imply a diagnosis. Output only the summary text."
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
    capabilities = ["trend_detection", "baseline_comparison", "personal_anomaly_detection"]
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
        personal_anomalies = []
        trend_summaries = []
        for t in trends:
            classification = None
            delta_pct = None

            if t.baseline_value is not None and t.current_value is not None:
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

            if t.is_personal_anomaly:
                personal_anomalies.append(t.metric)
                findings.append(AgentFinding(
                    label=f"personal_anomaly:{t.metric}",
                    detail=(
                        f"Latest reading for {t.metric} is a statistical outlier vs this "
                        f"patient's own history (z-score {t.anomaly_z_score}, based on "
                        f"{t.readings_available} prior reading(s))"
                    ),
                    severity="high",
                    value=t.anomaly_z_score,
                ))

            if classification is not None or t.is_personal_anomaly:
                trend_summaries.append({
                    "metric": t.metric,
                    "baseline_value": t.baseline_value,
                    "current_value": t.current_value,
                    "delta_pct": round(delta_pct * 100, 1) if delta_pct is not None else None,
                    "window": t.window,
                    "classification": classification,
                    "is_personal_anomaly": t.is_personal_anomaly,
                    "anomaly_z_score": t.anomaly_z_score,
                    "readings_available": t.readings_available,
                })

        confidence = 0.7 if findings else 0.0
        template_summary = (
            f"Compared {len(trends)} tracked metric(s) to patient baseline. "
            f"Deteriorating: {deteriorating or 'none'}. Improving: {improving or 'none'}. "
            f"Stable: {stable or 'none'}."
            + (f" Personal statistical anomalies detected in: {personal_anomalies}." if personal_anomalies else "")
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
            requires_human_review=bool(deteriorating) or bool(personal_anomalies),
            reasoning_summary=reasoning_summary,
            raw_payload={
                "deteriorating_metrics": deteriorating,
                "improving_metrics": improving,
                "stable_metrics": stable,
                "personal_anomalies": personal_anomalies,
                "narrative_source": narrative_source,
            },
        )