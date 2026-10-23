"""
Signal Analysis Agent (spec 6.4 / section 15).

Handles neurological signal modalities (tremor accelerometer summaries, gait
metrics, EEG/EMG metadata). Per the explicit "do not fake features" rule:
if a real analysis model isn't wired up for a modality, this agent returns
MODEL_NOT_AVAILABLE for that modality rather than inventing a prediction.
What it *can* do without a trained model: simple, clearly-labeled threshold
checks on pre-computed summary metrics (e.g. "tremor_amplitude_hz" already
extracted upstream by the device/SDK), since that's arithmetic, not inference.
"""
from __future__ import annotations

import asyncio
import math

from pydantic import BaseModel

from app.agents.base import BaseNeuroAgent
from app.ai.gemini_service import GeminiServiceError, generate_structured
from app.schemas.agent import AgentFinding, AgentResult, AgentStatus, AgentTask
from app.schemas.patient_state import PatientState

# Modalities with a real (even if simple) analyzer implemented in this prototype.
# Only metrics with an actual analyzer below. (Audit fix: "steps" and "heart_rate_avg" were listed
# here although nothing analysed them, so the summary claimed they were "evaluated".)
SUPPORTED_SUMMARY_METRICS = {"tremor_amplitude", "gait_speed_m_s"}
RECORDED_ONLY_METRICS = {"steps", "heart_rate_avg"}  # stored/validated upstream, not analysed here

# Modalities explicitly requested by the spec that require a trained model
# NEURO-MAS does not ship (clinical-grade EEG anomaly detection, video-derived
# gait classification, voice biomarkers, etc.)
UNSUPPORTED_MODALITIES = {"eeg_raw", "emg_raw", "video_gait", "voice"}


class _SignalNarrative(BaseModel):
    explanation: str


async def _explain_signal_findings_with_gemini(
    finding_summaries: list[dict],
    model_not_available: list[str],
) -> str | None:
    """Ask Gemini to narrate already-computed threshold findings.

    Gemini receives ONLY the finding labels/details/severities already computed
    deterministically -- it does not add a finding, change a severity, or
    invent a value.
    """
    prompt = (
        "A wearable-signal analysis agent computed the following findings via "
        "simple deterministic threshold checks against the patient's own stored "
        "baseline (not a trained ML model):\n\n"
        f"Findings: {finding_summaries}\n"
        f"Modalities with no available model (raw data stored, no prediction made): "
        f"{model_not_available if model_not_available else 'none'}\n\n"
        "Write a short, plain-language paragraph explaining these findings to a "
        "clinician, referencing only the findings and unavailable-modality list above."
    )
    system_instruction = (
        "You are writing a short explanatory note for a clinician reviewing "
        "wearable-signal threshold findings. You may ONLY restate and explain the "
        "findings and modality list given to you. Do NOT invent new findings, "
        "severities, or values. Do NOT diagnose a condition or suggest a "
        "clinical action. Keep it to 2-4 sentences."
    )
    try:
        result = await asyncio.to_thread(
            generate_structured,
            prompt,
            _SignalNarrative,
            system_instruction=system_instruction,
        )
        return result.explanation
    except GeminiServiceError:
        return None


class SignalAnalysisAgent(BaseNeuroAgent):
    name = "signal_analysis_agent"
    capabilities = ["tremor_threshold_check", "gait_summary_check"]
    reliability_weight = 0.6

    async def execute(self, task: AgentTask, state: PatientState) -> AgentResult:
        payload = task.input_payload or {}
        wearable = payload.get("wearable", {})

        if not wearable:
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                status=AgentStatus.SKIPPED,
                confidence=0.0,
                warnings=["no_signal_data_provided"],
                reasoning_summary="No wearable/signal data was provided for this event; agent skipped.",
                raw_payload={},
            )

        findings: list[AgentFinding] = []
        warnings: list[str] = []
        model_not_available: list[str] = []

        for key in wearable:
            if key in UNSUPPORTED_MODALITIES:
                model_not_available.append(key)

        def _num(v):
            return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None

        compared = 0
        evaluated: list[str] = []
        tremor_raw = wearable.get("tremor_amplitude")
        if tremor_raw is not None:
            tremor = _num(tremor_raw)
            baseline = _num((state.wearable_metrics or {}).get("tremor_amplitude_baseline"))
            if tremor is None:
                warnings.append("invalid_value:tremor_amplitude")
            else:
                evaluated.append("tremor_amplitude")
                if baseline is not None and baseline > 0:
                    compared += 1
                    if tremor > baseline * 1.5:
                        findings.append(AgentFinding(
                            label="tremor_increase",
                            detail=f"Tremor amplitude {tremor} vs baseline {baseline} (+{round((tremor/baseline-1)*100,1)}%)",
                            severity="high", value=tremor,
                        ))
                    else:
                        findings.append(AgentFinding(label="tremor_amplitude", detail=f"{tremor} (baseline {baseline})", severity="info", value=tremor))
                else:
                    warnings.append("tremor_baseline_missing_or_nonpositive_no_relative_comparison")
                    findings.append(AgentFinding(label="tremor_amplitude", detail=f"{tremor} (no usable baseline)", severity="info", value=tremor))

        gait_raw = wearable.get("gait_speed_m_s")
        if gait_raw is not None:
            gait = _num(gait_raw)
            baseline_gait = _num((state.wearable_metrics or {}).get("gait_speed_baseline"))
            if gait is None:
                warnings.append("invalid_value:gait_speed_m_s")
            else:
                evaluated.append("gait_speed_m_s")
                if baseline_gait is not None and baseline_gait > 0:
                    compared += 1
                    if gait < baseline_gait * 0.8:
                        findings.append(AgentFinding(
                            label="gait_slowing",
                            detail=f"Gait speed {gait} m/s vs baseline {baseline_gait} m/s (-{round((1-gait/baseline_gait)*100,1)}%)",
                            severity="medium", value=gait,
                        ))
                    else:
                        findings.append(AgentFinding(label="gait_speed", detail=f"{gait} m/s", severity="info", value=gait))
                else:
                    warnings.append("gait_baseline_missing_or_nonpositive_no_relative_comparison")
                    findings.append(AgentFinding(label="gait_speed", detail=f"{gait} m/s (no usable baseline)", severity="info", value=gait))

        recorded_only = [k for k in wearable if k in RECORDED_ONLY_METRICS]
        if recorded_only:
            warnings.append("recorded_but_not_analysed:" + ",".join(recorded_only))

        if model_not_available:
            warnings.append(
                "MODEL_NOT_AVAILABLE for modalities: " + ", ".join(model_not_available) +
                " -- no trained clinical-grade model is wired up in this prototype; "
                "raw data is stored for future analysis but no prediction is produced."
            )

        # Not a probability: share of evaluated metrics that had a usable baseline to compare against.
        confidence = round(compared / len(evaluated), 2) if evaluated else 0.0
        status = AgentStatus.COMPLETED if findings or model_not_available else AgentStatus.SKIPPED

        template_summary = (
            f"Evaluated {len(evaluated)} supported summary metric(s) ({', '.join(evaluated) or 'none'}); "
            f"{compared} had a usable baseline for threshold comparison (unvalidated 1.5x / 0.8x thresholds). "
            f"{len(model_not_available)} modality/modalities had no available model."
        )

        if status == AgentStatus.COMPLETED:
            finding_summaries = [
                {"label": f.label, "detail": f.detail, "severity": f.severity}
                for f in findings
            ]
            narrative = await _explain_signal_findings_with_gemini(finding_summaries, model_not_available)
            if narrative:
                reasoning_summary = narrative
                narrative_source = "gemini"
            else:
                reasoning_summary = template_summary
                narrative_source = "template_fallback"
                warnings.append("gemini_narrative_unavailable_used_template_summary")
        else:
            reasoning_summary = template_summary
            narrative_source = "not_applicable_no_findings"

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            status=status,
            findings=findings,
            confidence=confidence,
            warnings=warnings + ["consumer_wearable_signal_quality_is_NOT_equivalent_to_clinical_EEG_EMG"],
            recommended_next_agents=["longitudinal_agent", "risk_stratification_agent"],
            requires_human_review=any(f.severity in {"high", "critical"} for f in findings),
            reasoning_summary=reasoning_summary,
            raw_payload={"model_not_available": model_not_available, "narrative_source": narrative_source,
                         "evaluated_metrics": evaluated, "recorded_only_metrics": recorded_only, "metrics_with_baseline": compared},
        )