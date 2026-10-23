"""
Patient Data / Intake Agent (spec 6.2).
Normalizes raw, heterogeneous patient inputs (symptom reports, questionnaires,
medication logs, wearable readings, caregiver notes) into structured
observations, flags missing data, and produces a first-pass urgency
classification using the deterministic emergency keyword scan -- NOT an LLM
judgment call.

Gemini is used ONLY to narrate the reasoning_summary, and ONLY when the
deterministic scan found no emergency. When is_emergency is True, Gemini is
never called -- the original deterministic template sentence is used
directly, so the one path where speed matters most never waits on an
external API call, matching the same reasoning applied to
emergency_escalation_agent.
"""
from __future__ import annotations

import asyncio

from pydantic import BaseModel

from app.agents.base import BaseNeuroAgent
from app.ai.gemini_service import GeminiServiceError, generate_structured
from app.schemas.agent import AgentFinding, AgentResult, AgentStatus, AgentTask
from app.schemas.patient_state import PatientState
from app.services.clinical_rules import detect_emergency_detailed
import math

EXPECTED_FIELDS = ["symptoms", "medication_logs", "wearable", "vitals", "caregiver_notes"]


class _IntakeNarrative(BaseModel):
    explanation: str


async def _explain_intake_with_gemini(
    observation_count: int,
    present_count: int,
    expected_total: int,
    missing: list[str],
    urgency: str,
    data_quality: float,
) -> str | None:
    """
    Only called when is_emergency is False. Gives Gemini exactly the
    already-computed facts and nothing else, so it has nothing to fabricate.
    Returns None on any failure so the caller falls back to the template.
    """
    prompt = (
        f"A patient data intake was just normalized:\n"
        f"- {observation_count} structured observation(s) captured\n"
        f"- {present_count}/{expected_total} expected data modalities present "
        f"(missing: {', '.join(missing) if missing else 'none'})\n"
        f"- Data quality score: {data_quality}\n"
        f"- Urgency classified as: {urgency}\n\n"
        "Write a short (1-3 sentence) plain-language summary of this intake "
        "for a clinician reviewing it."
    )
    system_instruction = (
        "You summarize an already-computed, deterministic data-intake result "
        "for a clinician. Only restate the facts given above -- never invent "
        "additional missing fields, observations, or a clinical "
        "interpretation not present above. Never suggest a diagnosis or "
        "treatment. Output only the summary text."
    )
    try:
        result = await asyncio.to_thread(
            generate_structured,
            prompt,
            _IntakeNarrative,
            system_instruction=system_instruction,
        )
        return result.explanation
    except GeminiServiceError:
        return None


# Physiologic plausibility ranges (inclusive) for known numeric fields. Values outside are
# REJECTED (excluded from observations and reported), never silently passed downstream.
# These are plausibility bounds to catch unit/typing errors, NOT clinical normal ranges.
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "heart_rate_avg": (20, 250), "heart_rate": (20, 250), "spo2": (50, 100), "oxygen_saturation": (50, 100),
    "temperature_c": (30, 45), "systolic_bp": (50, 260), "diastolic_bp": (20, 160),
    "respiratory_rate": (4, 60), "tremor_amplitude": (0, 1000), "gait_speed_m_s": (0, 4),
    "steps": (0, 100000), "sleep_hours": (0, 24),
}


def _validate_numeric_block(block, label: str):
    """Return (clean_dict, invalid_list). Non-numeric, non-finite or implausible values are dropped."""
    clean, invalid = {}, []
    if not isinstance(block, dict):
        return clean, ([f"{label}:not_an_object"] if block else [])
    for k, v in block.items():
        if k == "modality":
            continue  # modality is set by the agent, never by the payload
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            if isinstance(v, (str, list, dict)) or v is None or isinstance(v, bool):
                # non-numeric payload entries (e.g. raw signal references) are kept only if the key is not a known metric
                if k in PLAUSIBLE_RANGES:
                    invalid.append(f"{label}.{k}:non_numeric")
                else:
                    clean[k] = v
                continue
        if isinstance(v, float) and not math.isfinite(v):
            invalid.append(f"{label}.{k}:non_finite")
            continue
        if k in PLAUSIBLE_RANGES and isinstance(v, (int, float)):
            lo, hi = PLAUSIBLE_RANGES[k]
            if not (lo <= v <= hi):
                invalid.append(f"{label}.{k}:out_of_plausible_range[{lo},{hi}]={v}")
                continue
        clean[k] = v
    return clean, invalid


class DataIntakeAgent(BaseNeuroAgent):
    name = "data_intake_agent"
    capabilities = ["normalization", "validation", "urgency_triage"]
    reliability_weight = 0.9  # deterministic, not model-based -> high trust

    async def execute(self, task: AgentTask, state: PatientState) -> AgentResult:
        raw = task.input_payload or {}
        observations: list[dict] = []
        missing: list[str] = []
        invalid: list[str] = []
        duplicates_removed = 0

        def _as_list(v, label):
            if v is None:
                return []
            if not isinstance(v, list):
                invalid.append(f"{label}:not_a_list")
                return []
            kept = [x for x in v if isinstance(x, dict)]
            if len(kept) != len(v):
                invalid.append(f"{label}:non_object_entries_dropped={len(v) - len(kept)}")
            return kept

        symptoms = _as_list(raw.get("symptoms"), "symptoms")
        seen_symptoms = set()
        deduped_symptoms = []
        for s_ in symptoms:
            key = (str(s_.get("name")), str(s_.get("free_text", "")), str(s_.get("severity")), str(s_.get("onset")))
            if key in seen_symptoms:
                duplicates_removed += 1
                continue
            seen_symptoms.add(key)
            deduped_symptoms.append(s_)
        symptoms = deduped_symptoms
        for s_ in symptoms:
            observations.append({
                "modality": "symptom",
                "name": s_.get("name"),
                "severity": s_.get("severity"),
                "onset": s_.get("onset"),
                "free_text": s_.get("free_text", "") or "",
            })
        if not symptoms:
            missing.append("symptoms")

        med_logs = _as_list(raw.get("medication_logs"), "medication_logs")
        for m in med_logs:
            observations.append({
                "modality": "medication_log",
                "name": m.get("name"),
                "taken": m.get("taken"),
                "scheduled_time": m.get("scheduled_time"),
            })
        if not med_logs:
            missing.append("medication_logs")

        wearable, bad = _validate_numeric_block(raw.get("wearable"), "wearable")
        invalid += bad
        if wearable:
            observations.append({**wearable, "modality": "wearable"})  # modality can never be overridden by the payload
        else:
            missing.append("wearable")

        vitals, bad = _validate_numeric_block(raw.get("vitals"), "vitals")
        invalid += bad
        if vitals:
            observations.append({**vitals, "modality": "vital"})
        else:
            missing.append("vitals")

        caregiver_notes = raw.get("caregiver_notes")
        if isinstance(caregiver_notes, str) and caregiver_notes.strip():
            observations.append({"modality": "caregiver_report", "note": caregiver_notes})
        else:
            caregiver_notes = None
            missing.append("caregiver_notes")

        text_blob = []
        for s_ in symptoms:  # scan EVERY text field (previously free_text hid an alarming name)
            text_blob += [str(s_.get(f)) for f in ("name", "free_text") if s_.get(f)]
        if caregiver_notes:
            text_blob.append(caregiver_notes)
        _em = detect_emergency_detailed(text_blob)
        is_emergency, matched = _em["is_emergency"], _em["matched"]

        # `completeness` = share of expected modalities present. Historically exposed as
        # "data_quality"; the key is kept for compatibility but it is COMPLETENESS, not quality.
        completeness = round(1 - (len(missing) / len(EXPECTED_FIELDS)), 2)
        n_supplied_values = len(wearable) + len(vitals) + len(invalid)
        valid_fraction = (1 - len(invalid) / n_supplied_values) if n_supplied_values else 1.0
        data_quality = round(completeness * valid_fraction, 2)

        if is_emergency:
            urgency = "critical"
        elif any(str(s.get("severity") or "").lower() in {"severe", "high"} for s in symptoms):
            urgency = "high"
        elif symptoms:
            urgency = "medium"
        else:
            urgency = "low"

        findings = [
            AgentFinding(label="observations_captured", detail=f"{len(observations)} structured observations", value=len(observations)),
            AgentFinding(label="urgency", detail=urgency, severity=urgency if urgency != "low" else "info"),
        ]
        if matched:
            findings.append(AgentFinding(
                label="emergency_keyword_match",
                detail=f"Deterministic keyword scan matched: {', '.join(matched)}",
                severity="critical",
            ))

        present_count = len(EXPECTED_FIELDS) - len(missing)
        template_summary = (
            f"Normalized {len(observations)} observations across "
            f"{present_count}/{len(EXPECTED_FIELDS)} expected modalities. "
            f"Urgency classified as '{urgency}' via deterministic rules."
        )

        warnings = [f"missing_data:{m}" for m in missing] + [f"invalid_input:{i}" for i in invalid]
        if duplicates_removed:
            warnings.append(f"duplicate_symptoms_removed:{duplicates_removed}")

        if is_emergency:
            # Zero API dependency on the time-critical path -- same reasoning
            # as emergency_escalation_agent.
            reasoning_summary = template_summary
            narrative_source = "not_applicable_emergency_bypass"
        else:
            narrative = await _explain_intake_with_gemini(
                observation_count=len(observations),
                present_count=present_count,
                expected_total=len(EXPECTED_FIELDS),
                missing=missing,
                urgency=urgency,
                data_quality=data_quality,
            )
            if narrative:
                reasoning_summary = narrative
                narrative_source = "gemini"
            else:
                reasoning_summary = template_summary
                narrative_source = "template_fallback"
                warnings.append("gemini_narrative_unavailable_used_template_summary")

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            status=AgentStatus.COMPLETED,
            findings=findings,
            confidence=round(valid_fraction, 2),  # share of supplied values that passed validation (NOT a probability)
            warnings=warnings,
            recommended_next_agents=["neuro_diagnostic_agent", "risk_stratification_agent"],
            requires_human_review=is_emergency,
            reasoning_summary=reasoning_summary,
            raw_payload={
                "observations": observations,
                "data_quality": data_quality,
                "completeness": completeness,
                "invalid_inputs": invalid,
                "duplicates_removed": duplicates_removed,
                "emergency_matches": matched,
                "missing_data": missing,
                "urgency": urgency,
                "narrative_source": narrative_source,
            },
        )