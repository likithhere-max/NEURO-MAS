"""
Repository layer: builds the in-memory PatientState (the agents' shared
blackboard) from persisted Supabase rows, and persists workflow outcomes back
to Supabase (agent event logs, audit events, actions, alerts).

Keeping this translation in one place means the agents/orchestrator never
touch the Supabase client directly -- they only ever see the Pydantic
PatientState / AgentResult contracts, which keeps the "core" package
DB-agnostic.
"""
from __future__ import annotations

from supabase import Client

from app.ml.personalized_anomaly import detect_personal_anomaly
from app.orchestrator.orchestrator import WorkflowState
from app.schemas.patient_state import (
    AdherenceState,
    BehavioralState,
    ClinicianConstraints,
    CognitiveState,
    Demographics,
    DiseaseState,
    EnvironmentalState,
    LongitudinalTrend,
    MedicationState,
    PatientPreferences,
    PatientState,
    RiskState,
    SymptomState,
)


def build_patient_state(client: Client, patient: dict) -> PatientState:
    patient_id = patient["id"]

    conditions_rows = client.table("conditions").select("*").eq("patient_id", patient_id).execute().data
    medications_rows = client.table("medications").select("*").eq("patient_id", patient_id).execute().data
    allergies_rows = client.table("allergies").select("*").eq("patient_id", patient_id).execute().data
    metrics_rows = client.table("longitudinal_metrics").select("*").eq("patient_id", patient_id).execute().data
    readings_rows = (
        client.table("metric_readings")
        .select("*")
        .eq("patient_id", patient_id)
        .order("recorded_at")
        .execute()
        .data
    )

    conditions = [c["preferred_name"] for c in conditions_rows]

    active_meds = [
        {"name": med["name"], "active": med["active"], "schedule": med["schedule"]}
        for med in medications_rows if med["active"]
    ]
    adherence_values = [med["adherence_pct_30d"] for med in medications_rows if med["adherence_pct_30d"] is not None]
    adherence_avg = round(sum(adherence_values) / len(adherence_values), 3) if adherence_values else None

    # Group individual readings by metric (already ordered oldest-first by the
    # query above), so personalized within-patient anomaly detection (Roadmap
    # item 6, Track 1) can compare the latest reading against THIS patient's
    # own prior readings for that metric.
    readings_by_metric: dict[str, list[float]] = {}
    for r in readings_rows:
        readings_by_metric.setdefault(r["metric"], []).append(r["value"])

    def _anomaly_fields(metric_name: str) -> dict:
        values = readings_by_metric.get(metric_name, [])
        if not values:
            return {}
        history, latest = values[:-1], values[-1]
        result = detect_personal_anomaly(history, latest)
        return {
            "anomaly_method": result["method"],
            "anomaly_z_score": result["z_score"],
            "is_personal_anomaly": result["is_anomaly"],
            "readings_available": result["readings_available"],
        }

    trends = [
        LongitudinalTrend(
            metric=met["metric"],
            baseline_value=met["baseline_value"],
            current_value=met["current_value"],
            window=met["window"],
            trend=(
                "unknown" if met["baseline_value"] is None or met["current_value"] is None
                else "deteriorating" if met["current_value"] > met["baseline_value"] * 1.05
                else "improving" if met["current_value"] < met["baseline_value"] * 0.95
                else "stable"
            ),
            **_anomaly_fields(met["metric"]),
        )
        for met in metrics_rows
    ]

    # Metrics that have raw readings but no longitudinal_metrics row at all
    # yet still surface their own-history anomaly signal (baseline/current
    # stay at their defaults -- there's genuinely nothing there).
    covered_metrics = {met["metric"] for met in metrics_rows}
    for metric_name in readings_by_metric:
        if metric_name in covered_metrics:
            continue
        trends.append(LongitudinalTrend(metric=metric_name, **_anomaly_fields(metric_name)))

    wearable_baselines = {}
    for met in metrics_rows:
        if met["metric"] == "tremor_amplitude":
            wearable_baselines["tremor_amplitude_baseline"] = met["baseline_value"]
        if met["metric"] == "gait_speed_m_s":
            wearable_baselines["gait_speed_baseline"] = met["baseline_value"]

    state = PatientState(
        patient_id=patient_id,
        demographics=Demographics(age=patient.get("age"), sex=patient.get("sex")),
        disease_state=DiseaseState(conditions=conditions),
        symptom_state=SymptomState(),
        medication_state=MedicationState(
            active_medications=active_meds,
            allergies=[a["substance"] for a in allergies_rows],
            adherence_pct_30d=adherence_avg,
        ),
        risk_state=RiskState(),
        behavioral_state=BehavioralState(),
        cognitive_state=CognitiveState(),
        adherence_state=AdherenceState(),
        environmental_state=EnvironmentalState(),
        longitudinal_trends=trends,
        wearable_metrics=wearable_baselines,
        patient_preferences=PatientPreferences(
            preferred_language=patient.get("preferred_language"),
            notification_opt_in=patient.get("notification_opt_in", True),
        ),
        # NOTE: `clinician_restricted_actions` existed on the old SQLite Patient
        # model but was never added as a column on the real Supabase `patients`
        # table (dropped when patients.py was migrated). Always empty until/
        # unless that column is added back -- flagged here rather than guessed.
        clinician_constraints=ClinicianConstraints(restricted_actions=[]),
    )
    state.data_completeness = state.compute_completeness()
    return state


def persist_workflow(client: Client, state: WorkflowState) -> dict:
    wf_row = {
        "id": state.workflow_id,
        "patient_id": state.patient_id,
        "trigger_reason": state.trigger_reason,
        "status": state.status,
        "is_emergency": state.is_emergency,
        "started_at": state.started_at,
        "completed_at": state.completed_at,
        "result_json": {
            "consensus": {
                "top_hypothesis": state.consensus.top_hypothesis if state.consensus else None,
                "disagreement": state.consensus.disagreement if state.consensus else None,
                "hypotheses": [h.__dict__ for h in state.consensus.hypotheses] if state.consensus else [],
                "action_permitted": state.consensus.action_permitted if state.consensus else None,
                "reason": state.consensus.reason if state.consensus else None,
            } if state.consensus else None,
        },
    }
    result = client.table("workflows").insert(wf_row).execute()
    if not result.data:
        raise RuntimeError("Failed to persist workflow")
    wf = result.data[0]

    event_log_rows = [
        {
            "workflow_id": state.workflow_id,
            "patient_id": state.patient_id,
            "agent_name": agent_name,
            "status": agent_result.status,
            "confidence": agent_result.confidence,
            "result_json": agent_result.model_dump(mode="json"),
        }
        for agent_name, agent_result in state.agent_results.items()
    ]
    if event_log_rows:
        client.table("agent_event_log").insert(event_log_rows).execute()

    audit_rows = [
        {
            "workflow_id": state.workflow_id,
            "patient_id": state.patient_id,
            "event": event["event"],
            "detail_json": event["detail"],
            "timestamp": event["timestamp"],
        }
        for event in state.audit_events
    ]
    if audit_rows:
        client.table("audit_events").insert(audit_rows).execute()

    action_rows = []
    alert_rows = []
    for action in state.proposed_actions:
        action_rows.append({
            "id": action.action_id,
            "patient_id": action.patient_id,
            "workflow_id": action.workflow_id,
            "action_type": action.action_type,
            "reason": action.reason,
            "evidence_json": action.evidence,
            "risk_level": action.risk_level,
            "autonomy_level": action.autonomy_level,
            "requires_human_approval": action.requires_human_approval,
            "approved_by": action.approved_by,
            "status": action.status,
            "safety_reasons_json": action.safety_reasons,
        })
        # Medium+ risk / human-review actions surface as Alerts for the UI.
        if action.requires_human_approval or action.risk_level in {"high", "restricted"}:
            alert_rows.append({
                "patient_id": action.patient_id,
                "workflow_id": action.workflow_id,
                "severity": "critical" if state.is_emergency and action.action_type == "emergency_escalation" else (
                    "high" if action.risk_level in {"high", "restricted"} else "medium"
                ),
                "reason": action.reason,
                "evidence_json": action.evidence,
                "status": "open",
            })
    if action_rows:
        client.table("actions").insert(action_rows).execute()
    if alert_rows:
        client.table("alerts").insert(alert_rows).execute()

    return wf