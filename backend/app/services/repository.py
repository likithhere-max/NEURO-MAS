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

from app.core.metric_semantics import classify_change
from app.ml.longitudinal import analyze_series
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

    # Group individual readings by metric as (timestamp, value) pairs (already ordered oldest-first by the
    # query above). Longitudinal analysis (app/ml/longitudinal.py) uses the real timestamps; it only ever
    # compares a patient with their OWN history, never with a population.
    readings_by_metric: dict[str, tuple[list, list]] = {}
    for r in readings_rows:
        times, vals = readings_by_metric.setdefault(r["metric"], ([], []))
        times.append(r.get("recorded_at"))
        vals.append(r["value"])

    def _anomaly_fields(metric_name: str) -> dict:
        times, values = readings_by_metric.get(metric_name, ([], []))
        if not values:
            return {}
        report = analyze_series(metric_name, times, values)
        point = report.statistics.get("point_t_interval", {}) if report.statistics else {}
        if report.detection_status in ("insufficient_data", "invalid_input"):
            method = "insufficient_history" if report.detection_status == "insufficient_data" else "invalid_input"
        else:
            method = report.method_version
        return {
            "anomaly_method": method,
            "anomaly_z_score": point.get("z_score"),
            "is_personal_anomaly": report.any_signal,
            "readings_available": report.observation_count,
            "anomaly_report": report.as_dict(),
        }

    trends = [
        LongitudinalTrend(
            metric=met["metric"],
            baseline_value=met["baseline_value"],
            current_value=met["current_value"],
            window=met["window"],
            trend=classify_change(met["metric"], met["baseline_value"], met["current_value"])["classification"],
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
    state.external_ai_consent = bool(patient.get("external_ai_consent", False))
    state.redaction_identifiers = [n for n in [patient.get("full_name")] if n]
    state.data_completeness = state.compute_completeness()
    return state


def consensus_to_dict(c) -> dict | None:
    if c is None:
        return None
    return {
        "top_hypothesis": c.top_hypothesis, "disagreement": c.disagreement,
        "hypotheses": [h.__dict__ for h in c.hypotheses], "margin": c.margin,
        "action_permitted": c.action_permitted, "reason": c.reason,
        # honest labelling of what the numbers are (docs/phase2/CONSENSUS_DESIGN.md)
        "score_kind": c.score_kind, "calibrated": c.calibrated, "probability": c.probability,
        "ties": c.ties, "caveats": c.caveats, "non_voting_agents": c.non_voting_agents,
    }


def build_persist_payload(state: WorkflowState) -> dict:
    """Everything that must be stored for one workflow, as plain JSON-able rows. Pure function (unit-testable
    without a database); consumed by both the RPC path and the sequential fallback."""
    wf_row = {
        "id": state.workflow_id,
        "patient_id": state.patient_id,
        "trigger_reason": state.trigger_reason,
        "status": state.status,
        "is_emergency": state.is_emergency,
        "started_at": state.started_at,
        "completed_at": state.completed_at,
        "result_json": {
            "consensus": consensus_to_dict(state.consensus),
            "degraded": getattr(state, "degraded", False),
            "failed_agents": getattr(state, "failed_agents", []),
            "timed_out_agents": getattr(state, "timed_out_agents", []),
            "deadline_exceeded": getattr(state, "deadline_exceeded", False),
            "emergency_screen": getattr(state, "emergency_screen", {}),
        },
    }
    events = [
        {"workflow_id": state.workflow_id, "patient_id": state.patient_id, "agent_name": n,
         "status": r.status, "confidence": r.confidence, "result_json": r.model_dump(mode="json")}
        for n, r in state.agent_results.items()
    ]
    audits = [
        {"workflow_id": state.workflow_id, "patient_id": state.patient_id, "event": e["event"],
         "detail_json": e["detail"], "timestamp": e["timestamp"]}
        for e in state.audit_events
    ]
    actions, alerts = [], []
    for a in state.proposed_actions:
        actions.append({
            "id": a.action_id, "patient_id": a.patient_id, "workflow_id": a.workflow_id, "action_type": a.action_type,
            "reason": a.reason, "evidence_json": a.evidence, "risk_level": a.risk_level,
            "autonomy_level": a.autonomy_level, "requires_human_approval": a.requires_human_approval,
            "approved_by": a.approved_by, "status": a.status, "safety_reasons_json": a.safety_reasons,
        })
        # Medium+ risk / human-review actions surface as Alerts for the UI.
        if a.requires_human_approval or a.risk_level in {"high", "restricted"}:
            alerts.append({
                "patient_id": a.patient_id, "workflow_id": a.workflow_id,
                "severity": "critical" if state.is_emergency and a.action_type == "emergency_escalation" else (
                    "high" if a.risk_level in {"high", "restricted"} else "medium"),
                "reason": a.reason, "evidence_json": a.evidence, "status": "open",
            })
    return {"workflow": wf_row, "agent_events": events, "audit_events": audits, "actions": actions, "alerts": alerts}


def persist_workflow(client: Client, state: WorkflowState) -> dict:
    """
    Store a finished workflow. Two paths:

      * settings.persist_via_rpc  -> one transaction (RPC `persist_workflow_atomic`; all-or-nothing, idempotent on
        the workflow id via ON CONFLICT DO NOTHING).
      * default (sequential)      -> the workflow row is written FIRST with result_json.persist_state="persisting",
        children next, and the marker is flipped to "complete" LAST. A crash in between therefore leaves a row that
        says it is incomplete rather than a silently partial "completed" run, and a retry with the same workflow id
        (idempotency key) detects it and rewrites the children instead of duplicating them.
    """
    from app.core.config import get_settings

    payload = build_persist_payload(state)
    wf_row = payload["workflow"]

    if get_settings().persist_via_rpc:
        res = client.rpc("persist_workflow_atomic", {"p_payload": payload}).execute()
        data = res.data
        if isinstance(data, list):
            data = data[0] if data else None
        if not data:
            raise RuntimeError("persist_workflow_atomic returned no result")
        return data if isinstance(data, dict) else {"id": state.workflow_id}

    existing = client.table("workflows").select("*").eq("id", state.workflow_id).execute().data
    if existing:
        if (existing[0].get("result_json") or {}).get("persist_state", "complete") == "complete":
            return existing[0]                       # idempotent replay: already fully stored
        for table in ("agent_event_log", "audit_events", "actions"):   # incomplete earlier attempt: rewrite children
            client.table(table).delete().eq("workflow_id", state.workflow_id).execute()
        wf = existing[0]
    else:
        pending = {**wf_row, "result_json": {**wf_row["result_json"], "persist_state": "persisting"}}
        result = client.table("workflows").insert(pending).execute()
        if not result.data:
            raise RuntimeError("Failed to persist workflow")
        wf = result.data[0]

    try:
        for table, rows in (("agent_event_log", payload["agent_events"]), ("audit_events", payload["audit_events"]),
                            ("actions", payload["actions"]), ("alerts", payload["alerts"])):
            if rows:
                client.table(table).insert(rows).execute()
        final = {**wf_row["result_json"], "persist_state": "complete"}
        client.table("workflows").update({"result_json": final}).eq("id", state.workflow_id).execute()
    except Exception:
        try:   # best-effort: record that the stored run is incomplete (never raises over the original error)
            client.table("workflows").update(
                {"result_json": {**wf_row["result_json"], "persist_state": "incomplete"}}).eq("id", state.workflow_id).execute()
        except Exception:
            pass
        raise
    return wf


def record_emergency_checkpoint(client: Client, state: WorkflowState) -> bool:
    """Durable early record of a pre-scan emergency, written BEFORE the slow agents run so that a later crash,
    timeout, client disconnect or persistence failure cannot lose it. Idempotent per workflow id."""
    if not state.is_emergency:
        return False
    marker = "EMERGENCY PRE-SCAN"
    existing = client.table("alerts").select("id,reason").eq("workflow_id", state.workflow_id).execute().data or []
    if any(str(a.get("reason", "")).startswith(marker) for a in existing):
        return True
    client.table("alerts").insert({
        "patient_id": state.patient_id, "workflow_id": state.workflow_id, "severity": "critical",
        "reason": f"{marker}: possible emergency detected by the deterministic screen (workflow {state.workflow_id}); "
                  "full assessment still running -- clinician review required.",
        "evidence_json": state.emergency_screen.get("matched", []), "status": "open",
    }).execute()
    return True


def build_case_context(client: Client, patient_id: str, patient: dict | None = None) -> dict:
    """
    Compact, structured summary of a patient's most recent assessment --
    NOT a raw dump of workflow/agent-event rows. Built for two consumers:
    the AI Assistant chat (app/services/chat_assistant.py) and, later, any
    "explain this result" UI panel. Kept in one place (repository.py, next
    to build_patient_state) so both stay in sync with the real DB shape
    instead of each re-deriving it independently.

    Returns {"has_assessment": False, ...} rather than raising when the
    patient has no workflow yet -- an empty case is a normal state, not an
    error.
    """
    if patient is None:
        patient_result = client.table("patients").select("*").eq("id", patient_id).execute()
        patient = patient_result.data[0] if patient_result.data else {}

    context: dict = {
        "patient": {
            "full_name": patient.get("full_name"),
            "age": patient.get("age"),
            "sex": patient.get("sex"),
        },
        "has_assessment": False,
    }

    conditions_rows = client.table("conditions").select("preferred_name").eq("patient_id", patient_id).execute().data
    context["patient"]["known_conditions"] = [c["preferred_name"] for c in conditions_rows]

    wf_result = (
        client.table("workflows")
        .select("*")
        .eq("patient_id", patient_id)
        .order("started_at", desc=True)
        .limit(1)
        .execute()
    )
    if not wf_result.data:
        return context
    wf = wf_result.data[0]

    events = (
        client.table("agent_event_log")
        .select("*")
        .eq("workflow_id", wf["id"])
        .execute()
        .data
    )
    by_agent = {e["agent_name"]: e["result_json"] for e in events}

    assessment_result = (
        client.table("symptom_assessments")
        .select("*")
        .eq("workflow_id", wf["id"])
        .limit(1)
        .execute()
    )
    assessment = assessment_result.data[0] if assessment_result.data else None

    diagnostic = by_agent.get("neuro_diagnostic_agent", {}).get("raw_payload", {})
    risk = by_agent.get("risk_stratification_agent", {}).get("raw_payload", {})
    knowledge = by_agent.get("knowledge_agent", {}).get("raw_payload", {})
    personalization = by_agent.get("personalization_agent", {}).get("raw_payload", {})
    data_intake = by_agent.get("data_intake_agent", {}).get("raw_payload", {})

    extracted = (assessment or {}).get("extracted_json") or {}

    context.update({
        "has_assessment": True,
        "workflow_id": wf["id"],
        "assessed_at": wf["started_at"],
        "trigger_reason": wf["trigger_reason"],
        "is_emergency": wf["is_emergency"],
        "reported_symptoms": [s.get("name") for s in extracted.get("symptoms", [])] if extracted else None,
        "associated_features": extracted.get("associated_features") if extracted else None,
        "symptom_duration": extracted.get("duration") if extracted else None,
        "symptom_progression": extracted.get("progression") if extracted else None,
        "diagnostic_candidates": [
            {
                "name": c.get("preferred_name"),
                "confidence": c.get("confidence"),
                "matched_terms": c.get("matched_terms"),
            }
            for c in diagnostic.get("candidate_conditions", [])[:5]
        ],
        "diagnostic_red_flags": diagnostic.get("red_flags", []),
        "diagnostic_missing_information": diagnostic.get("missing_information", []),
        "risk_scores": risk.get("heuristic_scores", {}),
        "risk_overall_bucket": risk.get("overall_bucket"),
        "evidence_citations": [
            {"condition": c.get("disease"), "source_url": c.get("source_url")}
            for c in knowledge.get("citations", [])
        ],
        "monitoring_plan": personalization.get("monitoring_plan", []),
        "personalization_reasons": personalization.get("personalization_reasons", []),
        "urgency": data_intake.get("urgency"),
        "data_quality": data_intake.get("data_quality"),
        "reasoning_summaries": {
            agent_name: result.get("reasoning_summary")
            for agent_name, result in by_agent.items()
            if result.get("reasoning_summary")
        },
    })
    return context