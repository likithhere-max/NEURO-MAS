"""
Agent registry: single place the orchestrator looks up agent instances by
name. Keeping this separate from the orchestrator avoids circular imports
and duplicated instantiation logic.
"""
from __future__ import annotations

from app.agents.base import BaseNeuroAgent
from app.agents.copilot import PatientCopilotAgent
from app.agents.data_intake import DataIntakeAgent
from app.agents.diagnostic import NeuroDiagnosticAgent
from app.agents.emergency import EmergencyEscalationAgent
from app.agents.knowledge import KnowledgeAgent
from app.agents.longitudinal import LongitudinalAgent
from app.agents.medication import MedicationSafetyAgent
from app.agents.personalization import PersonalizationAgent
from app.agents.risk import RiskStratificationAgent
from app.agents.signal_analysis import SignalAnalysisAgent

_REGISTRY: dict[str, BaseNeuroAgent] = {
    a.name: a
    for a in [
        DataIntakeAgent(),
        NeuroDiagnosticAgent(),
        SignalAnalysisAgent(),
        LongitudinalAgent(),
        RiskStratificationAgent(),
        MedicationSafetyAgent(),
        PersonalizationAgent(),
        KnowledgeAgent(),
        EmergencyEscalationAgent(),
        PatientCopilotAgent(),
    ]
}

# Fixed execution order groups. Agents within a group conceptually run in
# parallel (spec: "PARALLEL AGENTS"); groups run in sequence because later
# groups consume earlier groups' outputs.
WORKFLOW_STAGES: list[list[str]] = [
    ["data_intake_agent"],
    ["emergency_escalation_agent"],
    ["neuro_diagnostic_agent", "signal_analysis_agent"],
    ["longitudinal_agent", "medication_safety_agent", "knowledge_agent"],
    ["risk_stratification_agent"],
    ["personalization_agent"],
]


def get_agent(name: str) -> BaseNeuroAgent:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown agent: {name}")
    return _REGISTRY[name]


def all_agents() -> dict[str, BaseNeuroAgent]:
    return _REGISTRY
