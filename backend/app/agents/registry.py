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

# Dependency graph: an agent starts only after every agent it lists has FINISHED (completed, failed, timed out,
# abstained...). Dependencies are ordering/data-flow edges, not success requirements: a failed dependency leaves the
# dependent running in degraded mode with that upstream status visible, never silently absent.
# Independent agents run concurrently (bounded by settings.max_agent_concurrency).
AGENT_DEPENDENCIES: dict[str, list[str]] = {
    "data_intake_agent": [],
    "emergency_escalation_agent": ["data_intake_agent"],
    "neuro_diagnostic_agent": ["data_intake_agent"],
    "signal_analysis_agent": ["data_intake_agent"],
    "longitudinal_agent": ["data_intake_agent"],
    "medication_safety_agent": ["data_intake_agent"],
    "knowledge_agent": ["neuro_diagnostic_agent"],
    "risk_stratification_agent": ["data_intake_agent", "neuro_diagnostic_agent", "signal_analysis_agent",
                                  "longitudinal_agent", "emergency_escalation_agent"],
    "personalization_agent": ["risk_stratification_agent", "longitudinal_agent", "medication_safety_agent"],
}


def validate_dependencies(deps: dict[str, list[str]]) -> list[list[str]]:
    """Topological layers; raises ValueError on unknown nodes or cycles."""
    for n, ds in deps.items():
        for d in ds:
            if d not in deps:
                raise ValueError(f"{n} depends on unknown agent {d}")
    remaining, layers, done = dict(deps), [], set()
    while remaining:
        ready = sorted(n for n, ds in remaining.items() if all(d in done for d in ds))
        if not ready:
            raise ValueError(f"dependency cycle among {sorted(remaining)}")
        layers.append(ready)
        done.update(ready)
        for n in ready:
            del remaining[n]
    return layers


def ancestors(name: str, deps: dict[str, list[str]] = AGENT_DEPENDENCIES) -> set[str]:
    out: set[str] = set()
    stack = list(deps.get(name, []))
    while stack:
        d = stack.pop()
        if d not in out:
            out.add(d)
            stack.extend(deps.get(d, []))
    return out


# Derived (for the /agents/status endpoint and documentation): layers of agents that CAN run concurrently.
WORKFLOW_STAGES: list[list[str]] = validate_dependencies(AGENT_DEPENDENCIES)


def get_agent(name: str) -> BaseNeuroAgent:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown agent: {name}")
    return _REGISTRY[name]


def all_agents() -> dict[str, BaseNeuroAgent]:
    return _REGISTRY
