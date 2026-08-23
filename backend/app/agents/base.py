"""
BaseNeuroAgent: the common contract every specialized agent implements.

Spec section 35. Keeping this interface tiny and uniform is what lets the
orchestrator treat wildly different analyses (an EEG-signal check vs a
drug-interaction check) the same way: give it a task + the shared patient
state, get back a structured AgentResult.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.agent import AgentResult, AgentTask
from app.schemas.patient_state import PatientState


class BaseNeuroAgent(ABC):
    name: str = "base_agent"
    capabilities: list[str] = []
    # Reliability weight used by the consensus engine (w_i). Calibrated
    # informally for this prototype -- NOT derived from clinical validation.
    reliability_weight: float = 0.7

    @abstractmethod
    async def execute(self, task: AgentTask, state: PatientState) -> AgentResult:
        """Run this agent's analysis and return a structured result.

        Implementations must never raise for "normal" bad conditions (missing
        data, low signal quality, model unavailable) -- they should return an
        AgentResult with an appropriate status/warnings instead, so the
        orchestrator can keep the workflow moving in degraded mode.
        """
        raise NotImplementedError
