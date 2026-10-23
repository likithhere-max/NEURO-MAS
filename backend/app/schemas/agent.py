"""
Agent task/result contracts and the BaseNeuroAgent interface (spec section 35).
"""
from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.core.evidence import HypothesisEvidence
from app.core.messaging import EvidenceItem


class AgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    MODEL_NOT_AVAILABLE = "model_not_available"  # legacy alias of ABSTAINED for model-backed agents
    # Phase 2 typed outcomes. Only COMPLETED results may contribute evidence to a consensus.
    ABSTAINED = "abstained"                  # agent ran and declined to assess (out of domain / not applicable)
    TIMED_OUT = "timed_out"                  # per-agent or global deadline reached; no result
    INSUFFICIENT_DATA = "insufficient_data"  # agent ran; the data cannot support any assessment


# Statuses whose output may be used as evidence. Everything else is reported but never votes.
EVIDENCE_ELIGIBLE_STATUSES = frozenset({"completed"})


class AgentTask(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str
    patient_id: str
    agent_name: str
    reason: str
    input_payload: dict[str, Any] = Field(default_factory=dict)


class AgentFinding(BaseModel):
    label: str
    detail: str
    severity: Optional[str] = None  # "info" | "low" | "medium" | "high" | "critical"
    value: Optional[float] = None


class AgentResult(BaseModel):
    agent_name: str
    task_id: str
    status: AgentStatus
    findings: list[AgentFinding] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    recommended_next_agents: list[str] = Field(default_factory=list)
    requires_human_review: bool = False
    reasoning_summary: str = ""
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    # Typed, hypothesis-specific evidence (see app/core/evidence.py). Empty for agents that make
    # patient-level statements only; those must not be counted as votes for any named disease.
    hypothesis_evidence: list["HypothesisEvidence"] = Field(default_factory=list)
    agent_version: Optional[str] = None
    duration_ms: Optional[float] = None

    model_config = {"use_enum_values": True}
