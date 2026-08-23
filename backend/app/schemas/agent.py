"""
Agent task/result contracts and the BaseNeuroAgent interface (spec section 35).
"""
from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.core.messaging import EvidenceItem


class AgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    MODEL_NOT_AVAILABLE = "model_not_available"


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

    model_config = {"use_enum_values": True}
