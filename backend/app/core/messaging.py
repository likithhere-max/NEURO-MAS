"""
Structured agent-to-agent communication contract.

Per the project spec (section 7): agents must NOT pass arbitrary strings.
Every inter-agent exchange is an AgentMessage instance with a typed payload,
confidence, evidence and priority — enabling the orchestrator to reason
about *what kind* of thing it just received rather than parsing prose.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    TASK = "TASK"
    OBSERVATION = "OBSERVATION"
    ANALYSIS = "ANALYSIS"
    HYPOTHESIS = "HYPOTHESIS"
    EVIDENCE = "EVIDENCE"
    CONFLICT = "CONFLICT"
    CONSENSUS_REQUEST = "CONSENSUS_REQUEST"
    CONSENSUS_RESULT = "CONSENSUS_RESULT"
    ACTION_PROPOSAL = "ACTION_PROPOSAL"
    ACTION_APPROVAL = "ACTION_APPROVAL"
    ALERT = "ALERT"
    ESCALATION = "ESCALATION"
    ERROR = "ERROR"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EvidenceItem(BaseModel):
    """A single piece of evidence backing an agent's finding.

    `source_type` distinguishes a patient-specific data point from a
    knowledge-base citation, so downstream consumers (and the UI) never
    conflate "this patient's data" with "general medical literature."
    """

    source_type: str  # "patient_data" | "knowledge_base" | "signal_analysis" | "heuristic"
    description: str
    reference: Optional[str] = None  # e.g. source URL, observation id
    strength: float = Field(default=0.5, ge=0.0, le=1.0)


class AgentMessage(BaseModel):
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str
    patient_id: str
    sender_agent: str
    receiver_agent: str = "orchestrator"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    message_type: MessageType
    priority: Priority = Priority.LOW
    payload: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    requires_response: bool = False

    model_config = {"use_enum_values": True}
