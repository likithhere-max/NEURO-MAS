"""
Typed, hypothesis-specific evidence (Phase 2, D2).

Why this exists
---------------
The previous consensus treated "an agent's confidence" as a vote. Different agents answer different questions:

  DIAGNOSTIC          "how well do this patient's symptoms match hypothesis H?"          (may support / contradict H)
  VALIDATED_MODEL     a trained model that has passed a documented validation protocol    (may vote; only if validated)
  LONGITUDINAL        within-patient change over time. Not tied to a named disease, so it can only vote for a
                      hypothesis if an explicit, documented mapping exists (none ships by default)
  KNOWLEDGE_RETRIEVAL literature/KB citations that look up the SAME candidates the diagnostic agent produced.
                      Circular, therefore context only: it never votes
  DATA_QUALITY        missingness, contradictions in the record, OOD abstentions. May LIMIT or CONTRADICT; never supports

Only `votes_for_ranking(...)` evidence enters the ranking. Everything else is still reported, so the clinician sees
it, but it cannot inflate a hypothesis.

`independence_group` stops one underlying signal from being counted twice: evidence items that share a group are
treated as ONE observation (strongest item kept), because e.g. a lexical match and a semantic match on the same
symptom text are not independent confirmations.
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class EvidenceType(str, Enum):
    DIAGNOSTIC = "diagnostic"
    VALIDATED_MODEL = "validated_model"
    LONGITUDINAL = "longitudinal"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    DATA_QUALITY = "data_quality"


class Direction(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"


# What each type is allowed to do. The table is the policy; tests pin it.
CAN_SUPPORT = {EvidenceType.DIAGNOSTIC, EvidenceType.VALIDATED_MODEL, EvidenceType.LONGITUDINAL}
CAN_CONTRADICT = {EvidenceType.DIAGNOSTIC, EvidenceType.VALIDATED_MODEL, EvidenceType.LONGITUDINAL,
                  EvidenceType.DATA_QUALITY}


class HypothesisEvidence(BaseModel):
    source_agent: str
    hypothesis: str
    evidence_type: EvidenceType
    direction: Direction = Direction.SUPPORTS
    # Strength in [0, 1] on the scale named by `score_kind`. None = the source cannot quantify it.
    strength: Optional[float] = None
    score_kind: str = "unspecified"           # e.g. "retrieval_similarity_capped", "model_probability_uncalibrated"
    independence_group: str = ""              # items with the same group are ONE observation; "" -> source_agent
    provenance: str = ""                      # where the number came from (method/version/dataset)
    validated: bool = False                   # True only when a documented validation protocol was passed
    out_of_domain: bool = False               # evidence computed on an input outside the model's domain
    note: str = ""

    @field_validator("strength")
    @classmethod
    def _finite_unit(cls, v):
        if v is None:
            return None
        if not math.isfinite(v):
            return None
        return min(1.0, max(0.0, float(v)))

    @property
    def group(self) -> str:
        return self.independence_group or self.source_agent

    def votes_for_ranking(self) -> tuple[bool, str]:
        """(eligible, reason). Eligibility is a pure function of the item, so it is auditable."""
        if self.out_of_domain:
            return False, "out_of_domain: model output on an input outside its domain is excluded"
        if self.strength is None:
            return False, "unscored: source could not quantify this evidence"
        if self.evidence_type == EvidenceType.KNOWLEDGE_RETRIEVAL:
            return False, "knowledge_retrieval is circular (looks up the diagnostic agent's own candidates): context only"
        if self.direction == Direction.SUPPORTS and self.evidence_type not in CAN_SUPPORT:
            return False, f"{self.evidence_type.value} evidence cannot support a diagnosis"
        if self.direction == Direction.CONTRADICTS and self.evidence_type not in CAN_CONTRADICT:
            return False, f"{self.evidence_type.value} evidence cannot contradict a diagnosis"
        if self.evidence_type == EvidenceType.VALIDATED_MODEL and not self.validated:
            return False, "model has not passed a documented validation protocol: reported, not ranked"
        return True, ""
