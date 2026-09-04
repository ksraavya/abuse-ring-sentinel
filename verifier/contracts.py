from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from events.schema import TransactionEvent


class EvidenceType(str, Enum):
    """Controlled vocabulary for verifier evidence provenance."""

    RING_STRUCTURE = "ring_structure"
    BEHAVIORAL_ACCELERATION = "behavioral_acceleration"
    PEER_SYNCHRONY = "peer_synchrony"
    MERCHANT_CONVERGENCE = "merchant_convergence"
    INFRASTRUCTURE_SHARING = "infrastructure_sharing"
    INFRASTRUCTURE_CHURN = "infrastructure_churn"
    ACCOUNT_CONTEXT = "account_context"
    TEMPORAL_CONTEXT = "temporal_context"


class EvidenceStrength(str, Enum):
    """Qualitative strength used for explanation and later policy fusion."""

    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


class VerificationRequest(BaseModel):
    """Immutable input contract presented to the verifier for one alert.

    The verifier receives the detector's score and the current transaction.
    Historical graph/state context is supplied separately by the runtime so
    investigators cannot accidentally treat post-event state as evidence.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    alert_event: TransactionEvent
    detector_probability: float = Field(ge=0.0, le=1.0)
    detector_threshold: float = Field(ge=0.0, le=1.0)
    detector_model: str = Field(min_length=1)
    alerted_at: datetime

    @field_validator("alerted_at")
    @classmethod
    def validate_alerted_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("alerted_at must be timezone-aware")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_alert(self) -> "VerificationRequest":
        event_time = self.alert_event.timestamp.astimezone(timezone.utc)
        if self.alerted_at < event_time:
            raise ValueError("alerted_at cannot precede the transaction timestamp")
        return self

    @property
    def event_id(self) -> str:
        return self.alert_event.event_id

    @property
    def decision_time(self) -> datetime:
        return self.alerted_at


class EvidenceItem(BaseModel):
    """One auditable, time-bounded piece of verifier evidence.

    Every item carries provenance back to one or more observed events. The
    evidence timestamp describes when the supporting observation was valid;
    ``EvidenceBundle`` additionally enforces that no item is newer than the
    verifier decision time.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(min_length=1)
    evidence_type: EvidenceType
    strength: EvidenceStrength
    source_agent: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    observed_at: datetime
    source_event_ids: tuple[str, ...] = Field(min_length=1)
    subject_account_ids: tuple[str, ...] = Field(min_length=1)
    metric_name: str | None = Field(default=None, min_length=1)
    metric_value: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("observed_at")
    @classmethod
    def validate_observed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")
        return value.astimezone(timezone.utc)

    @field_validator("source_event_ids", "subject_account_ids")
    @classmethod
    def validate_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value for value in values):
            raise ValueError("provenance IDs must be non-empty")
        if len(values) != len(set(values)):
            raise ValueError("provenance IDs must be unique within an evidence item")
        return values


class EvidenceBundle(BaseModel):
    """Complete evidence collection for one detector alert.

    This is an evidence contract, not an intervention policy. ALLOW/REVIEW/
    BLOCK decisions are intentionally left to a later policy layer.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    alert_event_id: str = Field(min_length=1)
    decision_time: datetime
    verifier_version: str = Field(min_length=1)
    items: tuple[EvidenceItem, ...] = Field(default_factory=tuple)

    @field_validator("decision_time")
    @classmethod
    def validate_decision_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("decision_time must be timezone-aware")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_bundle(self) -> "EvidenceBundle":
        evidence_ids = [item.evidence_id for item in self.items]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence_id values must be unique within a bundle")
        future = [
            item.evidence_id
            for item in self.items
            if item.observed_at > self.decision_time
        ]
        if future:
            raise ValueError(
                "evidence contains observations newer than decision_time: "
                + ", ".join(future)
            )
        return self

    @property
    def evidence_count(self) -> int:
        return len(self.items)


class EvidenceContext(BaseModel):
    """Read-only event-time boundary passed to evidence investigators.

    The concrete graph/state snapshots are deliberately represented as an
    opaque mapping in 10A. Later investigator commits can add typed views
    without changing the evidence-item contract.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    as_of: datetime
    state: dict[str, Any] = Field(default_factory=dict)

    @field_validator("as_of")
    @classmethod
    def validate_as_of(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        return value.astimezone(timezone.utc)


__all__ = [
    "EvidenceBundle",
    "EvidenceContext",
    "EvidenceItem",
    "EvidenceStrength",
    "EvidenceType",
    "VerificationRequest",
]
