from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
import json
from math import isfinite
from typing import Iterable

from verifier.action_execution import ActionExecutionReceipt, ActionExecutionStatus
from verifier.contracts import EvidenceBundle, VerificationRequest
from verifier.policy import PolicyDecision


class AuditEventType(str, Enum):
    """Lifecycle event recorded by the responder audit trail."""

    POLICY_DECISION = "policy_decision"
    ACTION_EXECUTED = "action_executed"
    ACTION_IDEMPOTENT_REPLAY = "action_idempotent_replay"
    ACTION_FAILED = "action_failed"


class AuditTrailError(RuntimeError):
    """Base error for invalid audit-trail operations."""


@dataclass(frozen=True)
class AuditRecord:
    """Immutable, hash-addressed audit record for one responder lifecycle event."""

    sequence: int
    record_id: str
    event_type: AuditEventType
    event_id: str
    account_id: str
    recorded_at: datetime
    detector_probability: float
    detector_threshold: float
    detector_model: str
    policy_version: str
    action: str
    risk_tier: str
    verification_confidence: float
    agent_coverage: int
    evidence_count: int
    strong_evidence_count: int
    reason_codes: tuple[str, ...]
    execution_id: str | None
    idempotency_key: str | None
    execution_status: str | None
    error_type: str | None
    error_message: str | None
    prev_hash: str
    record_hash: str


class InMemoryAuditTrail:
    """Append-only, tamper-evident reference audit store.

    The trail keeps immutable records in insertion order and hash-chains every
    record to the previous record. It is intentionally in-memory for the
    development/evaluation environment; a production adapter can persist the
    same canonical record fields to durable append-only storage.
    """

    GENESIS_HASH = "0" * 64

    def __init__(self) -> None:
        self._records: list[AuditRecord] = []

    @property
    def records(self) -> tuple[AuditRecord, ...]:
        """Return the audit trail as an immutable snapshot."""
        return tuple(self._records)

    @property
    def last_hash(self) -> str:
        return self._records[-1].record_hash if self._records else self.GENESIS_HASH

    def append_policy_decision(
        self,
        request: VerificationRequest,
        bundle: EvidenceBundle,
        decision: PolicyDecision,
        *,
        recorded_at: datetime | None = None,
    ) -> AuditRecord:
        self._validate_common(request, bundle, decision)
        return self._append(
            event_type=AuditEventType.POLICY_DECISION,
            request=request,
            decision=decision,
            recorded_at=recorded_at,
            execution_id=None,
            idempotency_key=None,
            execution_status=None,
            error_type=None,
            error_message=None,
        )

    def append_execution(
        self,
        request: VerificationRequest,
        decision: PolicyDecision,
        receipt: ActionExecutionReceipt,
        *,
        recorded_at: datetime | None = None,
    ) -> AuditRecord:
        """Record a successful execution or idempotent replay returned by 11B."""
        if receipt.event_id != request.event_id:
            raise AuditTrailError("execution receipt belongs to a different event")
        if receipt.action is not decision.action:
            raise AuditTrailError("execution receipt action does not match policy decision")
        self._validate_decision_request(request, decision)
        event_type = (
            AuditEventType.ACTION_IDEMPOTENT_REPLAY
            if receipt.status is ActionExecutionStatus.IDEMPOTENT_REPLAY
            else AuditEventType.ACTION_EXECUTED
        )
        return self._append(
            event_type=event_type,
            request=request,
            decision=decision,
            recorded_at=recorded_at or receipt.executed_at,
            execution_id=receipt.execution_id,
            idempotency_key=receipt.idempotency_key,
            execution_status=receipt.status.value,
            error_type=None,
            error_message=None,
        )

    def append_execution_failure(
        self,
        request: VerificationRequest,
        decision: PolicyDecision,
        error: Exception,
        *,
        attempted_at: datetime | None = None,
    ) -> AuditRecord:
        """Record a failed 11B attempt without pretending the action succeeded."""
        self._validate_decision_request(request, decision)
        return self._append(
            event_type=AuditEventType.ACTION_FAILED,
            request=request,
            decision=decision,
            recorded_at=attempted_at,
            execution_id=None,
            idempotency_key=None,
            execution_status="failed",
            error_type=type(error).__name__,
            error_message=str(error) or type(error).__name__,
        )

    def verify_integrity(self) -> bool:
        """Verify sequence numbers, links, and hashes across the entire trail."""
        previous = self.GENESIS_HASH
        for expected_sequence, record in enumerate(self._records, start=1):
            if record.sequence != expected_sequence:
                return False
            if record.prev_hash != previous:
                return False
            if record.record_hash != self._hash_record(record):
                return False
            previous = record.record_hash
        return True

    def _append(
        self,
        *,
        event_type: AuditEventType,
        request: VerificationRequest,
        decision: PolicyDecision,
        recorded_at: datetime | None,
        execution_id: str | None,
        idempotency_key: str | None,
        execution_status: str | None,
        error_type: str | None,
        error_message: str | None,
    ) -> AuditRecord:
        timestamp = self._timestamp(recorded_at)
        sequence = len(self._records) + 1
        previous = self.last_hash
        record_id = f"audit-{sequence:08d}"
        draft = AuditRecord(
            sequence=sequence,
            record_id=record_id,
            event_type=event_type,
            event_id=request.event_id,
            account_id=request.alert_event.account_id,
            recorded_at=timestamp,
            detector_probability=decision.detector_probability,
            detector_threshold=request.detector_threshold,
            detector_model=request.detector_model,
            policy_version=decision.policy_version,
            action=decision.action.value,
            risk_tier=decision.risk_tier.value,
            verification_confidence=decision.verification_confidence,
            agent_coverage=decision.agent_coverage,
            evidence_count=decision.evidence_count,
            strong_evidence_count=decision.strong_evidence_count,
            reason_codes=decision.reason_codes,
            execution_id=execution_id,
            idempotency_key=idempotency_key,
            execution_status=execution_status,
            error_type=error_type,
            error_message=error_message,
            prev_hash=previous,
            record_hash="",
        )
        record = AuditRecord(**{**draft.__dict__, "record_hash": self._hash_record(draft)})
        self._records.append(record)
        return record

    @staticmethod
    def _validate_common(
        request: VerificationRequest,
        bundle: EvidenceBundle,
        decision: PolicyDecision,
    ) -> None:
        if bundle.alert_event_id != request.event_id:
            raise AuditTrailError("evidence bundle belongs to a different alert event")
        if bundle.decision_time != request.decision_time:
            raise AuditTrailError("evidence bundle decision_time must match request")
        InMemoryAuditTrail._validate_decision_request(request, decision)
        if decision.evidence_count != bundle.evidence_count:
            raise AuditTrailError("decision evidence_count does not match evidence bundle")

    @staticmethod
    def _validate_decision_request(request: VerificationRequest, decision: PolicyDecision) -> None:
        if decision.detector_probability != request.detector_probability:
            raise AuditTrailError("decision detector probability does not match request")
        if not 0.0 <= decision.verification_confidence <= 1.0 or not isfinite(
            decision.verification_confidence
        ):
            raise AuditTrailError("verification_confidence must be finite and in [0, 1]")
        if not decision.policy_version.strip():
            raise AuditTrailError("policy_version must be non-empty")

    @staticmethod
    def _timestamp(value: datetime | None) -> datetime:
        timestamp = value or datetime.now(timezone.utc)
        if timestamp.tzinfo is None:
            raise ValueError("recorded_at must be timezone-aware")
        return timestamp.astimezone(timezone.utc)

    @classmethod
    def _hash_record(cls, record: AuditRecord) -> str:
        payload = {
            "sequence": record.sequence,
            "record_id": record.record_id,
            "event_type": record.event_type.value,
            "event_id": record.event_id,
            "account_id": record.account_id,
            "recorded_at": record.recorded_at.isoformat(),
            "detector_probability": record.detector_probability,
            "detector_threshold": record.detector_threshold,
            "detector_model": record.detector_model,
            "policy_version": record.policy_version,
            "action": record.action,
            "risk_tier": record.risk_tier,
            "verification_confidence": record.verification_confidence,
            "agent_coverage": record.agent_coverage,
            "evidence_count": record.evidence_count,
            "strong_evidence_count": record.strong_evidence_count,
            "reason_codes": list(record.reason_codes),
            "execution_id": record.execution_id,
            "idempotency_key": record.idempotency_key,
            "execution_status": record.execution_status,
            "error_type": record.error_type,
            "error_message": record.error_message,
            "prev_hash": record.prev_hash,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return sha256(canonical.encode("utf-8")).hexdigest()


class AuditTrailRecorder:
    """Small integration seam for recording the complete responder lifecycle."""

    def __init__(self, trail: InMemoryAuditTrail) -> None:
        self.trail = trail

    def record_decision(
        self,
        request: VerificationRequest,
        bundle: EvidenceBundle,
        decision: PolicyDecision,
        *,
        recorded_at: datetime | None = None,
    ) -> AuditRecord:
        return self.trail.append_policy_decision(
            request, bundle, decision, recorded_at=recorded_at
        )

    def record_execution(
        self,
        request: VerificationRequest,
        decision: PolicyDecision,
        receipt: ActionExecutionReceipt,
        *,
        recorded_at: datetime | None = None,
    ) -> AuditRecord:
        return self.trail.append_execution(
            request, decision, receipt, recorded_at=recorded_at
        )

    def record_execution_failure(
        self,
        request: VerificationRequest,
        decision: PolicyDecision,
        error: Exception,
        *,
        attempted_at: datetime | None = None,
    ) -> AuditRecord:
        return self.trail.append_execution_failure(
            request, decision, error, attempted_at=attempted_at
        )


__all__ = [
    "AuditEventType",
    "AuditRecord",
    "AuditTrailError",
    "AuditTrailRecorder",
    "InMemoryAuditTrail",
]
