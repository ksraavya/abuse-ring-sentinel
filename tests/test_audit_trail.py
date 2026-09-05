from datetime import datetime, timezone

import pytest

from events.schema import EventType, TransactionEvent, WorldId
from verifier.action_execution import ActionExecutionStatus, ActionExecutor, InMemoryActionExecutionBackend
from verifier.audit_trail import AuditEventType, AuditTrailError, InMemoryAuditTrail
from verifier.contracts import EvidenceBundle, EvidenceItem, EvidenceStrength, EvidenceType, VerificationRequest
from verifier.policy import AutoResponderPolicy, PolicyAction

TS = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def request(score: float = 0.8) -> VerificationRequest:
    event = TransactionEvent(
        event_id="evt-1",
        event_type=EventType.TRANSACTION,
        world_id=WorldId.WORLD_C,
        timestamp=TS,
        account_id="acc-1",
        merchant_id="merch-1",
        counterparty_account_id=None,
        amount=100.0,
        channel="upi",
        device_id="dev-1",
        ip_prefix="10.0.0",
    )
    return VerificationRequest(
        alert_event=event,
        detector_probability=score,
        detector_threshold=0.01,
        detector_model="temporal-world-a-frozen",
        alerted_at=TS,
    )


def build_decision(score: float = 0.8):
    req = request(score)
    items = (
        EvidenceItem(
            evidence_id="e1",
            evidence_type=EvidenceType.RING_STRUCTURE,
            strength=EvidenceStrength.STRONG,
            source_agent="ring-investigator",
            summary="ring corroboration",
            confidence=0.9,
            observed_at=TS,
            source_event_ids=(req.event_id,),
            subject_account_ids=("acc-1",),
        ),
        EvidenceItem(
            evidence_id="e2",
            evidence_type=EvidenceType.INFRASTRUCTURE_SHARING,
            strength=EvidenceStrength.MODERATE,
            source_agent="infrastructure-investigator",
            summary="shared infrastructure",
            confidence=0.9,
            observed_at=TS,
            source_event_ids=(req.event_id,),
            subject_account_ids=("acc-1",),
        ),
    )
    bundle = EvidenceBundle(
        alert_event_id=req.event_id,
        decision_time=TS,
        verifier_version="10e-test",
        items=items,
    )
    decision = AutoResponderPolicy().decide(req, bundle, 0.7)
    return req, bundle, decision


def test_policy_decision_record_contains_audit_context():
    req, bundle, decision = build_decision()
    trail = InMemoryAuditTrail()
    record = trail.append_policy_decision(req, bundle, decision, recorded_at=TS)

    assert record.sequence == 1
    assert record.event_type is AuditEventType.POLICY_DECISION
    assert record.action == PolicyAction.BLOCK.value
    assert record.detector_probability == req.detector_probability
    assert record.verification_confidence == decision.verification_confidence
    assert record.agent_coverage == 2
    assert record.evidence_count == 2
    assert record.strong_evidence_count == 1
    assert record.execution_id is None
    assert trail.verify_integrity()


def test_execution_receipt_is_recorded_separately_from_policy_decision():
    req, bundle, decision = build_decision()
    trail = InMemoryAuditTrail()
    trail.append_policy_decision(req, bundle, decision, recorded_at=TS)
    receipt = ActionExecutor(InMemoryActionExecutionBackend()).execute(req, decision, executed_at=TS)
    record = trail.append_execution(req, decision, receipt)

    assert record.sequence == 2
    assert record.event_type is AuditEventType.ACTION_EXECUTED
    assert record.execution_id == receipt.execution_id
    assert record.idempotency_key == receipt.idempotency_key
    assert record.execution_status == ActionExecutionStatus.EXECUTED.value
    assert record.error_type is None
    assert trail.verify_integrity()


def test_idempotent_replay_gets_distinct_audit_event():
    req, bundle, decision = build_decision()
    backend = InMemoryActionExecutionBackend()
    executor = ActionExecutor(backend)
    trail = InMemoryAuditTrail()
    receipt1 = executor.execute(req, decision, executed_at=TS)
    receipt2 = executor.execute(req, decision, executed_at=TS.replace(second=1))

    first = trail.append_execution(req, decision, receipt1, recorded_at=TS)
    second = trail.append_execution(req, decision, receipt2, recorded_at=TS.replace(second=1))

    assert first.event_type is AuditEventType.ACTION_EXECUTED
    assert second.event_type is AuditEventType.ACTION_IDEMPOTENT_REPLAY
    assert second.execution_id == first.execution_id
    assert second.sequence == 2
    assert trail.verify_integrity()


def test_execution_failure_is_audited_without_claiming_success():
    req, bundle, decision = build_decision()
    trail = InMemoryAuditTrail()
    error = RuntimeError("provider unavailable")
    record = trail.append_execution_failure(req, decision, error, attempted_at=TS)

    assert record.event_type is AuditEventType.ACTION_FAILED
    assert record.execution_status == "failed"
    assert record.error_type == "RuntimeError"
    assert record.error_message == "provider unavailable"
    assert record.execution_id is None
    assert record.idempotency_key is None
    assert trail.verify_integrity()


def test_tampering_breaks_hash_chain():
    req, bundle, decision = build_decision()
    trail = InMemoryAuditTrail()
    trail.append_policy_decision(req, bundle, decision, recorded_at=TS)
    receipt = ActionExecutor(InMemoryActionExecutionBackend()).execute(req, decision, executed_at=TS)
    trail.append_execution(req, decision, receipt, recorded_at=TS)

    original = trail._records[0]
    trail._records[0] = original.__class__(**{**original.__dict__, "action": "allow"})
    assert not trail.verify_integrity()


def test_records_are_returned_as_immutable_snapshot():
    req, bundle, decision = build_decision()
    trail = InMemoryAuditTrail()
    trail.append_policy_decision(req, bundle, decision, recorded_at=TS)
    snapshot = trail.records
    assert isinstance(snapshot, tuple)
    with pytest.raises(AttributeError):
        snapshot[0].action = "allow"


def test_mismatched_receipt_is_rejected():
    req, bundle, decision = build_decision()
    other_req, _, other_decision = build_decision()
    other_req = other_req.model_copy(update={"alert_event": other_req.alert_event.model_copy(update={"event_id": "evt-2"})})
    backend = InMemoryActionExecutionBackend()
    receipt = ActionExecutor(backend).execute(req, decision, executed_at=TS)
    trail = InMemoryAuditTrail()

    with pytest.raises(AuditTrailError, match="different event"):
        trail.append_execution(other_req, other_decision, receipt)

def test_empty_trail_integrity_passes():
    assert InMemoryAuditTrail().verify_integrity() is True
