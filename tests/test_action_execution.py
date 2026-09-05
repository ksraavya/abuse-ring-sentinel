from datetime import datetime, timezone

import pytest

from events.schema import EventType, TransactionEvent, WorldId
from verifier.action_execution import (
    ActionConflictError,
    ActionExecutionStatus,
    ActionExecutor,
    InMemoryActionExecutionBackend,
)
from verifier.contracts import EvidenceBundle, EvidenceItem, EvidenceStrength, EvidenceType, VerificationRequest
from verifier.policy import AutoResponderPolicy, PolicyAction, RiskTier, PolicyDecision


TS = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def request(score: float = 0.01) -> VerificationRequest:
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


def evidence(event_id: str = "evt-1") -> tuple[EvidenceItem, ...]:
    return (
        EvidenceItem(
            evidence_id="e1",
            evidence_type=EvidenceType.RING_STRUCTURE,
            strength=EvidenceStrength.STRONG,
            source_agent="ring-investigator",
            summary="ring corroboration",
            confidence=0.9,
            observed_at=TS,
            source_event_ids=(event_id,),
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
            source_event_ids=(event_id,),
            subject_account_ids=("acc-1",),
        ),
    )


def decision(score: float = 0.8):
    req = request(score)
    bundle = EvidenceBundle(
        alert_event_id=req.event_id,
        decision_time=TS,
        verifier_version="10e-test",
        items=evidence(),
    )
    return req, AutoResponderPolicy().decide(req, bundle, 0.7)


def test_block_decision_executes_as_block():
    req, dec = decision()
    backend = InMemoryActionExecutionBackend()
    receipt = ActionExecutor(backend).execute(req, dec, executed_at=TS)

    assert dec.action is PolicyAction.BLOCK
    assert receipt.action is PolicyAction.BLOCK
    assert receipt.status is ActionExecutionStatus.EXECUTED
    assert receipt.event_id == "evt-1"
    assert "evt-1" in backend.blocked_event_ids
    assert not backend.review_event_ids
    assert not backend.allowed_event_ids


def test_review_decision_executes_as_review():
    req = request(0.05)
    bundle = EvidenceBundle(
        alert_event_id=req.event_id,
        decision_time=TS,
        verifier_version="10e-test",
        items=(),
    )
    dec = AutoResponderPolicy().decide(req, bundle, 0.05)
    backend = InMemoryActionExecutionBackend()
    receipt = ActionExecutor(backend).execute(req, dec, executed_at=TS)

    assert dec.action is PolicyAction.REVIEW
    assert receipt.action is PolicyAction.REVIEW
    assert "evt-1" in backend.review_event_ids


def test_allow_decision_executes_as_allow():
    req = request(0.009)
    bundle = EvidenceBundle(
        alert_event_id=req.event_id,
        decision_time=TS,
        verifier_version="10e-test",
        items=(),
    )
    dec = AutoResponderPolicy().decide(req, bundle, 0.009)
    backend = InMemoryActionExecutionBackend()
    receipt = ActionExecutor(backend).execute(req, dec, executed_at=TS)

    assert dec.action is PolicyAction.ALLOW
    assert receipt.action is PolicyAction.ALLOW
    assert "evt-1" in backend.allowed_event_ids


def test_repeated_same_command_is_idempotent():
    req, dec = decision()
    backend = InMemoryActionExecutionBackend()
    executor = ActionExecutor(backend)

    first = executor.execute(req, dec, executed_at=TS)
    second = executor.execute(req, dec, executed_at=TS.replace(second=1))

    assert first.status is ActionExecutionStatus.EXECUTED
    assert second.status is ActionExecutionStatus.IDEMPOTENT_REPLAY
    assert second.execution_id == first.execution_id
    assert second.idempotency_key == first.idempotency_key
    assert len(backend.receipts) == 1


def test_contradictory_action_for_same_event_is_rejected():
    req, dec = decision()
    backend = InMemoryActionExecutionBackend()
    executor = ActionExecutor(backend)
    executor.execute(req, dec, executed_at=TS)

    review_decision = dec.__class__(
        action=PolicyAction.REVIEW,
        risk_tier=dec.risk_tier,
        policy_version=dec.policy_version,
        detector_probability=dec.detector_probability,
        verification_confidence=dec.verification_confidence,
        agent_coverage=dec.agent_coverage,
        evidence_count=dec.evidence_count,
        strong_evidence_count=dec.strong_evidence_count,
        reason_codes=dec.reason_codes,
        rationale=dec.rationale,
    )
    with pytest.raises(ActionConflictError, match="already executed"):
        executor.execute(req, review_decision, executed_at=TS)


def test_executor_rejects_detector_score_mismatch():
    req, dec = decision()
    bad = dec.__class__(
        action=dec.action,
        risk_tier=dec.risk_tier,
        policy_version=dec.policy_version,
        detector_probability=0.7,
        verification_confidence=dec.verification_confidence,
        agent_coverage=dec.agent_coverage,
        evidence_count=dec.evidence_count,
        strong_evidence_count=dec.strong_evidence_count,
        reason_codes=dec.reason_codes,
        rationale=dec.rationale,
    )
    with pytest.raises(ValueError, match="detector probability"):
        ActionExecutor(InMemoryActionExecutionBackend()).execute(req, bad, executed_at=TS)


def test_execution_time_must_be_timezone_aware():
    req, dec = decision()
    with pytest.raises(ValueError, match="timezone-aware"):
        ActionExecutor(InMemoryActionExecutionBackend()).execute(
            req, dec, executed_at=datetime(2026, 1, 1, 12, 0)
        )


def test_execution_does_not_recompute_policy():
    req, dec = decision()
    backend = InMemoryActionExecutionBackend()
    receipt = ActionExecutor(backend).execute(req, dec, executed_at=TS)

    assert receipt.action is dec.action
    assert receipt.execution_id.startswith("exec-")
    assert len(receipt.idempotency_key) == 64

def test_executor_is_policy_agnostic():
    # Executor should execute any valid decision without re-evaluating policy
    req = request(0.99)
    allow_dec = PolicyDecision(
        action=PolicyAction.ALLOW,
        risk_tier=RiskTier.LOW,
        policy_version="11a-initial-v1",
        detector_probability=0.99,
        verification_confidence=0.0,
        agent_coverage=0,
        evidence_count=0,
        strong_evidence_count=0,
        reason_codes=("manual_override",),
        rationale="Manual override for testing.",
    )
    backend = InMemoryActionExecutionBackend()
    receipt = ActionExecutor(backend).execute(req, allow_dec, executed_at=TS)
    assert receipt.action is PolicyAction.ALLOW
