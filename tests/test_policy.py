from datetime import datetime, timezone

import pytest

from events.schema import EventType, TransactionEvent, WorldId
from verifier.contracts import (
    EvidenceBundle,
    EvidenceItem,
    EvidenceStrength,
    EvidenceType,
    VerificationRequest,
)
from verifier.policy import AutoResponderPolicy, AutoResponderPolicyConfig, PolicyAction, RiskTier


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


def item(eid: str, agent: str, strength: EvidenceStrength) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=eid,
        evidence_type=EvidenceType.RING_STRUCTURE,
        strength=strength,
        source_agent=agent,
        summary="corroborating evidence",
        confidence=0.9,
        observed_at=TS,
        source_event_ids=("evt-1",),
        subject_account_ids=("acc-1",),
    )


def bundle(items: tuple[EvidenceItem, ...] = ()) -> EvidenceBundle:
    return EvidenceBundle(
        alert_event_id="evt-1",
        decision_time=TS,
        verifier_version="10e-test",
        items=items,
    )


def test_below_alert_threshold_allows():
    decision = AutoResponderPolicy().decide(request(0.009), bundle(), 0.009)
    assert decision.action is PolicyAction.ALLOW
    assert decision.risk_tier is RiskTier.LOW


def test_alert_without_corroboration_goes_to_review():
    decision = AutoResponderPolicy().decide(request(0.05), bundle(), 0.05)
    assert decision.action is PolicyAction.REVIEW
    assert decision.risk_tier is RiskTier.ELEVATED


def test_high_alert_needs_multi_agent_and_strong_evidence_to_block():
    items = (item("e1", "ring-investigator", EvidenceStrength.STRONG),
             item("e2", "infrastructure-investigator", EvidenceStrength.MODERATE))
    decision = AutoResponderPolicy().decide(request(0.8), bundle(items), 0.7)
    assert decision.action is PolicyAction.BLOCK
    assert decision.agent_coverage == 2
    assert decision.strong_evidence_count == 1


def test_one_agent_strong_evidence_does_not_block():
    items = (item("e1", "ring-investigator", EvidenceStrength.STRONG),)
    decision = AutoResponderPolicy().decide(request(0.8), bundle(items), 0.7)
    assert decision.action is PolicyAction.REVIEW


def test_detector_alone_cannot_block():
    decision = AutoResponderPolicy().decide(request(0.99), bundle(), 0.99)
    assert decision.action is PolicyAction.REVIEW


def test_policy_does_not_execute_actions():
    items = (item("e1", "ring-investigator", EvidenceStrength.STRONG),
             item("e2", "context-investigator", EvidenceStrength.STRONG))
    decision = AutoResponderPolicy().decide(request(0.9), bundle(items), 0.9)
    assert decision.action is PolicyAction.BLOCK
    assert decision.policy_version == "11a-initial-v1"


def test_bundle_must_match_alert():
    bad = EvidenceBundle(alert_event_id="evt-other", decision_time=TS, verifier_version="x", items=())
    with pytest.raises(ValueError, match="different alert event"):
        AutoResponderPolicy().decide(request(), bad, 0.2)


def test_bundle_decision_time_must_match_request():
    later = EvidenceBundle(alert_event_id="evt-1", decision_time=TS.replace(minute=1), verifier_version="x", items=())
    with pytest.raises(ValueError, match="decision_time"):
        AutoResponderPolicy().decide(request(), later, 0.2)


def test_invalid_confidence_rejected():
    with pytest.raises(ValueError, match="verification_confidence"):
        AutoResponderPolicy().decide(request(), bundle(), 1.2)


def test_config_prevents_unsafe_block_threshold():
    with pytest.raises(ValueError):
        AutoResponderPolicyConfig(detector_alert_threshold=0.2, block_detector_threshold=0.1)
