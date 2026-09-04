from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from events.schema import TransactionChannel, TransactionEvent, WorldId
from verifier.contracts import (
    EvidenceBundle,
    EvidenceContext,
    EvidenceItem,
    EvidenceStrength,
    EvidenceType,
    VerificationRequest,
)


T0 = datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc)


def transaction() -> TransactionEvent:
    return TransactionEvent(
        event_id="evt-1",
        world_id=WorldId.WORLD_B,
        timestamp=T0,
        account_id="acct-1",
        merchant_id="merchant-1",
        amount=100.0,
        channel=TransactionChannel.UPI,
        device_id="device-1",
        ip_prefix="10.0.0.0/24",
    )


def request() -> VerificationRequest:
    return VerificationRequest(
        alert_event=transaction(),
        detector_probability=0.42,
        detector_threshold=0.01,
        detector_model="temporal@world-a-freeze",
        alerted_at=T0,
    )


def evidence(*, observed_at: datetime = T0, evidence_id: str = "ev-1") -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        evidence_type=EvidenceType.RING_STRUCTURE,
        strength=EvidenceStrength.STRONG,
        source_agent="ring-investigator",
        summary="Observed a dense recent behavioral structure.",
        confidence=0.9,
        observed_at=observed_at,
        source_event_ids=("evt-1", "evt-0"),
        subject_account_ids=("acct-1", "acct-2"),
        metric_name="recent_peer_count",
        metric_value=5.0,
        details={"window_hours": 24},
    )


def test_verification_request_normalizes_aware_timestamp():
    result = request()
    assert result.event_id == "evt-1"
    assert result.decision_time == T0


def test_verification_request_rejects_alert_before_event():
    with pytest.raises(ValueError, match="cannot precede"):
        VerificationRequest(
            alert_event=transaction(),
            detector_probability=0.42,
            detector_threshold=0.01,
            detector_model="temporal@world-a-freeze",
            alerted_at=T0 - timedelta(seconds=1),
        )


def test_evidence_requires_provenance():
    with pytest.raises(ValueError):
        EvidenceItem(
            evidence_id="ev-1",
            evidence_type=EvidenceType.TEMPORAL_CONTEXT,
            strength=EvidenceStrength.WEAK,
            source_agent="context-investigator",
            summary="Missing provenance",
            confidence=0.5,
            observed_at=T0,
            source_event_ids=(),
            subject_account_ids=("acct-1",),
        )


def test_bundle_rejects_future_evidence():
    with pytest.raises(ValueError, match="newer than decision_time"):
        EvidenceBundle(
            alert_event_id="evt-1",
            decision_time=T0,
            verifier_version="10a-contract-v1",
            items=(evidence(observed_at=T0 + timedelta(seconds=1)),),
        )


def test_bundle_rejects_duplicate_evidence_ids():
    with pytest.raises(ValueError, match="unique"):
        EvidenceBundle(
            alert_event_id="evt-1",
            decision_time=T0,
            verifier_version="10a-contract-v1",
            items=(evidence(), evidence()),
        )


def test_bundle_accepts_multiple_causal_items():
    bundle = EvidenceBundle(
        alert_event_id="evt-1",
        decision_time=T0,
        verifier_version="10a-contract-v1",
        items=(
            evidence(evidence_id="ev-ring"),
            evidence(
                evidence_id="ev-infra",
            ).model_copy(update={"evidence_type": EvidenceType.INFRASTRUCTURE_SHARING}),
        ),
    )
    assert bundle.evidence_count == 2


def test_context_rejects_naive_as_of():
    with pytest.raises(ValueError, match="timezone-aware"):
        EvidenceContext(as_of=datetime(2026, 1, 10, 12, 0))
