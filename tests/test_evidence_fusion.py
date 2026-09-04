from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from events.schema import TransactionEvent, TransactionChannel, WorldId
from verifier.contracts import (
    EvidenceItem,
    EvidenceStrength,
    EvidenceType,
    VerificationRequest,
)
from verifier.evidence_fusion import DeterministicEvidenceFusion, EvidenceFusionConfig


def _request() -> VerificationRequest:
    t = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    event = TransactionEvent(
        event_id="evt-1",
        timestamp=t,
        world_id=WorldId.WORLD_A,
        account_id="acc-1",
        merchant_id="m-1",
        counterparty_account_id=None,
        amount=1000.0,
        channel=TransactionChannel.UPI,
        device_id="dev-1",
        ip_prefix="10.0.0.1/24",
    )
    return VerificationRequest(
        alert_event=event,
        detector_probability=0.40,
        detector_threshold=0.01,
        detector_model="temporal-world-a-frozen",
        alerted_at=t,
    )


def _item(
    evidence_id: str,
    evidence_type: EvidenceType,
    strength: EvidenceStrength,
    confidence: float,
    agent: str = "ring-investigator",
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        evidence_type=evidence_type,
        strength=strength,
        source_agent=agent,
        summary="test evidence",
        confidence=confidence,
        observed_at=_request().alert_event.timestamp,
        source_event_ids=("evt-1",),
        subject_account_ids=("acc-1",),
        metric_name="test_metric",
        metric_value=confidence,
    )


def test_empty_evidence_preserves_detector_score() -> None:
    request = _request()
    fusion = DeterministicEvidenceFusion()
    assert fusion.fuse(request, []) == pytest.approx(request.detector_probability)


def test_fusion_never_reduces_detector_score() -> None:
    request = _request()
    fusion = DeterministicEvidenceFusion()
    assert fusion.fuse(request, []) == pytest.approx(request.detector_probability)
    assert fusion.fuse(request, [_item("a", EvidenceType.RING_STRUCTURE, EvidenceStrength.STRONG, 1.0)]) >= request.detector_probability


def test_stronger_evidence_increases_confidence() -> None:
    request = _request()
    fusion = DeterministicEvidenceFusion()
    weak = fusion.fuse(request, [_item("a", EvidenceType.RING_STRUCTURE, EvidenceStrength.WEAK, 0.8)])
    strong = fusion.fuse(request, [_item("b", EvidenceType.RING_STRUCTURE, EvidenceStrength.STRONG, 0.8)])
    assert strong > weak


def test_same_type_does_not_stack_linearly() -> None:
    request = _request()
    fusion = DeterministicEvidenceFusion()
    one = fusion.fuse(request, [_item("a", EvidenceType.RING_STRUCTURE, EvidenceStrength.STRONG, 1.0)])
    two = fusion.fuse(
        request,
        [
            _item("a", EvidenceType.RING_STRUCTURE, EvidenceStrength.STRONG, 1.0),
            _item("b", EvidenceType.RING_STRUCTURE, EvidenceStrength.STRONG, 1.0),
        ],
    )
    assert two == pytest.approx(one)


def test_multiple_investigators_add_bounded_coverage_bonus() -> None:
    request = _request()
    fusion = DeterministicEvidenceFusion()
    one = fusion.fuse(request, [_item("a", EvidenceType.RING_STRUCTURE, EvidenceStrength.STRONG, 1.0)])
    three = fusion.fuse(
        request,
        [
            _item("a", EvidenceType.RING_STRUCTURE, EvidenceStrength.STRONG, 1.0, "ring-investigator"),
            _item("b", EvidenceType.INFRASTRUCTURE_SHARING, EvidenceStrength.STRONG, 1.0, "infrastructure-investigator"),
            _item("c", EvidenceType.TEMPORAL_CONTEXT, EvidenceStrength.STRONG, 1.0, "context-investigator"),
        ],
    )
    assert three > one
    assert three - one < 0.35


def test_order_does_not_change_result() -> None:
    request = _request()
    fusion = DeterministicEvidenceFusion()
    items = [
        _item("a", EvidenceType.RING_STRUCTURE, EvidenceStrength.MODERATE, 0.8),
        _item("b", EvidenceType.TEMPORAL_CONTEXT, EvidenceStrength.STRONG, 0.9, "context-investigator"),
        _item("c", EvidenceType.PEER_SYNCHRONY, EvidenceStrength.WEAK, 0.9),
    ]
    assert fusion.fuse(request, items) == pytest.approx(fusion.fuse(request, list(reversed(items))))


def test_duplicate_evidence_id_rejected() -> None:
    request = _request()
    item = _item("dup", EvidenceType.RING_STRUCTURE, EvidenceStrength.STRONG, 1.0)
    with pytest.raises(ValueError, match="duplicate evidence_id"):
        DeterministicEvidenceFusion().fuse(request, [item, item])


def test_future_evidence_rejected() -> None:
    request = _request()
    item = _item("future", EvidenceType.RING_STRUCTURE, EvidenceStrength.STRONG, 1.0)
    future = item.model_copy(update={"observed_at": request.alert_event.timestamp + timedelta(seconds=1)})
    with pytest.raises(ValueError, match="newer than (decision_time|alert event time)"):
        DeterministicEvidenceFusion().fuse(request, [future])


def test_custom_config_is_validated() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        EvidenceFusionConfig(type_weights={e: 0.0 for e in EvidenceType})


def test_explain_is_auditable() -> None:
    request = _request()
    item = _item("a", EvidenceType.RING_STRUCTURE, EvidenceStrength.STRONG, 0.8)
    breakdown = DeterministicEvidenceFusion().explain(request, [item])
    assert breakdown.detector_probability == pytest.approx(0.40)
    assert breakdown.evidence_count == 1
    assert breakdown.strongest_by_type[EvidenceType.RING_STRUCTURE] == pytest.approx(0.8)
    assert breakdown.contributing_agent_names == ("ring-investigator",)
    assert 0.0 <= breakdown.fused_confidence <= 1.0

def test_full_agent_coverage_bonus_is_bounded() -> None:
    request = _request()
    fusion = DeterministicEvidenceFusion()
    breakdown = fusion.explain(request, [
        _item("a", EvidenceType.RING_STRUCTURE, EvidenceStrength.STRONG, 1.0, "ring-investigator"),
        _item("b", EvidenceType.INFRASTRUCTURE_SHARING, EvidenceStrength.STRONG, 1.0, "infrastructure-investigator"),
        _item("c", EvidenceType.TEMPORAL_CONTEXT, EvidenceStrength.STRONG, 1.0, "context-investigator"),
    ])
    assert breakdown.agent_coverage == pytest.approx(1.0)
    assert breakdown.coverage_bonus == pytest.approx(0.08)
