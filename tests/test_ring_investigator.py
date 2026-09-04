from datetime import datetime, timedelta, timezone

import pytest

from events.schema import TransactionChannel, TransactionEvent, WorldId
from features.temporal import TemporalFeatureState
from graph.behavioral_state import BehavioralState
from verifier.contracts import EvidenceContext, EvidenceType, VerificationRequest
from verifier.ring_investigator import RingInvestigator, RingInvestigatorConfig

T0 = datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc)


def tx(account: str = "a1", merchant: str = "m0") -> TransactionEvent:
    return TransactionEvent(
        event_id=f"alert-{account}",
        world_id=WorldId.WORLD_B,
        timestamp=T0,
        account_id=account,
        merchant_id=merchant,
        amount=100.0,
        channel=TransactionChannel.UPI,
        device_id="d1",
        ip_prefix="10.0.0.0/24",
    )


def request() -> VerificationRequest:
    return VerificationRequest(
        alert_event=tx(),
        detector_probability=0.8,
        detector_threshold=0.01,
        detector_model="temporal@world-a-freeze",
        alerted_at=T0,
    )


def context(graph: BehavioralState, temporal: TemporalFeatureState) -> EvidenceContext:
    return EvidenceContext(
        as_of=T0,
        state={
            "behavioral_state": graph,
            "temporal_feature_state": temporal,
        },
    )


def seed_ring() -> tuple[BehavioralState, TemporalFeatureState]:
    graph = BehavioralState()
    temporal = TemporalFeatureState()

    rows = [
        ("a1", "a2", 8),
        ("a2", "a1", 7),
        ("a1", "a3", 6),
        ("a3", "a1", 5),
        ("a2", "a3", 4),
        ("a3", "a4", 3),
        ("a4", "a1", 2),
    ]
    rows = sorted(rows, key=lambda row: row[2], reverse=True)
    for sender, receiver, hours_ago in rows:
        ts = T0 - timedelta(hours=hours_ago)
        graph.update(sender, receiver, ts.isoformat().replace("+00:00", "Z"), 50.0)
        temporal.update(
            TransactionEvent(
                event_id=f"p2p-{sender}-{receiver}-{hours_ago}",
                world_id=WorldId.WORLD_B,
                timestamp=ts,
                account_id=sender,
                counterparty_account_id=receiver,
                amount=50.0,
                channel=TransactionChannel.UPI,
                device_id="d",
                ip_prefix="10.0.0.0/24",
            )
        )

    for account, merchant in [
        ("a1", "m1"),
        ("a1", "m2"),
        ("a2", "m1"),
        ("a2", "m2"),
        ("a3", "m1"),
        ("a3", "m2"),
    ]:
        temporal.update(
            TransactionEvent(
                event_id=f"merchant-{account}-{merchant}",
                world_id=WorldId.WORLD_B,
                timestamp=T0 - timedelta(hours=2),
                account_id=account,
                merchant_id=merchant,
                amount=75.0,
                channel=TransactionChannel.CARD,
                device_id="d",
                ip_prefix="10.0.0.0/24",
            )
        )

    return graph, temporal


def test_ring_investigator_emits_structural_evidence():
    graph, temporal = seed_ring()
    evidence = RingInvestigator().collect(request(), context(graph, temporal))

    types = {item.evidence_type for item in evidence}
    assert EvidenceType.RING_STRUCTURE in types
    assert EvidenceType.PEER_SYNCHRONY in types
    assert EvidenceType.MERCHANT_CONVERGENCE in types
    assert all(item.observed_at == T0 for item in evidence)
    assert all(item.source_event_ids == ("alert-a1",) for item in evidence)


def test_no_current_transaction_is_added_before_investigation():
    graph = BehavioralState()
    temporal = TemporalFeatureState()
    evidence = RingInvestigator().collect(request(), context(graph, temporal))
    assert evidence == []


def test_future_context_is_rejected():
    graph, temporal = seed_ring()
    future = EvidenceContext(
        as_of=T0 + timedelta(seconds=1),
        state={"behavioral_state": graph, "temporal_feature_state": temporal},
    )
    with pytest.raises(ValueError, match="after verifier decision_time"):
        RingInvestigator().collect(request(), future)


def test_wrong_context_time_is_rejected():
    graph, temporal = seed_ring()
    before = EvidenceContext(
        as_of=T0 - timedelta(seconds=1),
        state={"behavioral_state": graph, "temporal_feature_state": temporal},
    )
    with pytest.raises(ValueError, match="to equal alert event time"):
        RingInvestigator().collect(request(), before)


def test_neighbor_scan_is_bounded_deterministically():
    graph, temporal = seed_ring()
    config = RingInvestigatorConfig(max_neighbors_scanned=2)
    evidence = RingInvestigator(config).collect(request(), context(graph, temporal))
    assert all(item.metric_value is not None for item in evidence)
