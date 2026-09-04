from datetime import datetime, timedelta, timezone

import pytest

from events.schema import TransactionChannel, TransactionEvent
from features.temporal import TemporalFeatureState
from verifier.context_investigator import ContextInvestigator, ContextInvestigatorConfig
from verifier.contracts import EvidenceContext, EvidenceType, VerificationRequest


def ts(hour: int) -> datetime:
    return datetime(2026, 1, 10, hour, 0, tzinfo=timezone.utc)


def txn(event_id: str, t: datetime, *, merchant=None, counterparty=None, amount=100.0) -> TransactionEvent:
    return TransactionEvent(
        event_id=event_id,
        world_id="world_a",
        timestamp=t,
        account_id="A",
        merchant_id=merchant,
        counterparty_account_id=counterparty,
        amount=amount,
        channel=TransactionChannel.UPI,
        device_id="D1",
        ip_prefix="IP1",
    )


def request(event: TransactionEvent) -> VerificationRequest:
    return VerificationRequest(
        alert_event=event,
        detector_probability=0.8,
        detector_threshold=0.01,
        detector_model="temporal-test",
        alerted_at=event.timestamp,
    )


def context(event: TransactionEvent, state: TemporalFeatureState) -> EvidenceContext:
    return EvidenceContext(as_of=event.timestamp, state={"temporal_feature_state": state})


def test_merchant_novelty_is_pre_event_only():
    state = TemporalFeatureState()
    state.update(txn("h1", ts(1), merchant="M1", amount=100))
    state.update(txn("h2", ts(2), merchant="M2", amount=110))
    state.update(txn("h3", ts(3), merchant="M3", amount=90))
    event = txn("current", ts(4), merchant="M4", amount=120)

    items = ContextInvestigator().collect(request(event), context(event, state))

    assert any(i.evidence_type == EvidenceType.ACCOUNT_CONTEXT for i in items)
    assert all(i.observed_at <= event.timestamp for i in items)


def test_large_amount_relative_to_recent_history_emits_temporal_context():
    state = TemporalFeatureState()
    for i, amount in enumerate([100, 110, 90, 105], start=1):
        state.update(txn(f"h{i}", ts(i), merchant=f"M{i}", amount=amount))
    event = txn("current", ts(6), merchant="M-new", amount=500)

    items = ContextInvestigator().collect(request(event), context(event, state))

    assert any(
        i.evidence_type == EvidenceType.TEMPORAL_CONTEXT
        and i.metric_name == "amount_to_recent_median_ratio"
        for i in items
    )


def test_p2p_history_before_merchant_alert_is_captured():
    state = TemporalFeatureState()
    for i in range(1, 5):
        state.update(txn(f"p{i}", ts(i), counterparty=f"B{i}", amount=50))
    state.update(txn("merchant-history", ts(5), merchant="M0", amount=60))
    event = txn("current", ts(6), merchant="M9", amount=60)

    items = ContextInvestigator().collect(request(event), context(event, state))

    assert any(
        i.metric_name == "recent_p2p_share" and i.evidence_type == EvidenceType.ACCOUNT_CONTEXT
        for i in items
    )


def test_p2p_dominated_history_does_not_make_p2p_alert_strong():
    state = TemporalFeatureState()
    for i in range(1, 5):
        state.update(txn(f"p{i}", ts(i), counterparty=f"B{i}", amount=50))
    event = txn("current", ts(6), counterparty="B9", amount=60)

    items = ContextInvestigator().collect(request(event), context(event, state))

    assert len(items) == 1
    assert items[0].metric_name == "recent_p2p_share"
    assert items[0].strength.value == "weak"


def test_no_history_is_not_artificially_suspicious():
    state = TemporalFeatureState()
    event = txn("current", ts(1), merchant="M1", amount=100)

    assert ContextInvestigator().collect(request(event), context(event, state)) == []


def test_future_context_is_rejected():
    state = TemporalFeatureState()
    event = txn("current", ts(4), merchant="M1", amount=100)
    bad = EvidenceContext(as_of=ts(5), state={"temporal_feature_state": state})

    with pytest.raises(ValueError):
        ContextInvestigator().collect(request(event), bad)


def test_current_event_is_not_included_in_history():
    state = TemporalFeatureState()
    state.update(txn("h1", ts(1), merchant="M1", amount=100))
    event = txn("current", ts(4), merchant="M2", amount=1000)

    investigator = ContextInvestigator(
        ContextInvestigatorConfig(amount_history_min_transactions=2)
    )
    items = investigator.collect(request(event), context(event, state))

    assert all(i.metric_name != "amount_to_recent_median_ratio" for i in items)

def test_known_merchant_does_not_emit_novelty_evidence():
    state = TemporalFeatureState()
    for i in range(1, 5):
        state.update(txn(f"h{i}", ts(i), merchant="M1", amount=100))
    event = txn("current", ts(6), merchant="M1", amount=120)

    items = ContextInvestigator().collect(request(event), context(event, state))

    assert not any(
        i.evidence_type == EvidenceType.ACCOUNT_CONTEXT
        and "novel" in i.summary
        for i in items
    )
