from datetime import datetime, timezone

import pytest

from events.schema import TransactionChannel, TransactionEvent, WorldId
from graph.behavioral_state import BehavioralState
from graph.infrastructure_state import InfrastructureState
from verifier.contracts import EvidenceContext, EvidenceType, VerificationRequest
from verifier.infrastructure_investigator import InfrastructureInvestigator, InfrastructureInvestigatorConfig

T0 = datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc)


def tx() -> TransactionEvent:
    return TransactionEvent(
        event_id="alert-a1",
        world_id=WorldId.WORLD_B,
        timestamp=T0,
        account_id="a1",
        merchant_id="m1",
        amount=100.0,
        channel=TransactionChannel.UPI,
        device_id="d1",
        ip_prefix="ip1",
    )


def request() -> VerificationRequest:
    return VerificationRequest(
        alert_event=tx(),
        detector_probability=0.8,
        detector_threshold=0.01,
        detector_model="temporal@world-a-freeze",
        alerted_at=T0,
    )


def context(infra: InfrastructureState, behavioral: BehavioralState | None = None) -> EvidenceContext:
    state = {"infrastructure_state": infra}
    if behavioral is not None:
        state["behavioral_state"] = behavioral
    return EvidenceContext(as_of=T0, state=state)


def test_shared_infrastructure_emits_evidence():
    infra = InfrastructureState()
    infra.add_or_update("a1", "d1", "ip1")
    infra.add_or_update("a2", "d1", "ip2")
    infra.add_or_update("a3", "d1", "ip3")

    evidence = InfrastructureInvestigator().collect(request(), context(infra))
    assert len(evidence) == 1
    item = evidence[0]
    assert item.evidence_type is EvidenceType.INFRASTRUCTURE_SHARING
    assert item.metric_value == 2
    assert item.observed_at == T0
    assert item.source_event_ids == ("alert-a1",)
    assert set(item.subject_account_ids) == {"a1", "a2", "a3"}


def test_behavioral_overlap_strengthens_evidence():
    infra = InfrastructureState()
    infra.add_or_update("a1", "d1", "ip1")
    infra.add_or_update("a2", "d1", "ip2")
    infra.add_or_update("a3", "d1", "ip3")

    behavioral = BehavioralState()
    behavioral.update("a1", "a2", T0.isoformat().replace("+00:00", "Z"), 50.0)

    evidence = InfrastructureInvestigator().collect(request(), context(infra, behavioral))[0]
    assert evidence.details["device_behavioral_overlap"] == ("a2",)
    assert evidence.strength.value == "moderate"


def test_single_shared_account_is_below_default_threshold():
    infra = InfrastructureState()
    infra.add_or_update("a1", "d1", "ip1")
    infra.add_or_update("a2", "d1", "ip2")
    assert InfrastructureInvestigator().collect(request(), context(infra)) == []


def test_no_sharing_emits_no_evidence():
    infra = InfrastructureState()
    infra.add_or_update("a1", "d1", "ip1")
    assert InfrastructureInvestigator().collect(request(), context(infra)) == []


def test_future_context_is_rejected():
    infra = InfrastructureState()
    infra.add_or_update("a1", "d1", "ip1")
    infra.add_or_update("a2", "d1", "ip2")
    future = EvidenceContext(
        as_of=T0.replace(hour=13),
        state={"infrastructure_state": infra},
    )
    with pytest.raises(ValueError, match="after verifier decision_time"):
        InfrastructureInvestigator().collect(request(), future)


def test_churn_evidence_is_not_emitted_without_history():
    infra = InfrastructureState()
    infra.add_or_update("a1", "d1", "ip1")
    infra.add_or_update("a2", "d1", "ip2")
    infra.add_or_update("a3", "d1", "ip3")
    evidence = InfrastructureInvestigator().collect(request(), context(infra))[0]
    assert evidence.evidence_type is EvidenceType.INFRASTRUCTURE_SHARING
    assert evidence.evidence_type is not EvidenceType.INFRASTRUCTURE_CHURN


def test_subject_list_is_bounded_deterministically():
    infra = InfrastructureState()
    for i in range(8):
        infra.add_or_update(f"a{i}", "d1", f"ip{i}")
    config = InfrastructureInvestigatorConfig(max_accounts_in_evidence=3)
    evidence = InfrastructureInvestigator(config).collect(request(), context(infra))[0]
    assert evidence.subject_account_ids == tuple(sorted(evidence.subject_account_ids))
    assert len(evidence.subject_account_ids) == 3
