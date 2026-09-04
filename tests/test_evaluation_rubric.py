from __future__ import annotations

from datetime import datetime, timezone

import pytest

from evaluation.rubric import (
    EvaluationCosts,
    build_world_b_rubric,
    exposure_metrics,
    pre_abuse_metrics,
    ring_detection_metrics,
    transaction_metrics,
)
from evaluation.schema import PredictionRecord


def row(
    event_id: str,
    timestamp: str,
    account_id: str,
    *,
    amount: float = 100.0,
    probability: float = 0.9,
    predicted: bool = True,
    fraud: bool = False,
    ring_id: str | None = None,
) -> PredictionRecord:
    return PredictionRecord(
        model="test",
        event_id=event_id,
        timestamp=datetime.fromisoformat(timestamp.replace("Z", "+00:00")),
        account_id=account_id,
        amount=amount,
        probability=probability,
        predicted_fraud=predicted,
        is_fraud=fraud,
        ring_id=ring_id,
    )


def test_transaction_metrics_and_cost_are_locked() -> None:
    records = [
        row("1", "2026-01-01T00:00:00Z", "a", predicted=True, fraud=True),
        row("2", "2026-01-01T00:01:00Z", "b", predicted=True, fraud=False),
        row("3", "2026-01-01T00:02:00Z", "c", predicted=False, fraud=True),
        row("4", "2026-01-01T00:03:00Z", "d", predicted=False, fraud=False),
    ]
    metrics = transaction_metrics(records)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["fpr"] == pytest.approx(0.5)
    assert metrics["economic_cost"] == 5500.0


def test_ring_detection_requires_a_positive_fraud_transaction() -> None:
    records = [
        row("1", "2026-01-01T00:00:00Z", "a", predicted=True, fraud=False, ring_id=None),
        row("2", "2026-01-01T00:01:00Z", "a", predicted=False, fraud=True, ring_id="r1"),
        row("3", "2026-01-01T00:02:00Z", "b", predicted=True, fraud=True, ring_id="r2"),
    ]
    metrics = ring_detection_metrics(records)
    assert metrics["fraud_bearing_rings"] == 2
    assert metrics["detected_rings"] == 1
    assert metrics["ring_detection_recall"] == pytest.approx(0.5)


def test_pre_abuse_detection_is_strictly_before_first_fraud() -> None:
    records = [
        row("1", "2026-01-01T00:00:00Z", "a", predicted=True, fraud=False),
        row("2", "2026-01-01T00:10:00Z", "b", predicted=True, fraud=False),
        row("3", "2026-01-01T01:00:00Z", "a", predicted=True, fraud=True, ring_id="r1"),
        row("4", "2026-01-01T02:00:00Z", "c", predicted=True, fraud=True, ring_id="r2"),
    ]
    metrics = pre_abuse_metrics(records, {"r1": {"a", "b"}, "r2": {"c"}})
    assert metrics["rings_with_abuse"] == 2
    assert metrics["rings_detected_pre_abuse"] == 1
    assert metrics["pre_abuse_detection_recall"] == pytest.approx(0.5)
    assert metrics["lead_time_days_mean"] == pytest.approx(1 / 24)


def test_exposure_uses_only_blocked_fraud() -> None:
    records = [
        row("1", "2026-01-01T00:00:00Z", "a", amount=1000, predicted=True, fraud=True),
        row("2", "2026-01-01T00:01:00Z", "b", amount=2000, predicted=False, fraud=True),
        row("3", "2026-01-01T00:02:00Z", "c", amount=5000, predicted=True, fraud=False),
    ]
    metrics = exposure_metrics(records)
    assert metrics["fraud_exposure"] == pytest.approx(3000)
    assert metrics["exposure_prevented"] == pytest.approx(1000)
    assert metrics["exposure_prevented_pct"] == pytest.approx(1 / 3)


def test_combined_rubric_uses_locked_costs() -> None:
    records = [
        row("1", "2026-01-01T00:00:00Z", "a", predicted=True, fraud=True, ring_id="r1"),
        row("2", "2026-01-01T00:01:00Z", "b", predicted=False, fraud=False),
    ]
    rubric = build_world_b_rubric(records, {"r1": {"a"}}, EvaluationCosts())
    assert rubric["transaction"]["economic_cost"] == 0.0
    assert rubric["ring_detection"]["ring_detection_recall"] == 1.0

def test_pre_abuse_alert_on_non_member_does_not_count():
    records = [
        row("1", "2026-01-01T00:00:00Z", "outsider", predicted=True, fraud=False),
        row("2", "2026-01-01T01:00:00Z", "member", predicted=True, fraud=True, ring_id="r1"),
    ]
    metrics = pre_abuse_metrics(records, {"r1": {"member"}})
    assert metrics["rings_detected_pre_abuse"] == 0
