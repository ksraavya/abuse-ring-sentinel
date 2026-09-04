from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sklearn.metrics import average_precision_score, roc_auc_score

from evaluation.schema import PredictionRecord


@dataclass(frozen=True)
class EvaluationCosts:
    """Locked transaction decision costs used throughout model evaluation."""

    false_positive: float = 500.0
    false_negative: float = 5000.0

    def __post_init__(self) -> None:
        if self.false_positive < 0 or self.false_negative < 0:
            raise ValueError("evaluation costs must be non-negative")


def _as_records(records: Iterable[PredictionRecord]) -> list[PredictionRecord]:
    result = list(records)
    ids = [row.event_id for row in result]
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation predictions contain duplicate event_id values")
    if not result:
        raise ValueError("evaluation predictions cannot be empty")
    return result


def transaction_metrics(
    records: Iterable[PredictionRecord],
    costs: EvaluationCosts = EvaluationCosts(),
) -> dict[str, Any]:
    """Compute the locked transaction-level evaluation rubric."""
    rows = _as_records(records)
    tp = sum(r.predicted_fraud and r.is_fraud for r in rows)
    fp = sum(r.predicted_fraud and not r.is_fraud for r in rows)
    tn = sum((not r.predicted_fraud) and (not r.is_fraud) for r in rows)
    fn = sum((not r.predicted_fraud) and r.is_fraud for r in rows)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0

    y_true = [int(r.is_fraud) for r in rows]
    probabilities = [r.probability for r in rows]

    return {
        "transactions": len(rows),
        "fraud_transactions": sum(y_true),
        "fraud_rate": sum(y_true) / len(rows),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "fnr": fnr,
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if len(set(y_true)) == 2 else None,
        "false_positive_cost": costs.false_positive,
        "false_negative_cost": costs.false_negative,
        "economic_cost": costs.false_positive * fp + costs.false_negative * fn,
    }


def ring_detection_metrics(records: Iterable[PredictionRecord]) -> dict[str, Any]:
    """Measure whether each fraud-bearing ring is detected at transaction level.

    A ring is counted as detected when at least one of its fraudulent
    transactions receives a positive prediction. This is deliberately a
    post-hoc evaluation metric and never a model feature.
    """
    rows = _as_records(records)
    fraud_by_ring: dict[str, bool] = {}
    detected: set[str] = set()
    for row in rows:
        if row.is_fraud and row.ring_id is not None:
            fraud_by_ring.setdefault(row.ring_id, False)
            if row.predicted_fraud:
                detected.add(row.ring_id)
                fraud_by_ring[row.ring_id] = True

    total = len(fraud_by_ring)
    detected_count = len(detected)
    return {
        "fraud_bearing_rings": total,
        "detected_rings": detected_count,
        "undetected_rings": total - detected_count,
        "ring_detection_recall": detected_count / total if total else 0.0,
    }


def pre_abuse_metrics(
    records: Iterable[PredictionRecord],
    ring_members: Mapping[str, set[str]],
) -> dict[str, Any]:
    """Measure alerts on ring members strictly before each ring's first fraud.

    Ring membership comes from the generator manifest on the evaluation side;
    it is not supplied to the detector. An alert qualifies as pre-abuse only if
    it is positive, occurs before the ring's first fraudulent transaction, and
    belongs to an account in that ring.
    """
    rows = _as_records(records)
    first_fraud: dict[str, datetime] = {}
    for row in rows:
        if row.is_fraud and row.ring_id is not None:
            ts = row.utc_timestamp
            prior = first_fraud.get(row.ring_id)
            if prior is None or ts < prior:
                first_fraud[row.ring_id] = ts

    account_to_rings: dict[str, set[str]] = {}
    for ring_id, members in ring_members.items():
        for account_id in members:
            account_to_rings.setdefault(account_id, set()).add(ring_id)

    first_alert: dict[str, datetime] = {}
    for row in rows:
        if not row.predicted_fraud:
            continue
        for ring_id in account_to_rings.get(row.account_id, set()):
            abuse_start = first_fraud.get(ring_id)
            if abuse_start is not None and row.utc_timestamp < abuse_start:
                prior = first_alert.get(ring_id)
                if prior is None or row.utc_timestamp < prior:
                    first_alert[ring_id] = row.utc_timestamp

    lead_times_days = [
        (first_fraud[ring_id] - first_alert[ring_id]).total_seconds() / 86400.0
        for ring_id in first_alert
    ]
    detected = len(first_alert)
    total = len(first_fraud)
    return {
        "rings_with_abuse": total,
        "rings_detected_pre_abuse": detected,
        "pre_abuse_detection_recall": detected / total if total else 0.0,
        "lead_time_days_mean": sum(lead_times_days) / len(lead_times_days) if lead_times_days else None,
        "lead_time_days_median": _median(lead_times_days),
        "lead_time_days_min": min(lead_times_days) if lead_times_days else None,
        "lead_time_days_max": max(lead_times_days) if lead_times_days else None,
    }


def exposure_metrics(records: Iterable[PredictionRecord]) -> dict[str, float]:
    """Estimate counterfactual fraud exposure blocked by positive decisions.

    The explicit intervention assumption is: a positive decision blocks the
    current transaction. Therefore only fraudulent transactions predicted
    positive count as prevented exposure.
    """
    rows = _as_records(records)
    total = sum(r.amount for r in rows if r.is_fraud)
    prevented = sum(r.amount for r in rows if r.is_fraud and r.predicted_fraud)
    return {
        "fraud_exposure": total,
        "exposure_prevented": prevented,
        "exposure_prevented_pct": prevented / total if total else 0.0,
    }


def build_world_b_rubric(
    records: Iterable[PredictionRecord],
    ring_members: Mapping[str, set[str]],
    costs: EvaluationCosts = EvaluationCosts(),
) -> dict[str, Any]:
    rows = _as_records(records)
    return {
        "transaction": transaction_metrics(rows, costs),
        "ring_detection": ring_detection_metrics(rows),
        "pre_abuse": pre_abuse_metrics(rows, ring_members),
        "exposure": exposure_metrics(rows),
    }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0
