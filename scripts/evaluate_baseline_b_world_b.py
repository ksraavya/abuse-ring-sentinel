from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from lightgbm import Booster
from sklearn.metrics import average_precision_score, roc_auc_score

from features.baseline_b import FEATURE_COLUMNS, event_to_feature_row
from graph.infrastructure_state import InfrastructureState
from models.baseline_b import load_artifact
from confluent_kafka import TopicPartition
from streaming.config import KafkaConfig

MODEL_NAME = "baseline_b"
DEFAULT_GROUP_ID = "risk-manager-eval-baseline-b"
TRANSACTION_EVENT_TYPE = "transaction"
ACCOUNT_EVENT_TYPES = {"account_created", "account_updated"}
BATCH_SIZE = 10_000


def _count_ground_truth(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not raw.get("event_id"):
                raise ValueError(f"Ground truth line {line_number} has no event_id")
            count += 1
    return count


def _load_ring_members(manifest_path: Path) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rings: dict[str, set[str]] = {}
    account_to_rings: dict[str, set[str]] = {}
    for ring in manifest.get("rings", []):
        ring_id = ring.get("ring_id")
        members = ring.get("account_ids")
        if not ring_id or not isinstance(members, list):
            raise ValueError("World B manifest contains an invalid ring entry")
        if ring_id in rings:
            raise ValueError(f"Duplicate ring_id in manifest: {ring_id}")
        member_set = set(members)
        rings[ring_id] = member_set
        for account_id in member_set:
            account_to_rings.setdefault(account_id, set()).add(ring_id)
    return rings, account_to_rings


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _consumer_at_beginning(cfg: KafkaConfig, topic: str, group_id: str):
    from confluent_kafka import Consumer, TopicPartition

    consumer = Consumer({
        "bootstrap.servers": cfg.bootstrap_servers,
        "group.id": group_id,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    })
    metadata = consumer.list_topics(topic=topic, timeout=10)
    partitions = metadata.topics[topic].partitions
    if set(partitions) != {0}:
        consumer.close()
        raise RuntimeError(
            f"Expected exactly one partition for {topic}, found {sorted(partitions)}"
        )
    consumer.assign([TopicPartition(topic, 0, 0)])
    low, high = consumer.get_watermark_offsets(TopicPartition(topic, 0), timeout=10)
    if low != 0:
        consumer.close()
        raise RuntimeError(f"Evaluation topic must start at offset 0; low watermark is {low}")
    return consumer, high


def _read_truth_row(handle, line_number: int) -> tuple[str, bool, str | None]:
    line = handle.readline()
    if not line:
        raise RuntimeError("World B ground truth ended before Kafka transactions")
    raw = json.loads(line)
    event_id = raw.get("event_id")
    if not event_id:
        raise ValueError(f"Ground truth line {line_number} has no event_id")
    return event_id, bool(raw["is_fraud"]), raw.get("ring_id")


def _event_timestamp(raw: dict[str, Any]) -> datetime:
    timestamp = datetime.fromisoformat(str(raw["timestamp"]).replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("World B event timestamp must be timezone-aware")
    return timestamp.astimezone(timezone.utc)


def evaluate_world_b(
    *,
    events_path: Path,
    ground_truth_path: Path,
    manifest_path: Path,
    artifact_dir: Path,
    output_dir: Path,
    group_id: str,
) -> dict[str, Any]:
    """Evaluate frozen Baseline B on retained World B via Kafka.

    InfrastructureState is updated only by account lifecycle events. For every
    transaction, features are extracted from state strictly before the current
    transaction and the transaction itself does not mutate infrastructure state.
    Ground truth is read from a separate file after model scoring and never
    enters Kafka or model features.
    """
    for path in (events_path, ground_truth_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(path)

    model, metadata = load_artifact(artifact_dir)
    feature_list = tuple(metadata.get("feature_list", ()))
    if feature_list != FEATURE_COLUMNS:
        raise ValueError("Baseline B artifact feature list does not match the locked 17-feature contract")
    threshold = float(metadata["threshold"]["value"])
    if not 0.01 <= threshold <= 0.99:
        raise ValueError(f"Frozen Baseline B threshold is outside the locked grid: {threshold}")

    expected_transactions = _count_ground_truth(ground_truth_path)
    if expected_transactions == 0:
        raise ValueError("World B ground truth is empty")
    _, account_to_rings = _load_ring_members(manifest_path)

    probabilities = np.empty(expected_transactions, dtype=np.float32)
    labels = np.empty(expected_transactions, dtype=np.int8)

    tp = fp = tn = fn = 0
    fraud_exposure = 0.0
    exposure_prevented = 0.0
    fraud_rings: set[str] = set()
    detected_fraud_rings: set[str] = set()
    first_fraud: dict[str, datetime] = {}
    first_ring_member_alert: dict[str, datetime] = {}
    transactions = 0
    account_events = 0
    previous_event_key: tuple[datetime, str] | None = None

    state = InfrastructureState()
    cfg = KafkaConfig()
    topic = cfg.topic("world_b")
    consumer, high_watermark = _consumer_at_beginning(cfg, topic, group_id)
    started = time.perf_counter()
    last_message_time = started

    with ground_truth_path.open("r", encoding="utf-8") as gt_handle:
        truth_line_number = 0
        try:
            while transactions < expected_transactions:
                batch_raws: list[dict[str, Any]] = []

                while len(batch_raws) < BATCH_SIZE:
                    msg = consumer.poll(1.0)
                    if msg is None:
                        position = consumer.position([TopicPartition(topic, 0)])[0].offset
                        if position >= high_watermark:
                            break
                        continue
                    last_message_time = time.perf_counter()
                    if msg.error():
                        from confluent_kafka import KafkaException
                        raise KafkaException(msg.error())
                    raw = json.loads(msg.value().decode("utf-8"))
                    if not isinstance(raw, dict):
                        raise ValueError("Kafka event payload must decode to a JSON object")

                    timestamp = _event_timestamp(raw)
                    event_id = str(raw.get("event_id", ""))
                    if not event_id:
                        raise ValueError("Kafka event has no event_id")
                    event_key = (timestamp, event_id)
                    if previous_event_key is not None and event_key < previous_event_key:
                        raise ValueError("Kafka World B stream is not chronological")
                    previous_event_key = event_key

                    event_type = raw.get("event_type")
                    if event_type == "account_created":
                        state.add_or_update(str(raw["account_id"]), str(raw["device_id"]), str(raw["ip_prefix"]))
                        account_events += 1
                        continue
                    if event_type == "account_updated":
                        state.add_or_update(str(raw["account_id"]), str(raw["new_device_id"]), str(raw["new_ip_prefix"]))
                        account_events += 1
                        continue
                    if event_type != TRANSACTION_EVENT_TYPE:
                        raise ValueError(f"Unknown event_type {event_type!r}")
                    batch_raws.append(raw)

                if not batch_raws:
                    if consumer.position([TopicPartition(topic, 0)])[0].offset >= high_watermark:
                        break
                    if time.perf_counter() - last_message_time > 15:
                        raise TimeoutError("Kafka evaluation stalled before the expected transaction count")
                    continue

                # CRITICAL: all features in this batch are extracted from the
                # state that existed immediately before each transaction. No
                # transaction in the batch mutates InfrastructureState.
                x = np.asarray(
                    [event_to_feature_row(raw, state) for raw in batch_raws],
                    dtype=np.float32,
                )
                batch_probabilities = np.asarray(model.predict(x), dtype=np.float32)

                for raw, probability in zip(batch_raws, batch_probabilities, strict=True):
                    truth_line_number += 1
                    truth_id, is_fraud, ring_id = _read_truth_row(gt_handle, truth_line_number)
                    event_id = str(raw["event_id"])
                    if event_id != truth_id:
                        raise ValueError(
                            "Kafka transaction order does not match World B ground-truth order: "
                            f"expected {truth_id}, got {event_id}"
                        )

                    timestamp = _event_timestamp(raw)
                    predicted = float(probability) >= threshold
                    probabilities[transactions] = float(probability)
                    labels[transactions] = int(is_fraud)

                    if predicted and is_fraud:
                        tp += 1
                    elif predicted and not is_fraud:
                        fp += 1
                    elif not predicted and not is_fraud:
                        tn += 1
                    else:
                        fn += 1

                    amount = float(raw["amount"])
                    if is_fraud:
                        fraud_exposure += amount
                        if ring_id is not None:
                            fraud_rings.add(ring_id)
                            if predicted:
                                detected_fraud_rings.add(ring_id)
                            prior = first_fraud.get(ring_id)
                            if prior is None or timestamp < prior:
                                first_fraud[ring_id] = timestamp
                    if is_fraud and predicted:
                        exposure_prevented += amount

                    if predicted:
                        for candidate_ring in account_to_rings.get(str(raw["account_id"]), set()):
                            prior = first_ring_member_alert.get(candidate_ring)
                            if prior is None or timestamp < prior:
                                first_ring_member_alert[candidate_ring] = timestamp

                    transactions += 1

                if transactions % 100_000 < len(batch_raws) and transactions >= 100_000:
                    print(f"  Evaluated {transactions:,}/{expected_transactions:,} transactions")
        finally:
            consumer.close()

        extra = next((line for line in gt_handle if line.strip()), None)
        if extra is not None:
            raise RuntimeError("World B ground truth contains more records than Kafka transactions")

    if transactions != expected_transactions:
        raise RuntimeError(
            f"Transaction count mismatch: Kafka={transactions}, ground_truth={expected_transactions}"
        )

    y = labels[:transactions]
    p = probabilities[:transactions]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0

    lead_times = [
        (first_fraud[ring_id] - first_ring_member_alert[ring_id]).total_seconds() / 86400.0
        for ring_id in first_fraud
        if ring_id in first_ring_member_alert
        and first_ring_member_alert[ring_id] < first_fraud[ring_id]
    ]
    lead_times.sort()
    median_lead = None
    if lead_times:
        mid = len(lead_times) // 2
        median_lead = lead_times[mid] if len(lead_times) % 2 else (lead_times[mid - 1] + lead_times[mid]) / 2.0

    pre_abuse_detected = sum(
        1 for ring_id in first_fraud
        if ring_id in first_ring_member_alert
        and first_ring_member_alert[ring_id] < first_fraud[ring_id]
    )

    metrics: dict[str, Any] = {
        "model": MODEL_NAME,
        "world": "world_b",
        "threshold": threshold,
        "transactions": transactions,
        "fraud_transactions": int(y.sum()),
        "fraud_rate": float(y.mean()),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "fnr": fnr,
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None,
        "false_positive_cost": 500.0,
        "false_negative_cost": 5000.0,
        "economic_cost": 500.0 * fp + 5000.0 * fn,
        "ring_detection": {
            "fraud_bearing_rings": len(fraud_rings),
            "detected_rings": len(detected_fraud_rings),
            "ring_detection_recall": len(detected_fraud_rings) / len(fraud_rings) if fraud_rings else 0.0,
        },
        "pre_abuse": {
            "rings_with_abuse": len(first_fraud),
            "rings_detected_pre_abuse": pre_abuse_detected,
            "pre_abuse_detection_recall": pre_abuse_detected / len(first_fraud) if first_fraud else 0.0,
            "lead_time_days_mean": float(np.mean(lead_times)) if lead_times else None,
            "lead_time_days_median": median_lead,
            "lead_time_days_min": min(lead_times) if lead_times else None,
            "lead_time_days_max": max(lead_times) if lead_times else None,
        },
        "exposure": {
            "fraud_exposure": fraud_exposure,
            "exposure_prevented": exposure_prevented,
            "exposure_prevented_pct": exposure_prevented / fraud_exposure if fraud_exposure else 0.0,
        },
        "stream": {
            "topic": topic,
            "partitions": 1,
            "group_id": group_id,
            "high_watermark_at_start": high_watermark,
            "account_events_consumed": account_events,
            "ground_truth_published_to_kafka": False,
        },
        "state": {
            "infrastructure_state": True,
            "behavioral_graph": False,
            "transaction_history": False,
            "transaction_mutates_infrastructure_state": False,
            "read_state_before_score": True,
        },
        "frozen_artifact": {
            "artifact_dir": str(artifact_dir),
            "model_sha256": _sha256(artifact_dir / "model.lgbm"),
            "threshold": threshold,
            "feature_count": len(feature_list),
            "feature_list": list(feature_list),
        },
        "evaluation_contract": {
            "world_b_source_events": str(events_path),
            "ground_truth_source": str(ground_truth_path),
            "manifest_source": str(manifest_path),
            "world_b_kafka_is_evaluation_input": True,
            "threshold_tuned_on_world_b": False,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen Baseline B on held-out World B via Kafka.")
    parser.add_argument("--events", default="data/generated/world_b/events.jsonl")
    parser.add_argument("--ground-truth", default="data/generated/world_b/ground_truth.jsonl")
    parser.add_argument("--manifest", default="data/generated/world_b/manifest.json")
    parser.add_argument("--artifact-dir", default="artifacts/baseline_b")
    parser.add_argument("--output-dir", default="artifacts/evaluation/world_b/baseline_b")
    parser.add_argument("--group-id", default=DEFAULT_GROUP_ID)
    args = parser.parse_args()

    metrics = evaluate_world_b(
        events_path=Path(args.events),
        ground_truth_path=Path(args.ground_truth),
        manifest_path=Path(args.manifest),
        artifact_dir=Path(args.artifact_dir),
        output_dir=Path(args.output_dir),
        group_id=args.group_id,
    )
    print("Baseline B World B evaluation complete.")
    print(f"  Transactions: {metrics['transactions']:,}")
    print(f"  Threshold:    {metrics['threshold']:.2f}")
    print(f"  Precision:    {metrics['precision']:.6f}")
    print(f"  Recall:       {metrics['recall']:.6f}")
    print(f"  FPR:          {metrics['fpr']:.6f}")
    print(f"  PR-AUC:       {metrics['pr_auc']:.6f}")
    print(f"  ROC-AUC:      {metrics['roc_auc']}")
    print(f"  FP:           {metrics['false_positives']:,}")
    print(f"  FN:           {metrics['false_negatives']:,}")
    print(f"  Econ. cost:   ₹{metrics['economic_cost']:,.0f}")
    print(f"  Results:      {args.output_dir}")


if __name__ == "__main__":
    main()
