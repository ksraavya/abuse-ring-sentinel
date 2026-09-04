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

from graph.temporal_replay import TemporalReplay, TEMPORAL_MODEL_FEATURE_COLUMNS
from models.temporal import FEATURE_COLUMNS, EXPECTED_FEATURE_COUNT, load_artifact
from streaming.config import KafkaConfig

MODEL_NAME = "temporal"
DEFAULT_GROUP_ID = "risk-manager-eval-temporal"
TRANSACTION_EVENT_TYPE = "transaction"
EXPECTED_PARTITIONS = 1


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


def _verify_frozen_artifact(artifact_dir: Path) -> tuple[Booster, dict[str, Any]]:
    model_path = artifact_dir / "model.lgbm"
    metadata_path = artifact_dir / "metadata.json"
    manifest_path = artifact_dir / "freeze_manifest.json"
    for path in (model_path, metadata_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Frozen Temporal artifact is incomplete; missing {path}")

    model, metadata = load_artifact(artifact_dir)
    if model.num_feature() != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Temporal model has {model.num_feature()} features; expected {EXPECTED_FEATURE_COUNT}"
        )
    if tuple(metadata.get("feature_list", ())) != FEATURE_COLUMNS:
        raise ValueError("Temporal artifact feature list does not match the locked 26-feature contract")
    if tuple(TEMPORAL_MODEL_FEATURE_COLUMNS) != FEATURE_COLUMNS:
        raise ValueError("Temporal replay and model feature contracts disagree")

    threshold = metadata.get("threshold", {}).get("value")
    if not isinstance(threshold, (int, float)) or not 0.01 <= float(threshold) <= 0.99:
        raise ValueError(f"Frozen Temporal threshold is invalid: {threshold!r}")
    if metadata.get("training", {}).get("world") != "world_a":
        raise ValueError("Temporal artifact was not trained on World A")

    contract = metadata.get("feature_contract", {})
    if contract.get("infrastructure_raw_columns_in_model") != []:
        raise ValueError("Temporal classifier unexpectedly contains raw infrastructure columns")
    if contract.get("total") != EXPECTED_FEATURE_COUNT:
        raise ValueError("Temporal feature contract total is not 26")

    boundary = metadata.get("information_boundary", {})
    if boundary.get("future_state") is not False:
        raise ValueError("Temporal artifact does not assert future_state=false")
    if boundary.get("ground_truth_visible_to_detector") is not False:
        raise ValueError("Temporal artifact does not assert GT is hidden")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    current_sha = _sha256(model_path)
    if manifest.get("model_sha256") != current_sha:
        raise ValueError("Temporal model SHA does not match the World A freeze manifest")
    if float(manifest.get("threshold")) != float(threshold):
        raise ValueError("Temporal threshold does not match the World A freeze manifest")
    if manifest.get("feature_list") != list(FEATURE_COLUMNS):
        raise ValueError("Freeze manifest feature list does not match the locked contract")

    return model, metadata


def _consumer_at_beginning(cfg: KafkaConfig, topic: str, group_id: str):
    from confluent_kafka import Consumer, TopicPartition

    consumer = Consumer(
        {
            "bootstrap.servers": cfg.bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    metadata = consumer.list_topics(topic=topic, timeout=10)
    if topic not in metadata.topics:
        consumer.close()
        raise RuntimeError(f"Kafka topic {topic!r} does not exist")
    partitions = metadata.topics[topic].partitions
    if set(partitions) != {0}:
        consumer.close()
        raise RuntimeError(
            f"Expected exactly one partition for {topic}, found {sorted(partitions)}"
        )
    partition = TopicPartition(topic, 0)
    consumer.assign([TopicPartition(topic, 0, 0)])
    low, high = consumer.get_watermark_offsets(partition, timeout=10)
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
    return str(event_id), bool(raw["is_fraud"]), raw.get("ring_id")


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
    """Evaluate the frozen causal Temporal detector on retained World B via Kafka.

    The Kafka stream is the sole model input. TemporalReplay enforces the
    causal boundary for every transaction: pre-T state is read, 26 features
    are extracted, the model is scored, and only then is current state updated.
    Ground truth is joined afterward for evaluation only and is never passed to
    the model or replay state.
    """
    for path in (events_path, ground_truth_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(path)

    model, metadata = _verify_frozen_artifact(artifact_dir)
    threshold = float(metadata["threshold"]["value"])

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
    truth_line_number = 0
    previous_event_key: tuple[datetime, str] | None = None

    replay = TemporalReplay()

    cfg = KafkaConfig()
    topic = cfg.topic("world_b")
    consumer, high_watermark = _consumer_at_beginning(cfg, topic, group_id)
    started = time.perf_counter()
    last_message_time = started

    with ground_truth_path.open("r", encoding="utf-8") as gt_handle:
        try:
            while transactions < expected_transactions:
                msg = consumer.poll(1.0)
                if msg is None:
                    from confluent_kafka import TopicPartition
                    position = consumer.position([TopicPartition(topic, 0)])[0].offset
                    if position >= high_watermark:
                        break
                    if time.perf_counter() - last_message_time > 15:
                        raise TimeoutError(
                            "Kafka Temporal evaluation stalled before the expected transaction count"
                        )
                    continue

                last_message_time = time.perf_counter()
                if msg.error():
                    from confluent_kafka import KafkaException
                    raise KafkaException(msg.error())

                raw = json.loads(msg.value().decode("utf-8"))
                if not isinstance(raw, dict):
                    raise ValueError("Kafka event payload must decode to a JSON object")
                event_timestamp = _event_timestamp(raw)
                event_id = str(raw.get("event_id", ""))
                if not event_id:
                    raise ValueError("Kafka event has no event_id")
                event_key = (event_timestamp, event_id)
                if previous_event_key is not None and event_key < previous_event_key:
                    raise ValueError("Kafka World B stream is not chronological")
                previous_event_key = event_key

                if raw.get("event_type") in {"account_created", "account_updated"}:
                    replay.process_event(raw)
                    account_events += 1
                    continue
                if raw.get("event_type") != TRANSACTION_EVENT_TYPE:
                    raise ValueError(f"Unknown event_type {raw.get('event_type')!r}")

                def score_current_transaction(row: tuple[float, ...]) -> None:
                    nonlocal tp, fp, tn, fn
                    nonlocal transactions, truth_line_number
                    nonlocal fraud_exposure, exposure_prevented

                    if len(row) != EXPECTED_FEATURE_COUNT:
                        raise RuntimeError(
                            f"Temporal replay produced {len(row)} features; expected {EXPECTED_FEATURE_COUNT}"
                        )

                    # The callback executes BEFORE TemporalReplay commits the
                    # current transaction to history/behavioral state.
                    probability = float(model.predict(np.asarray([row], dtype=np.float32))[0])
                    predicted = probability >= threshold

                    truth_line_number += 1
                    truth_id, is_fraud, ring_id = _read_truth_row(gt_handle, truth_line_number)
                    if truth_id != event_id:
                        raise ValueError(
                            "Kafka transaction order does not match World B ground-truth order: "
                            f"expected {truth_id}, got {event_id}"
                        )

                    probabilities[transactions] = probability
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
                            fraud_rings.add(str(ring_id))
                            if predicted:
                                detected_fraud_rings.add(str(ring_id))
                            prior = first_fraud.get(str(ring_id))
                            if prior is None or event_timestamp < prior:
                                first_fraud[str(ring_id)] = event_timestamp
                    if is_fraud and predicted:
                        exposure_prevented += amount

                    if predicted:
                        for candidate_ring in account_to_rings.get(str(raw["account_id"]), set()):
                            prior = first_ring_member_alert.get(candidate_ring)
                            if prior is None or event_timestamp < prior:
                                first_ring_member_alert[candidate_ring] = event_timestamp

                    transactions += 1

                replay.process_event(raw, score_callback=score_current_transaction)

                if transactions and transactions % 100_000 == 0:
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
    if lead_times:
        mid = len(lead_times) // 2
        median_lead = lead_times[mid] if len(lead_times) % 2 else (lead_times[mid - 1] + lead_times[mid]) / 2.0
        mean_lead = float(np.mean(lead_times))
        min_lead = min(lead_times)
        max_lead = max(lead_times)
    else:
        median_lead = mean_lead = min_lead = max_lead = None

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
            "lead_time_days_mean": mean_lead,
            "lead_time_days_median": median_lead,
            "lead_time_days_min": min_lead,
            "lead_time_days_max": max_lead,
        },
        "exposure": {
            "fraud_exposure": fraud_exposure,
            "exposure_prevented": exposure_prevented,
            "exposure_prevented_pct": exposure_prevented / fraud_exposure if fraud_exposure else 0.0,
        },
        "stream": {
            "topic": topic,
            "partitions": EXPECTED_PARTITIONS,
            "group_id": group_id,
            "high_watermark_at_start": high_watermark,
            "account_events_consumed": account_events,
            "ground_truth_published_to_kafka": False,
        },
        "state": {
            "infrastructure_state": True,
            "behavioral_graph": True,
            "transaction_history": True,
            "transaction_mutates_infrastructure_state": False,
            "read_state_before_score": True,
            "score_before_current_transaction_state_update": True,
        },
        "frozen_artifact": {
            "artifact_dir": str(artifact_dir),
            "model_sha256": _sha256(artifact_dir / "model.lgbm"),
            "threshold": threshold,
            "feature_count": len(FEATURE_COLUMNS),
            "feature_list": list(FEATURE_COLUMNS),
        },
        "evaluation_contract": {
            "world_b_source_events": str(events_path),
            "ground_truth_source": str(ground_truth_path),
            "manifest_source": str(manifest_path),
            "world_b_kafka_is_evaluation_input": True,
            "threshold_tuned_on_world_b": False,
            "model_retrained_on_world_b": False,
            "feature_count": EXPECTED_FEATURE_COUNT,
            "infrastructure_raw_columns_in_model": [],
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen Temporal on held-out World B via Kafka.")
    parser.add_argument("--events", default="data/generated/world_b/events.jsonl")
    parser.add_argument("--ground-truth", default="data/generated/world_b/ground_truth.jsonl")
    parser.add_argument("--manifest", default="data/generated/world_b/manifest.json")
    parser.add_argument("--artifact-dir", default="artifacts/temporal")
    parser.add_argument("--output-dir", default="artifacts/evaluation/world_b/temporal")
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
    print("Temporal World B evaluation complete.")
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
    print(f"  Rings:        {metrics['ring_detection']['detected_rings']}/{metrics['ring_detection']['fraud_bearing_rings']}")
    print(f"  Pre-abuse:    {metrics['pre_abuse']['rings_detected_pre_abuse']}/{metrics['pre_abuse']['rings_with_abuse']}")
    print(f"  Exposure prevented: ₹{metrics['exposure']['exposure_prevented']:,.2f}")
    print(f"  Results:      {args.output_dir}")


if __name__ == "__main__":
    main()
