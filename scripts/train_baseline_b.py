from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from lightgbm import LGBMClassifier

from features.baseline_b import FEATURE_COLUMNS, event_to_feature_row
from graph.infrastructure_state import InfrastructureState
from graph.neo4j_infrastructure import InfrastructureGraph
from models.baseline_a import CostConfig, choose_economic_threshold, ranking_metrics
from models.baseline_b import build_model

TRANSACTION = "transaction"
ACCOUNT_CREATED = "account_created"
ACCOUNT_UPDATED = "account_updated"
ACCOUNT_EVENTS = {ACCOUNT_CREATED, ACCOUNT_UPDATED}
CHANNELS = {"upi", "card", "wallet", "netbanking"}

NEO4J_BATCH_SIZE = 500


def load_ground_truth(path: Path) -> dict[str, bool]:
    labels: dict[str, bool] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            event_id = row.get("event_id")
            if not event_id:
                raise ValueError(f"Ground truth line {line_number} has no event_id")
            if event_id in labels:
                raise ValueError(f"Duplicate ground-truth event_id: {event_id}")
            labels[event_id] = bool(row["is_fraud"])
    return labels


def count_transactions(events_path: Path) -> int:
    count = 0
    with events_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip() and '"event_type":"transaction"' in line:
                count += 1
    return count


def replay_world_a(
    events_path: Path,
    ground_truth: dict[str, bool],
    graph: InfrastructureGraph,
    state: InfrastructureState,
    n_transactions: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    x = np.empty((n_transactions, len(FEATURE_COLUMNS)), dtype=np.float32)
    y = np.empty(n_transactions, dtype=np.int8)
    event_ids: list[str] = []
    timestamps: list[str] = []

    previous_timestamp: str | None = None
    transaction_index = 0

    # InfrastructureState is updated immediately because it is the exact
    # event-time state used for feature extraction. Neo4j persistence is
    # batched separately to avoid a network round trip for every account event.
    account_buffer: dict[str, dict[str, str]] = {}

    def flush_account_buffer() -> None:
        if not account_buffer:
            return
        graph.upsert_accounts_batch(list(account_buffer.values()))
        account_buffer.clear()

    with events_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw: dict[str, Any] = json.loads(line)
            event_type = raw.get("event_type")
            timestamp = str(raw.get("timestamp", ""))

            if not timestamp:
                raise ValueError(f"Event line {line_number} has no timestamp")
            if previous_timestamp is not None and timestamp < previous_timestamp:
                raise ValueError(
                    "events.jsonl is not chronological by timestamp; "
                    f"line {line_number} is earlier than its predecessor"
                )
            previous_timestamp = timestamp

            if event_type == ACCOUNT_CREATED:
                account_id = raw["account_id"]
                device_id = raw["device_id"]
                ip_prefix = raw["ip_prefix"]

                # Update event-time state immediately. This is what subsequent
                # transactions read; Neo4j persistence is deliberately batched.
                state.add_or_update(account_id, device_id, ip_prefix)
                account_buffer[account_id] = {
                    "account_id": account_id,
                    "device_id": device_id,
                    "ip_prefix": ip_prefix,
                }
                if len(account_buffer) >= NEO4J_BATCH_SIZE:
                    flush_account_buffer()
                continue

            if event_type == ACCOUNT_UPDATED:
                account_id = raw["account_id"]
                device_id = raw["new_device_id"]
                ip_prefix = raw["new_ip_prefix"]

                state.add_or_update(account_id, device_id, ip_prefix)
                account_buffer[account_id] = {
                    "account_id": account_id,
                    "device_id": device_id,
                    "ip_prefix": ip_prefix,
                }
                if len(account_buffer) >= NEO4J_BATCH_SIZE:
                    flush_account_buffer()
                continue

            if event_type != TRANSACTION:
                raise ValueError(f"Unknown event_type {event_type!r} at line {line_number}")
            if raw.get("channel") not in CHANNELS:
                raise ValueError(f"Unknown channel {raw.get('channel')!r} at line {line_number}")

            event_id = raw.get("event_id")
            if not event_id or event_id not in ground_truth:
                raise ValueError(f"Missing ground truth for transaction {event_id!r}")

            # CRITICAL: read state before this transaction. This transaction
            # does not update infrastructure state.
            x[transaction_index] = event_to_feature_row(raw, state)
            y[transaction_index] = 1 if ground_truth[event_id] else 0
            event_ids.append(event_id)
            timestamps.append(timestamp)
            transaction_index += 1

            if transaction_index % 100_000 == 0:
                print(f"  Replayed {transaction_index:,}/{n_transactions:,} transactions")

    # Persist any account lifecycle events that remain in the final buffer.
    flush_account_buffer()

    if transaction_index != n_transactions:
        raise RuntimeError(
            f"Expected {n_transactions:,} transactions, extracted {transaction_index:,}"
        )

    unexpected = set(ground_truth) - set(event_ids)
    if unexpected:
        raise ValueError(
            f"Ground truth contains {len(unexpected):,} IDs absent from transactions"
        )

    return x, y, event_ids, timestamps


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Baseline B from chronological World A JSONL.")
    parser.add_argument("--events", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--artifact-dir", default="artifacts/baseline_b")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp-cost", type=float, default=500.0)
    parser.add_argument("--fn-cost", type=float, default=5000.0)
    args = parser.parse_args()

    if not 0.5 <= args.train_fraction < 1.0:
        raise SystemExit("--train-fraction must be in [0.5, 1.0)")

    started = time.perf_counter()
    events_path = Path(args.events)
    gt_path = Path(args.ground_truth)
    artifact_dir = Path(args.artifact_dir)

    print("Loading ground truth...")
    ground_truth = load_ground_truth(gt_path)
    print(f"  Ground-truth records: {len(ground_truth):,}")

    print("Counting World A transactions...")
    n_transactions = count_transactions(events_path)
    print(f"  Transactions: {n_transactions:,}")

    graph = InfrastructureGraph()
    state = InfrastructureState()
    graph.verify()
    graph.initialize()
    # reset() removes graph data but preserves Neo4j schema constraints.
    graph.reset()
    graph.initialize()

    try:
        print("Chronologically replaying World A and extracting Baseline B features...")
        x, y, event_ids, timestamps = replay_world_a(
            events_path, ground_truth, graph, state, n_transactions
        )
    finally:
        graph.close()

    split = int(n_transactions * args.train_fraction)
    train_x, valid_x = x[:split], x[split:]
    train_y, valid_y = y[:split], y[split:]

    print()
    print("Baseline B dataset")
    print(f"  Transactions: {n_transactions:,}")
    print(f"  Fraud:        {int(y.sum()):,}")
    print(f"  Fraud rate:   {y.mean():.6%}")
    print(f"  Train:        {len(train_x):,}")
    print(f"  Validation:   {len(valid_x):,}")
    print(f"  Split time:   {timestamps[split]}")
    print(f"  Train fraud:  {int(train_y.sum()):,}")
    print(f"  Valid fraud:  {int(valid_y.sum()):,}")
    print(f"  Features:     {len(FEATURE_COLUMNS)}")

    selection_model: LGBMClassifier = build_model(args.seed)
    selection_model.fit(train_x, train_y)
    valid_probabilities = selection_model.predict_proba(valid_x)[:, 1]

    costs = CostConfig(args.fp_cost, args.fn_cost)
    threshold, best, threshold_results = choose_economic_threshold(
        valid_y, valid_probabilities, costs
    )
    ranking = ranking_metrics(valid_y, valid_probabilities)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    selection_model.booster_.save_model(str(artifact_dir / "selection_model.lgbm"))

    with (artifact_dir / "threshold_search.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=threshold_results[0].keys())
        writer.writeheader()
        writer.writerows(threshold_results)

    with (artifact_dir / "validation_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["event_id", "timestamp", "is_fraud", "fraud_probability"])
        for local_idx, global_idx in enumerate(range(split, n_transactions)):
            writer.writerow(
                [
                    event_ids[global_idx],
                    timestamps[global_idx],
                    int(y[global_idx]),
                    f"{float(valid_probabilities[local_idx]):.10f}",
                ]
            )

    print()
    print("Baseline B validation results")
    print(f"  Threshold:    {threshold:.2f}")
    print(f"  Precision:    {best['precision']:.6f}")
    print(f"  Recall:       {best['recall']:.6f}")
    print(f"  FPR:          {best['fpr']:.6f}")
    print(f"  FNR:          {best['fnr']:.6f}")
    print(f"  PR-AUC:       {ranking['average_precision']:.6f}")
    print(f"  ROC-AUC:      {ranking['roc_auc']}")
    print(f"  FP:           {best['false_positives']:,}")
    print(f"  FN:           {best['false_negatives']:,}")
    print(f"  Econ. cost:   ₹{best['economic_cost']:,.0f}")

    print()
    print("Retraining final model on all World A transactions...")
    final_model = build_model(args.seed)
    final_model.fit(x, y)
    final_model.booster_.save_model(str(artifact_dir / "model.lgbm"))

    metadata = {
        "artifact_version": "1.0",
        "detector": "baseline_b",
        "model": {
            "library": "lightgbm",
            "model_type": "LGBMClassifier",
            "parameters": final_model.get_params(),
            "selection_model_file": "selection_model.lgbm",
            "final_model_file": "model.lgbm",
        },
        "feature_list": list(FEATURE_COLUMNS),
        "information_boundary": {
            "transaction_local": True,
            "infrastructure_graph": True,
            "behavioral_graph": False,
            "temporal_state": False,
            "transaction_history": False,
        },
        "training": {
            "world": "world_a",
            "seed": args.seed,
            "events_file": str(events_path),
            "ground_truth_file": str(gt_path),
            "transactions": n_transactions,
            "train_fraction_for_selection": args.train_fraction,
            "selection_train_rows": split,
            "selection_validation_rows": n_transactions - split,
            "final_training_rows": n_transactions,
            "final_training_fraud": int(y.sum()),
        },
        "split": {
            "strategy": "chronological",
            "basis": "event_timestamp",
            "ordering": ["timestamp", "event_id"],
            "split_timestamp": timestamps[split],
        },
        "threshold": {
            "value": threshold,
            "selection_method": "minimum_validation_economic_cost",
            "grid_start": 0.01,
            "grid_end": 0.99,
            "grid_step": 0.01,
            "false_positive_cost": costs.false_positive_cost,
            "false_negative_cost": costs.false_negative_cost,
        },
        "validation_metrics": {
            **best,
            **ranking,
            "validation_rows": len(valid_y),
            "validation_fraud": int(valid_y.sum()),
        },
        "graph": {
            "store": "neo4j_aura",
            "relationships": ["USES_DEVICE", "USES_IP"],
            "writes": ["account_created", "account_updated"],
            "transaction_graph_writes": False,
            "read_before_score": True,
            "persistence": "batched_account_lifecycle_writes",
            "batch_size": NEO4J_BATCH_SIZE,
        },
        "runtime": {
            "training_ingestion": "chronological_jsonl",
            "kafka_used_for_training": False,
            "kafka_reserved_for_streaming_evaluation_runtime": True,
        },
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    print()
    print(f"Baseline B complete in {time.perf_counter() - started:.2f}s")
    print(f"  Frozen threshold: {threshold:.2f}")
    print(f"  Artifacts: {artifact_dir}")


if __name__ == "__main__":
    main()
