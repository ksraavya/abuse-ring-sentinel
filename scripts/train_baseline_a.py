from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from lightgbm import LGBMClassifier

from features.transaction_local import FEATURE_COLUMNS, flat_event_to_feature_row
from models.baseline_a import (
    CostConfig,
    build_model,
    choose_economic_threshold,
    ranking_metrics,
    save_artifact,
)


CHANNELS = {"upi", "card", "wallet", "netbanking"}
TRANSACTION_TYPE = "transaction"
ACCOUNT_TYPES = {"account_created", "account_updated"}


def load_ground_truth(path: Path) -> dict[str, bool]:
    labels: dict[str, bool] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            event_id = raw.get("event_id")
            if not event_id:
                raise ValueError(f"Ground truth line {line_number} has no event_id")
            if event_id in labels:
                raise ValueError(f"Duplicate ground-truth event_id: {event_id}")
            labels[event_id] = bool(raw["is_fraud"])

    return labels


def load_world_a_transactions(
    events_path: Path,
    ground_truth: dict[str, bool],
) -> tuple[np.ndarray, np.ndarray, list[str], str, str]:
    """Fast reader for the generated JSONL.

    The generator already emits a validated chronological stream. We perform
    lightweight JSON parsing and schema sanity checks here instead of invoking
    Pydantic for every one of ~3M records.
    """
    n_transactions = sum(
        1
        for line in events_path.open("r", encoding="utf-8")
        if line.strip() and '"event_type":"transaction"' in line
    )

    x = np.empty((n_transactions, len(FEATURE_COLUMNS)), dtype=np.float32)
    y = np.empty(n_transactions, dtype=np.int8)
    event_ids: list[str] = []
    timestamps: list[str] = []

    # Keep timestamp strings only until the split; this avoids millions of
    # Python datetime objects.
    previous_timestamp: str | None = None
    i = 0

    with events_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            raw: dict[str, Any] = json.loads(line)
            event_type = raw.get("event_type")

            if event_type in ACCOUNT_TYPES:
                continue
            if event_type != TRANSACTION_TYPE:
                raise ValueError(
                    f"Unknown event_type {event_type!r} at line {line_number}"
                )

            event_id = raw.get("event_id")
            timestamp = str(raw.get("timestamp", ""))
            channel = raw.get("channel")

            if not event_id or not timestamp:
                raise ValueError(f"Transaction line {line_number} missing event_id/timestamp")
            if channel not in CHANNELS:
                raise ValueError(f"Unknown channel {channel!r} at line {line_number}")
            if event_id not in ground_truth:
                raise ValueError(f"Missing ground truth for transaction {event_id}")
            if previous_timestamp is not None and timestamp < previous_timestamp:
                raise ValueError(
                    "events.jsonl is not chronological by timestamp; "
                    f"line {line_number} is earlier than its predecessor"
                )

            x[i] = flat_event_to_feature_row(raw)
            y[i] = 1 if ground_truth[event_id] else 0
            event_ids.append(event_id)
            timestamps.append(timestamp)

            previous_timestamp = timestamp
            i += 1

    if i != n_transactions:
        raise RuntimeError(f"Count mismatch while loading: expected {n_transactions}, got {i}")

    event_id_set = set(event_ids)
    unexpected = set(ground_truth) - event_id_set
    if unexpected:
        raise ValueError(
            f"Ground truth contains {len(unexpected)} event_id(s) absent from transactions"
        )

    split_index = int(n_transactions * 0.70)
    split_timestamp = timestamps[split_index]
    first_timestamp = timestamps[0]
    last_timestamp = timestamps[-1]

    return x, y, event_ids, split_timestamp, f"{first_timestamp}|{last_timestamp}"


def save_validation_predictions(
    path: Path,
    event_ids: list[str],
    timestamps: list[str],
    y_true: np.ndarray,
    probabilities: np.ndarray,
    split_index: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    validation_count = len(event_ids) - split_index

    if len(probabilities) != validation_count:
        raise ValueError(
            "Validation probability count does not match validation rows: "
            f"{len(probabilities)} != {validation_count}"
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["event_id", "timestamp", "is_fraud", "fraud_probability"]
        )

        for local_idx, global_idx in enumerate(
            range(split_index, len(event_ids))
        ):
            writer.writerow(
                [
                    event_ids[global_idx],
                    timestamps[global_idx],
                    int(y_true[global_idx]),
                    f"{float(probabilities[local_idx]):.10f}",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--artifact-dir", default="artifacts/baseline_a")
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

    print("Loading World A transactions...")
    x, y, event_ids, split_timestamp, world_time_range = load_world_a_transactions(
        events_path, ground_truth
    )
    split_index = int(len(x) * args.train_fraction)

    # We need timestamp strings for the validation artifact. Re-read only the
    # event file to collect transaction timestamps; this keeps the main arrays
    # compact and avoids retaining another large object array.
    timestamps: list[str] = []
    with events_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            if raw.get("event_type") == TRANSACTION_TYPE:
                timestamps.append(str(raw["timestamp"]))

    train_x = x[:split_index]
    valid_x = x[split_index:]
    train_y = y[:split_index]
    valid_y = y[split_index:]

    print()
    print("Baseline A dataset")
    print(f"  Transactions: {len(x):,}")
    print(f"  Fraud:        {int(y.sum()):,}")
    print(f"  Fraud rate:   {y.mean():.6%}")
    print(f"  Train:        {len(train_x):,}")
    print(f"  Validation:   {len(valid_x):,}")
    print(f"  Split time:   {split_timestamp}")
    print(f"  Train fraud:  {int(train_y.sum()):,}")
    print(f"  Valid fraud:  {int(valid_y.sum()):,}")

    costs = CostConfig(args.fp_cost, args.fn_cost)

    print()
    print("Training selection model...")
    selection_model: LGBMClassifier = build_model(args.seed)
    selection_model.fit(train_x, train_y)

    valid_probabilities = selection_model.predict_proba(valid_x)[:, 1]

    threshold, best_threshold_result, threshold_results = choose_economic_threshold(
        valid_y,
        valid_probabilities,
        costs,
    )
    rank = ranking_metrics(valid_y, valid_probabilities)

    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Preserve the exact evidence used to choose the threshold.
    threshold_csv = artifact_dir / "threshold_search.csv"
    with threshold_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=threshold_results[0].keys())
        writer.writeheader()
        writer.writerows(threshold_results)

    validation_csv = artifact_dir / "validation_predictions.csv"
    save_validation_predictions(
        validation_csv,
        event_ids,
        timestamps,
        y,
        valid_probabilities,
        split_index,
    )

    selection_model.booster_.save_model(
        str(artifact_dir / "selection_model.lgbm")
    )

    print()
    print("Baseline A validation results")
    print(f"  Threshold:    {threshold:.2f}")
    print(f"  Precision:    {best_threshold_result['precision']:.6f}")
    print(f"  Recall:       {best_threshold_result['recall']:.6f}")
    print(f"  FPR:          {best_threshold_result['fpr']:.6f}")
    print(f"  FNR:          {best_threshold_result['fnr']:.6f}")
    print(f"  PR-AUC:       {rank['average_precision']:.6f}")
    print(f"  ROC-AUC:      {rank['roc_auc']}")
    print(f"  FP:           {best_threshold_result['false_positives']:,}")
    print(f"  FN:           {best_threshold_result['false_negatives']:,}")
    print(f"  Econ. cost:   ₹{best_threshold_result['economic_cost']:,.0f}")

    print()
    print("Retraining final model on all World A transactions...")
    final_model = build_model(args.seed)
    final_model.fit(x, y)

    metadata = {
        "artifact_version": "1.1",
        "detector": "baseline_a",
        "model": {
            "library": "lightgbm",
            "model_type": "LGBMClassifier",
            "parameters": final_model.get_params(),
            "selection_model_file": "selection_model.lgbm",
            "final_model_file": "model.lgbm",
        },
        "feature_list": list(FEATURE_COLUMNS),
        "training": {
            "world": "world_a",
            "seed": args.seed,
            "events_file": str(events_path),
            "ground_truth_file": str(gt_path),
            "train_fraction_for_selection": args.train_fraction,
            "selection_train_rows": int(split_index),
            "selection_validation_rows": int(len(x) - split_index),
            "final_training_rows": int(len(x)),
            "final_training_fraud": int(y.sum()),
        },
        "split": {
            "strategy": "chronological",
            "basis": "event_timestamp",
            "ordering": ["timestamp", "event_id"],
            "split_timestamp": split_timestamp,
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
            **best_threshold_result,
            **rank,
            "validation_rows": int(len(valid_y)),
            "validation_fraud": int(valid_y.sum()),
        },
        "validation_artifacts": {
            "predictions_file": "validation_predictions.csv",
            "threshold_search_file": "threshold_search.csv",
            "selection_model_file": "selection_model.lgbm",
        },
        "data_integrity": {
            "transaction_count": int(len(x)),
            "fraud_count": int(y.sum()),
            "legitimate_count": int((y == 0).sum()),
            "world_time_range": world_time_range,
            "ground_truth_join_key": "event_id",
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    save_artifact(final_model, artifact_dir, metadata)

    elapsed = time.perf_counter() - started
    print()
    print(f"Final artifact written to: {artifact_dir}")
    print(f"  model.lgbm")
    print(f"  metadata.json")
    print(f"  selection_model.lgbm")
    print(f"  validation_predictions.csv")
    print(f"  threshold_search.csv")
    print(f"Total elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
