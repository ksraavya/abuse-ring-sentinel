from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from lightgbm import LGBMClassifier

from graph.temporal_replay import (
    TEMPORAL_MODEL_FEATURE_COLUMNS,
    count_transactions,
    load_ground_truth,
    replay_world,
)
from features.temporal import TEMPORAL_FEATURE_NAMES
from features.transaction_local import FEATURE_COLUMNS as TRANSACTION_LOCAL_FEATURE_COLUMNS
from models.temporal import (
    CostConfig,
    build_model,
    choose_economic_threshold,
    ranking_metrics,
    save_artifact,
)


EXPECTED_FEATURE_COUNT = 26


def write_validation_predictions(
    path: Path,
    context_path: Path,
    probabilities: np.ndarray,
) -> None:
    rows_written = 0
    with context_path.open("r", encoding="utf-8") as context, path.open(
        "w", encoding="utf-8", newline=""
    ) as output:
        reader = csv.DictReader(context)
        writer = csv.writer(output)
        writer.writerow(["event_id", "timestamp", "is_fraud", "fraud_probability"])
        for row, probability in zip(reader, probabilities, strict=True):
            writer.writerow(
                [
                    row["event_id"],
                    row["timestamp"],
                    row["is_fraud"],
                    f"{float(probability):.10f}",
                ]
            )
            rows_written += 1
    if rows_written != len(probabilities):
        raise RuntimeError(
            f"Validation prediction count mismatch: wrote {rows_written}, "
            f"expected {len(probabilities)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the causal Temporal detector from chronological World A JSONL."
    )
    parser.add_argument("--events", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--artifact-dir", default="artifacts/temporal")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp-cost", type=float, default=500.0)
    parser.add_argument("--fn-cost", type=float, default=5000.0)
    args = parser.parse_args()

    if not 0.5 <= args.train_fraction < 1.0:
        raise SystemExit("--train-fraction must be in [0.5, 1.0)")

    if len(TEMPORAL_MODEL_FEATURE_COLUMNS) != EXPECTED_FEATURE_COUNT:
        raise SystemExit("Temporal model feature contract is not 26 columns")

    started = time.perf_counter()
    events_path = Path(args.events)
    gt_path = Path(args.ground_truth)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ground truth...")
    ground_truth = load_ground_truth(gt_path)
    print(f"  Ground-truth records: {len(ground_truth):,}")

    print("Counting World A transactions...")
    n_transactions = count_transactions(events_path)
    print(f"  Transactions: {n_transactions:,}")
    if n_transactions == 0:
        raise SystemExit("World A contains no transactions")
    if len(ground_truth) != n_transactions:
        raise SystemExit(
            "Ground-truth transaction count does not match event transaction count: "
            f"{len(ground_truth):,} != {n_transactions:,}"
        )

    split = int(n_transactions * args.train_fraction)
    if split <= 0 or split >= n_transactions:
        raise SystemExit("Chronological split must leave both train and validation rows")

    # A float32 memmap keeps the 26-column training matrix out of Python heap
    # memory. This matters for the ~3M transaction World A dataset.
    matrix_path = artifact_dir / ".temporal_training_features.dat"
    validation_context_path = artifact_dir / ".validation_context.csv"
    x = np.memmap(
        matrix_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_transactions, EXPECTED_FEATURE_COUNT),
    )
    y = np.memmap(
        artifact_dir / ".temporal_training_labels.dat",
        mode="w+",
        dtype=np.int8,
        shape=(n_transactions,),
    )

    transaction_index = 0
    first_timestamp: str | None = None
    split_timestamp: str | None = None

    context_handle = validation_context_path.open(
        "w", encoding="utf-8", newline=""
    )
    context_writer = csv.writer(context_handle)
    context_writer.writerow(["event_id", "timestamp", "is_fraud"])

    def collect(
        event_id: str,
        timestamp: str,
        row: tuple[float, ...],
        is_fraud: bool,
    ) -> None:
        nonlocal transaction_index, first_timestamp, split_timestamp
        if len(row) != EXPECTED_FEATURE_COUNT:
            raise RuntimeError(
                f"Temporal row has {len(row)} features; expected {EXPECTED_FEATURE_COUNT}"
            )
        if transaction_index >= n_transactions:
            raise RuntimeError("Replay produced more transactions than expected")

        if first_timestamp is None:
            first_timestamp = timestamp
        if transaction_index == split:
            split_timestamp = timestamp

        x[transaction_index] = np.asarray(row, dtype=np.float32)
        y[transaction_index] = 1 if is_fraud else 0

        if transaction_index >= split:
            context_writer.writerow([event_id, timestamp, int(is_fraud)])

        transaction_index += 1
        if transaction_index % 100_000 == 0:
            print(f"  Replayed {transaction_index:,}/{n_transactions:,} transactions")

    try:
        print("Chronologically replaying World A...")
        replayed = replay_world(
            events_path,
            ground_truth,
            on_transaction=collect,
        )
    finally:
        context_handle.close()

    if replayed != n_transactions or transaction_index != n_transactions:
        raise RuntimeError(
            f"Replay count mismatch: expected {n_transactions:,}, "
            f"replayed {replayed:,}, collected {transaction_index:,}"
        )
    if split_timestamp is None or first_timestamp is None:
        raise RuntimeError("Replay did not produce the expected split timestamps")

    x.flush()
    y.flush()

    train_x = x[:split]
    valid_x = x[split:]
    train_y = y[:split]
    valid_y = y[split:]

    print()
    print("Temporal dataset")
    print(f"  Transactions: {n_transactions:,}")
    print(f"  Fraud:        {int(y.sum()):,}")
    print(f"  Fraud rate:   {float(y.mean()):.6%}")
    print(f"  Train:        {len(train_x):,}")
    print(f"  Validation:   {len(valid_x):,}")
    print(f"  Split time:   {split_timestamp}")
    print(f"  Train fraud:  {int(train_y.sum()):,}")
    print(f"  Valid fraud:  {int(valid_y.sum()):,}")
    print(f"  Features:     {len(TEMPORAL_MODEL_FEATURE_COLUMNS)}")

    costs = CostConfig(args.fp_cost, args.fn_cost)

    print("\nTraining selection model...")
    selection_model: LGBMClassifier = build_model(args.seed)
    selection_model.fit(train_x, train_y)
    valid_probabilities = selection_model.predict_proba(valid_x)[:, 1]

    threshold, best, threshold_results = choose_economic_threshold(
        valid_y,
        valid_probabilities,
        costs,
    )
    ranking = ranking_metrics(valid_y, valid_probabilities)

    selection_model.booster_.save_model(
        str(artifact_dir / "selection_model.lgbm")
    )

    with (artifact_dir / "threshold_search.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=threshold_results[0].keys())
        writer.writeheader()
        writer.writerows(threshold_results)

    write_validation_predictions(
        artifact_dir / "validation_predictions.csv",
        validation_context_path,
        valid_probabilities,
    )

    print("\nTemporal validation results")
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

    print("\nRetraining final model on all World A transactions...")
    final_model = build_model(args.seed)
    final_model.fit(x, y)

    metadata: dict[str, Any] = {
        "artifact_version": "1.0",
        "detector": "temporal",
        "model": {
            "library": "lightgbm",
            "model_type": "LGBMClassifier",
            "parameters": final_model.get_params(),
            "selection_model_file": "selection_model.lgbm",
            "final_model_file": "model.lgbm",
        },
        "feature_list": list(TEMPORAL_MODEL_FEATURE_COLUMNS),
        "feature_contract": {
            "transaction_local": list(TRANSACTION_LOCAL_FEATURE_COLUMNS),
            "temporal_behavioral": list(TEMPORAL_FEATURE_NAMES),
            "infrastructure_raw_columns_in_model": [],
            "total": EXPECTED_FEATURE_COUNT,
        },
        "information_boundary": {
            "transaction_local": True,
            "infrastructure_state_maintained": True,
            "infrastructure_raw_features_in_classifier": False,
            "behavioral_graph": True,
            "temporal_state": True,
            "future_state": False,
            "ground_truth_visible_to_detector": False,
        },
        "training": {
            "world": "world_a",
            "seed": args.seed,
            "events_file": str(events_path),
            "ground_truth_file": str(gt_path),
            "train_fraction_for_selection": args.train_fraction,
            "selection_train_rows": split,
            "selection_validation_rows": n_transactions - split,
            "final_training_rows": n_transactions,
            "final_training_fraud": int(y.sum()),
        },
        "split": {
            "strategy": "chronological",
            "basis": "event_timestamp_then_event_id",
            "split_timestamp": split_timestamp,
        },
        "causal_order": [
            "read_pre_event_state",
            "extract_26_features",
            "score_current_transaction",
            "update_state",
        ],
        "state_updates": {
            "account_created": ["InfrastructureState"],
            "account_updated": ["InfrastructureState"],
            "transaction": ["TemporalFeatureState", "BehavioralState_if_p2p"],
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
        "runtime": {
            "training_ingestion": "chronological_jsonl",
            "kafka_used_for_training": False,
            "kafka_reserved_for_streaming_evaluation_runtime": True,
            "in_memory_state_authoritative": True,
        },
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    save_artifact(final_model, artifact_dir, metadata)

    # The matrix is an implementation detail, not a model artifact. Remove it
    # after training so a normal artifact directory stays small.
    del train_x, valid_x, train_y, valid_y
    x._mmap.close()
    y._mmap.close()
    del x, y
    for path in (
        matrix_path,
        artifact_dir / ".temporal_training_labels.dat",
        validation_context_path,
    ):
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    print()
    print(f"Temporal training complete in {time.perf_counter() - started:.2f}s")
    print(f"  Frozen threshold: {threshold:.2f}")
    print(f"  Artifacts: {artifact_dir}")


if __name__ == "__main__":
    main()
