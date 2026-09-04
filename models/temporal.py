from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from lightgbm import Booster, LGBMClassifier

from features.temporal import TEMPORAL_FEATURE_NAMES
from features.transaction_local import FEATURE_COLUMNS as TRANSACTION_LOCAL_FEATURE_COLUMNS

TRANSACTION_FEATURE_COLUMNS: tuple[str, ...] = TRANSACTION_LOCAL_FEATURE_COLUMNS
FEATURE_COLUMNS: tuple[str, ...] = (
    *TRANSACTION_LOCAL_FEATURE_COLUMNS,
    *TEMPORAL_FEATURE_NAMES,
)

EXPECTED_FEATURE_COUNT = 26

if len(FEATURE_COLUMNS) != EXPECTED_FEATURE_COUNT:
    raise RuntimeError(
        f"Temporal feature contract must contain {EXPECTED_FEATURE_COUNT} columns; "
        f"found {len(FEATURE_COLUMNS)}"
    )


class CostConfig:
    def __init__(
        self,
        false_positive_cost: float = 500.0,
        false_negative_cost: float = 5000.0,
    ) -> None:
        if false_positive_cost < 0 or false_negative_cost < 0:
            raise ValueError("costs must be non-negative")
        self.false_positive_cost = float(false_positive_cost)
        self.false_negative_cost = float(false_negative_cost)


def build_model(seed: int = 42) -> LGBMClassifier:
    """Fixed Temporal model configuration.

    The estimator is deliberately aligned with Baselines A/B so that the
    experiment changes the information available to the model rather than
    silently changing the learner family or tuning regime.
    """
    return LGBMClassifier(
        objective="binary",
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=100,
        subsample=0.9,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def threshold_grid() -> np.ndarray:
    return np.round(np.arange(0.01, 1.00, 0.01), 2)


def threshold_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    costs: CostConfig,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if y_true.shape[0] != probabilities.shape[0]:
        raise ValueError("y_true and probabilities must have equal length")

    predictions = probabilities >= threshold
    tp = int(np.sum(predictions & (y_true == 1)))
    fp = int(np.sum(predictions & (y_true == 0)))
    tn = int(np.sum(~predictions & (y_true == 0)))
    fn = int(np.sum(~predictions & (y_true == 1)))

    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    fpr = float(fp / (fp + tn)) if fp + tn else 0.0
    fnr = float(fn / (fn + tp)) if fn + tp else 0.0

    return {
        "threshold": float(threshold),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "fnr": fnr,
        "economic_cost": float(
            costs.false_positive_cost * fp
            + costs.false_negative_cost * fn
        ),
    }


def choose_economic_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    costs: CostConfig,
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    results = [
        threshold_metrics(y_true, probabilities, float(t), costs)
        for t in threshold_grid()
    ]
    best = min(results, key=lambda row: (row["economic_cost"], row["threshold"]))
    return float(best["threshold"]), best, results


def ranking_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float | None]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = np.asarray(y_true, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    return {
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "roc_auc": (
            float(roc_auc_score(y_true, probabilities))
            if np.unique(y_true).size == 2
            else None
        ),
    }


def save_artifact(
    model: LGBMClassifier,
    artifact_dir: str | Path,
    metadata: dict[str, Any],
) -> None:
    path = Path(artifact_dir)
    path.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(path / "model.lgbm"))
    payload = {
        **metadata,
        "feature_list": list(FEATURE_COLUMNS),
        "feature_count": len(FEATURE_COLUMNS),
    }
    (path / "metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_artifact(
    artifact_dir: str | Path,
) -> tuple[Booster, dict[str, Any]]:
    path = Path(artifact_dir)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("feature_list") != list(FEATURE_COLUMNS):
        raise ValueError("Artifact feature list does not match Temporal model")
    if metadata.get("feature_count") != EXPECTED_FEATURE_COUNT:
        raise ValueError("Artifact feature count does not match Temporal model")
    return Booster(model_file=str(path / "model.lgbm")), metadata
