from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from lightgbm import Booster, LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

from features.transaction_local import FEATURE_COLUMNS


@dataclass(frozen=True)
class CostConfig:
    false_positive_cost: float = 500.0
    false_negative_cost: float = 5000.0


def build_model(seed: int = 42) -> LGBMClassifier:
    """Fixed Baseline A configuration.

    The model is intentionally kept modest rather than heavily tuned. The
    experimental question is the value of transaction-local information.
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
    predictions = probabilities >= threshold

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
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
    """Evaluate exactly 99 thresholds: 0.01 through 0.99."""
    best: dict[str, Any] | None = None
    all_results: list[dict[str, Any]] = []

    for threshold in threshold_grid():
        result = threshold_metrics(y_true, probabilities, float(threshold), costs)
        all_results.append(result)

        # Lower threshold wins deterministic ties.
        if (
            best is None
            or result["economic_cost"] < best["economic_cost"]
            or (
                result["economic_cost"] == best["economic_cost"]
                and result["threshold"] < best["threshold"]
            )
        ):
            best = result

    assert best is not None
    return float(best["threshold"]), best, all_results


def ranking_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float | None]:
    y_true = np.asarray(y_true, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)

    return {
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
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
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)

    model.booster_.save_model(str(artifact_path / "model.lgbm"))

    (artifact_path / "metadata.json").write_text(
        json.dumps(
            {
                **metadata,
                "feature_list": list(FEATURE_COLUMNS),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def load_artifact(
    artifact_dir: str | Path,
) -> tuple[Booster, dict[str, Any]]:
    artifact_path = Path(artifact_dir)
    metadata = json.loads(
        (artifact_path / "metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("feature_list") != list(FEATURE_COLUMNS):
        raise ValueError("Artifact feature list does not match Baseline A")

    return Booster(model_file=str(artifact_path / "model.lgbm")), metadata
