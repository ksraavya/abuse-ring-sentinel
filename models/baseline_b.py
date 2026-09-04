from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import numpy as np
from lightgbm import Booster, LGBMClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score, roc_auc_score

from features.baseline_b import FEATURE_COLUMNS


class CostConfig:
    def __init__(self, false_positive_cost: float = 500.0, false_negative_cost: float = 5000.0):
        self.false_positive_cost = false_positive_cost
        self.false_negative_cost = false_negative_cost


def build_model(seed: int = 42) -> LGBMClassifier:
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


def threshold_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float, costs: CostConfig) -> dict[str, Any]:
    predictions = np.asarray(probabilities) >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "fnr": float(fn / (fn + tp)) if fn + tp else 0.0,
        "economic_cost": float(costs.false_positive_cost * fp + costs.false_negative_cost * fn),
    }


def choose_economic_threshold(y_true: np.ndarray, probabilities: np.ndarray, costs: CostConfig):
    results = [threshold_metrics(y_true, probabilities, float(t), costs) for t in threshold_grid()]
    best = min(results, key=lambda r: (r["economic_cost"], r["threshold"]))
    return float(best["threshold"]), best, results


def ranking_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float | None]:
    return {
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if np.unique(y_true).size == 2 else None,
    }


def save_artifact(model: LGBMClassifier, artifact_dir: str | Path, metadata: dict[str, Any]) -> None:
    path = Path(artifact_dir)
    path.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(path / "model.lgbm"))
    (path / "metadata.json").write_text(
        json.dumps({**metadata, "feature_list": list(FEATURE_COLUMNS)}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_artifact(artifact_dir: str | Path) -> tuple[Booster, dict[str, Any]]:
    path = Path(artifact_dir)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("feature_list") != list(FEATURE_COLUMNS):
        raise ValueError("Artifact feature list does not match Baseline B")
    return Booster(model_file=str(path / "model.lgbm")), metadata
