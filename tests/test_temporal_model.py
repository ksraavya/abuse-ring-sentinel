from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier

from features.temporal import TEMPORAL_FEATURE_NAMES
from features.transaction_local import FEATURE_COLUMNS as LOCAL_FEATURE_COLUMNS
from models.temporal import (
    EXPECTED_FEATURE_COUNT,
    FEATURE_COLUMNS,
    CostConfig,
    build_model,
    choose_economic_threshold,
    load_artifact,
    ranking_metrics,
    save_artifact,
    threshold_grid,
)


def test_temporal_feature_contract_is_exactly_26():
    assert EXPECTED_FEATURE_COUNT == 26
    assert FEATURE_COLUMNS == (*LOCAL_FEATURE_COLUMNS, *TEMPORAL_FEATURE_NAMES)
    assert len(FEATURE_COLUMNS) == 26
    assert "account_age_at_txn_days" not in FEATURE_COLUMNS
    assert "edge_creation_velocity" not in FEATURE_COLUMNS


def test_temporal_model_matches_baseline_learner_family():
    model = build_model(42)
    assert isinstance(model, LGBMClassifier)
    params = model.get_params()
    assert params["n_estimators"] == 250
    assert params["learning_rate"] == 0.05
    assert params["num_leaves"] == 31
    assert params["random_state"] == 42


def test_threshold_selection_is_deterministic_and_uses_costs():
    y = np.array([0, 0, 0, 1], dtype=np.int8)
    p = np.array([0.1, 0.2, 0.6, 0.7])
    threshold, best, results = choose_economic_threshold(y, p, CostConfig(500, 5000))
    assert threshold == 0.61
    assert best["economic_cost"] == 0.0
    assert len(results) == 99
    assert np.array_equal(threshold_grid(), np.round(np.arange(0.01, 1.00, 0.01), 2))


def test_ranking_metrics_require_only_predictions_and_labels():
    y = np.array([0, 1, 0, 1], dtype=np.int8)
    p = np.array([0.1, 0.8, 0.2, 0.7])
    metrics = ranking_metrics(y, p)
    assert metrics["average_precision"] > 0.0
    assert metrics["roc_auc"] is not None


def test_artifact_round_trip_preserves_26_feature_contract(tmp_path):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(40, EXPECTED_FEATURE_COUNT)).astype(np.float32)
    y = np.array([0] * 20 + [1] * 20, dtype=np.int8)
    model = build_model(42)
    model.fit(x, y)

    save_artifact(model, tmp_path, {"detector": "temporal"})
    booster, metadata = load_artifact(tmp_path)

    assert booster.num_feature() == EXPECTED_FEATURE_COUNT
    assert metadata["feature_list"] == list(FEATURE_COLUMNS)
    assert metadata["feature_count"] == EXPECTED_FEATURE_COUNT
