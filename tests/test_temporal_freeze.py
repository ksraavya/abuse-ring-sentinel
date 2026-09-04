from __future__ import annotations

import json
from pathlib import Path

import pytest

from models.temporal import FEATURE_COLUMNS
from scripts.freeze_temporal_world_a import validate_frozen_artifact


def _write_fake_artifact(tmp_path: Path, *, infrastructure_columns: list[str] | None = None) -> None:
    import numpy as np
    from models.temporal import build_model, save_artifact

    model = build_model(seed=42)
    x = np.zeros((8, len(FEATURE_COLUMNS)), dtype=np.float32)
    y = np.array([0, 1, 0, 0, 1, 0, 0, 1], dtype=np.int8)
    model.fit(x, y)

    metadata = {
        "artifact_version": "1.0",
        "detector": "temporal",
        "feature_contract": {
            "transaction_local": list(FEATURE_COLUMNS[:10]),
            "temporal_behavioral": list(FEATURE_COLUMNS[10:]),
            "infrastructure_raw_columns_in_model": infrastructure_columns or [],
            "total": 26,
        },
        "information_boundary": {
            "future_state": False,
            "ground_truth_visible_to_detector": False,
        },
        "training": {"world": "world_a"},
        "threshold": {"value": 0.37},
        "validation_metrics": {"validation_rows": 3, "precision": 0.5, "recall": 0.5},
    }
    save_artifact(model, tmp_path, metadata)
    model.booster_.save_model(str(tmp_path / "selection_model.lgbm"))
    (tmp_path / "threshold_search.csv").write_text("threshold,economic_cost\n0.37,0\n", encoding="utf-8")
    (tmp_path / "validation_predictions.csv").write_text(
        "event_id,timestamp,is_fraud,fraud_probability\ne1,t,0,0.1\n",
        encoding="utf-8",
    )


def test_freeze_validator_accepts_valid_artifact(tmp_path: Path) -> None:
    _write_fake_artifact(tmp_path)
    result = validate_frozen_artifact(tmp_path)
    assert result["feature_count"] == 26
    assert result["threshold"] == pytest.approx(0.37)
    assert len(result["model_sha256"]) == 64


def test_freeze_validator_rejects_raw_infrastructure_columns(tmp_path: Path) -> None:
    _write_fake_artifact(tmp_path, infrastructure_columns=["device_degree"])
    with pytest.raises(ValueError, match="raw infrastructure"):
        validate_frozen_artifact(tmp_path)


def test_freeze_manifest_contains_locked_feature_list(tmp_path: Path) -> None:
    _write_fake_artifact(tmp_path)
    from scripts.freeze_temporal_world_a import main

    import sys
    old_argv = sys.argv
    try:
        sys.argv = ["freeze_temporal_world_a", "--artifact-dir", str(tmp_path)]
        main()
    finally:
        sys.argv = old_argv

    manifest = json.loads((tmp_path / "freeze_manifest.json").read_text(encoding="utf-8"))
    assert manifest["feature_count"] == 26
    assert manifest["feature_list"] == list(FEATURE_COLUMNS)
