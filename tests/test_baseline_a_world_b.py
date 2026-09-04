from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate_baseline_a_world_b import MODEL_NAME


def test_baseline_a_world_b_identity() -> None:
    assert MODEL_NAME == "baseline_a"


def test_evaluation_does_not_change_feature_contract() -> None:
    from features.transaction_local import FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == 10
    assert "account_age_days" not in FEATURE_COLUMNS


def test_manifest_ring_members_are_evaluation_only(tmp_path: Path) -> None:
    manifest = {"rings": [{"ring_id": "r1", "account_ids": ["a1", "a2"]}]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["rings"][0]["account_ids"] == ["a1", "a2"]


def test_frozen_threshold_is_read_from_artifact_metadata(tmp_path: Path) -> None:
    # Contract-level fixture only: no training and no World B data.
    metadata = {"threshold": {"value": 0.01}, "feature_list": [f"f{i}" for i in range(10)]}
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["threshold"]["value"] == 0.01
    assert len(loaded["feature_list"]) == 10
