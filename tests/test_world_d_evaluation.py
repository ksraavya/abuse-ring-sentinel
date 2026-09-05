from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.evaluate_world_d import (
    _confusion,
    _lead_time_stats,
    _load_frozen_verifier,
)


def test_confusion_matches_transaction_level_block_metrics() -> None:
    labels = np.array([0, 1, 1, 0, 0, 1], dtype=np.int8)
    predictions = np.array([False, True, False, True, False, False])
    result = _confusion(labels, predictions)
    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["true_negatives"] == 2
    assert result["false_negatives"] == 2
    assert result["precision"] == 0.5
    assert result["recall"] == 1 / 3
    assert result["fpr"] == 1 / 3


def test_lead_time_uses_only_strictly_pre_abuse_detections() -> None:
    from datetime import datetime, timezone, timedelta

    t0 = datetime(2026, 1, 10, tzinfo=timezone.utc)
    first_fraud = {"r1": t0, "r2": t0}
    first_detection = {"r1": t0 - timedelta(days=2), "r2": t0 + timedelta(days=1)}
    count, stats = _lead_time_stats(first_fraud, first_detection)
    assert count == 1
    assert stats["mean_days"] == 2.0
    assert stats["median_days"] == 2.0


def test_frozen_verifier_config_round_trips(tmp_path: Path) -> None:
    config = {
        "world": "world_c",
        "purpose": "verifier_and_policy_development_then_freeze",
        "ground_truth_used_only_for_selection": True,
        "selection": {
            "constraint": "block_precision >= 0.7000",
            "priority": [
                "ring_recall",
                "pre_abuse_recall",
                "block_precision",
                "exposure_prevented_pct",
                "-false_positive_blocks",
            ],
        },
        "fusion": {
            "detector_weight": 0.65,
            "coverage_bonus": 0.08,
            "expected_agent_names": [
                "ring-investigator",
                "infrastructure-investigator",
                "context-investigator",
            ],
            "type_weights": {
                "ring_structure": 0.24,
                "behavioral_acceleration": 0.18,
                "peer_synchrony": 0.14,
                "merchant_convergence": 0.12,
                "infrastructure_sharing": 0.12,
                "infrastructure_churn": 0.05,
                "account_context": 0.07,
                "temporal_context": 0.08,
            },
            "strength_multipliers": {
                "weak": 0.35,
                "moderate": 0.65,
                "strong": 1.0,
            },
        },
        "policy": {
            "detector_alert_threshold": 0.01,
            "block_detector_threshold": 0.20,
            "block_verifier_threshold": 0.45,
            "review_verifier_threshold": 0.10,
            "min_block_agent_coverage": 2,
            "require_strong_evidence_for_block": True,
            "policy_version": "12b-world-c-frozen-v1",
        },
    }
    path = tmp_path / "freeze_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    fusion, policy, loaded = _load_frozen_verifier(path)
    assert fusion.config.detector_weight == 0.65
    assert fusion.config.coverage_bonus == 0.08
    assert policy.config.block_verifier_threshold == 0.45
    assert policy.config.min_block_agent_coverage == 2
    assert loaded["world"] == "world_c"
