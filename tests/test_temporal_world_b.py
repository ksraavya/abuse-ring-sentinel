from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from models.temporal import FEATURE_COLUMNS, EXPECTED_FEATURE_COUNT
from scripts.evaluate_temporal_world_b import (
    DEFAULT_GROUP_ID,
    EXPECTED_PARTITIONS,
    _event_timestamp,
)


def test_temporal_world_b_contract_is_exactly_26_features() -> None:
    assert EXPECTED_FEATURE_COUNT == 26
    assert len(FEATURE_COLUMNS) == 26
    assert FEATURE_COLUMNS[:10] == (
        "amount", "amount_log", "hour", "day_of_week", "is_weekend",
        "is_night", "channel_upi", "channel_card", "channel_wallet", "channel_netbanking",
    )
    assert "account_age_days" not in FEATURE_COLUMNS
    assert "edge_creation_velocity" not in FEATURE_COLUMNS


def test_temporal_evaluation_uses_dedicated_consumer_group() -> None:
    assert DEFAULT_GROUP_ID == "risk-manager-eval-temporal"
    assert DEFAULT_GROUP_ID not in {
        "risk-manager-eval-baseline-a",
        "risk-manager-eval-baseline-b",
    }


def test_temporal_evaluation_requires_one_partition() -> None:
    assert EXPECTED_PARTITIONS == 1


def test_event_timestamp_requires_timezone() -> None:
    with pytest.raises(ValueError):
        _event_timestamp({"timestamp": "2026-01-01T00:00:00"})


def test_temporal_evaluation_defaults_are_world_b_and_frozen_temporal() -> None:
    source = Path("scripts/evaluate_temporal_world_b.py").read_text(encoding="utf-8")
    assert 'default="data/generated/world_b/events.jsonl"' in source
    assert 'default="data/generated/world_b/ground_truth.jsonl"' in source
    assert 'default="data/generated/world_b/manifest.json"' in source
    assert 'default="artifacts/temporal"' in source
    assert 'default="artifacts/evaluation/world_b/temporal"' in source
    assert 'threshold_tuned_on_world_b' in source
    assert 'model_retrained_on_world_b' in source


def test_frozen_manifest_contract_is_explicit() -> None:
    source = Path("scripts/evaluate_temporal_world_b.py").read_text(encoding="utf-8")
    assert "freeze_manifest.json" in source
    assert "model SHA does not match" in source
    assert "future_state" in source
    assert "ground_truth_visible_to_detector" in source


def test_score_callback_boundary_is_delegated_to_temporal_replay() -> None:
    source = Path("scripts/evaluate_temporal_world_b.py").read_text(encoding="utf-8")
    assert "replay.process_event(raw, score_callback=score_current_transaction)" in source
    assert "score before any current-transaction state" not in source
    # The evaluator must not manually update TemporalFeatureState or BehavioralState;
    # TemporalReplay owns that causal boundary.
    assert "replay.state.temporal_features.update" not in source
    assert "replay.state.behavioral.update" not in source


def test_world_b_ground_truth_is_not_published_to_kafka() -> None:
    source = Path("scripts/evaluate_temporal_world_b.py").read_text(encoding="utf-8")
    assert '"ground_truth_published_to_kafka": False' in source
    assert "ground_truth_source" in source
