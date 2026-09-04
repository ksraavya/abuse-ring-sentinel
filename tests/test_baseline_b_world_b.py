from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from features.baseline_b import FEATURE_COLUMNS
from graph.infrastructure_state import InfrastructureState
from scripts.evaluate_baseline_b_world_b import DEFAULT_GROUP_ID, _event_timestamp


def test_baseline_b_world_b_contract_is_17_features() -> None:
    assert len(FEATURE_COLUMNS) == 17
    assert FEATURE_COLUMNS[:10] == (
        "amount", "amount_log", "hour", "day_of_week", "is_weekend",
        "is_night", "channel_upi", "channel_card", "channel_wallet", "channel_netbanking",
    )
    assert FEATURE_COLUMNS[10:] == (
        "degree", "device_degree", "ip_degree", "shared_device_accounts",
        "shared_ip_accounts", "max_device_sharing", "max_ip_sharing",
    )


def test_baseline_b_uses_lifecycle_state_before_transaction() -> None:
    state = InfrastructureState()
    state.add_or_update("a1", "d1", "10.0.0.0/24")
    state.add_or_update("a2", "d1", "10.0.0.0/24")

    before = state.features("a1")
    assert before["shared_device_accounts"] == 1
    assert before["shared_ip_accounts"] == 1

    # A transaction must not alter infrastructure state.
    after = state.features("a1")
    assert after == before


def test_evaluation_group_id_is_dedicated() -> None:
    assert DEFAULT_GROUP_ID == "risk-manager-eval-baseline-b"
    assert DEFAULT_GROUP_ID != "risk-manager-eval-baseline-a"


def test_event_timestamp_requires_timezone() -> None:
    with pytest.raises(ValueError):
        _event_timestamp({"timestamp": "2026-01-01T00:00:00"})


def test_evaluation_script_defaults_to_world_b_and_baseline_b() -> None:
    # Static source-level guard against accidentally changing the held-out
    # evaluation target or pointing at Baseline A artifacts.
    source = Path("scripts/evaluate_baseline_b_world_b.py").read_text(encoding="utf-8")
    assert '"world_b.events"' not in source  # topic comes from KafkaConfig
    assert 'default="artifacts/baseline_b"' in source
    assert 'default="artifacts/evaluation/world_b/baseline_b"' in source
    assert 'threshold_tuned_on_world_b' in source
