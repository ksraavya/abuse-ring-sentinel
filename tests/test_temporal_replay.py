from __future__ import annotations

import json
from pathlib import Path

import pytest

from graph.temporal_replay import (
    TEMPORAL_MODEL_FEATURE_COLUMNS,
    TemporalReplay,
    TemporalReplayState,
    replay_world,
)


def account_created(event_id: str, timestamp: str, account_id: str, device: str, ip: str) -> dict:
    return {
        "event_id": event_id,
        "event_type": "account_created",
        "world_id": "world_a",
        "timestamp": timestamp,
        "account_id": account_id,
        "device_id": device,
        "ip_prefix": ip,
    }


def account_updated(event_id: str, timestamp: str, account_id: str, old_device: str, old_ip: str, new_device: str, new_ip: str) -> dict:
    return {
        "event_id": event_id,
        "event_type": "account_updated",
        "world_id": "world_a",
        "timestamp": timestamp,
        "account_id": account_id,
        "old_device_id": old_device,
        "old_ip_prefix": old_ip,
        "new_device_id": new_device,
        "new_ip_prefix": new_ip,
        "update_reason": "test",
    }


def transaction(
    event_id: str,
    timestamp: str,
    account_id: str,
    *,
    counterparty: str | None = None,
    merchant: str | None = None,
) -> dict:
    return {
        "event_id": event_id,
        "event_type": "transaction",
        "world_id": "world_a",
        "timestamp": timestamp,
        "account_id": account_id,
        "merchant_id": merchant,
        "counterparty_account_id": counterparty,
        "amount": 100.0,
        "channel": "upi",
        "device_id": "d1",
        "ip_prefix": "10.0.0",
    }


def write_world(tmp_path: Path, events: list[dict], labels: dict[str, bool]) -> tuple[Path, Path]:
    events_path = tmp_path / "events.jsonl"
    gt_path = tmp_path / "ground_truth.jsonl"
    with events_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")
    with gt_path.open("w", encoding="utf-8") as handle:
        for event_id, is_fraud in labels.items():
            handle.write(
                json.dumps({"event_id": event_id, "world_id": "world_a", "is_fraud": is_fraud})
                + "\n"
            )
    return events_path, gt_path


def test_temporal_replay_has_10_local_plus_16_temporal_features():
    assert len(TEMPORAL_MODEL_FEATURE_COLUMNS) == 26
    assert TEMPORAL_MODEL_FEATURE_COLUMNS[:10] == (
        "amount", "amount_log", "hour", "day_of_week", "is_weekend",
        "is_night", "channel_upi", "channel_card", "channel_wallet", "channel_netbanking",
    )


def test_lifecycle_events_update_infrastructure_state():
    replay = TemporalReplay()
    replay.process_event(account_created("e1", "2026-01-01T00:00:00Z", "a1", "d1", "ip1"))
    replay.process_event(account_created("e2", "2026-01-01T00:01:00Z", "a2", "d1", "ip2"))

    state = replay.state.infrastructure
    assert state.account_to_device == {"a1": "d1", "a2": "d1"}
    assert state.device_to_accounts["d1"] == {"a1", "a2"}
    assert state.features("a1")["shared_device_accounts"] == 1

    replay.process_event(
        account_updated("e3", "2026-01-01T00:02:00Z", "a2", "d1", "ip2", "d2", "ip3")
    )
    assert state.account_to_device["a2"] == "d2"
    assert "a2" not in state.device_to_accounts["d1"]
    assert state.device_to_accounts["d2"] == {"a2"}


def test_transaction_does_not_mutate_infrastructure_state():
    replay = TemporalReplay()
    replay.process_event(account_created("e1", "2026-01-01T00:00:00Z", "a1", "d1", "ip1"))
    before = (
        dict(replay.state.infrastructure.account_to_device),
        dict(replay.state.infrastructure.account_to_ip),
        {k: set(v) for k, v in replay.state.infrastructure.device_to_accounts.items()},
        {k: set(v) for k, v in replay.state.infrastructure.ip_to_accounts.items()},
    )

    replay.process_event(transaction("e2", "2026-01-01T01:00:00Z", "a1", merchant="m1"))

    after = (
        dict(replay.state.infrastructure.account_to_device),
        dict(replay.state.infrastructure.account_to_ip),
        {k: set(v) for k, v in replay.state.infrastructure.device_to_accounts.items()},
        {k: set(v) for k, v in replay.state.infrastructure.ip_to_accounts.items()},
    )
    assert after == before


def test_current_transaction_is_not_in_temporal_state_when_scored():
    replay = TemporalReplay()
    replay.process_event(account_created("e1", "2026-01-01T00:00:00Z", "a1", "d1", "ip1"))

    observed = {}
    current = transaction("e2", "2026-01-01T01:00:00Z", "a1", counterparty="a2")

    def score(row):
        observed["row"] = row
        observed["edge"] = replay.state.behavioral.get_edge("a1", "a2")
        observed["history"] = list(replay.state.temporal_features.account_events.get("a1", []))

    replay.process_event(current, score_callback=score)

    assert len(observed["row"]) == 26
    assert observed["edge"] is None
    assert observed["history"] == []
    assert replay.state.behavioral.get_edge("a1", "a2") is not None
    assert len(replay.state.temporal_features.account_events["a1"]) == 1


def test_replay_world_callback_runs_before_state_update(tmp_path: Path):
    events = [
        account_created("e1", "2026-01-01T00:00:00Z", "a1", "d1", "ip1"),
        account_created("e2", "2026-01-01T00:01:00Z", "a2", "d2", "ip2"),
        transaction("e3", "2026-01-01T01:00:00Z", "a1", counterparty="a2"),
    ]
    events_path, gt_path = write_world(tmp_path, events, {"e3": False})
    from graph.temporal_replay import load_ground_truth

    seen = {}
    state = TemporalReplayState()

    def on_txn(event_id, timestamp, row, is_fraud):
        seen["event_id"] = event_id
        seen["length"] = len(row)
        seen["edge_before"] = state.behavioral.get_edge("a1", "a2")
        seen["infra"] = state.infrastructure.features("a1")
        seen["label"] = is_fraud

    replay_world(events_path, load_ground_truth(gt_path), on_transaction=on_txn, state=state)

    assert seen == {
        "event_id": "e3",
        "length": 26,
        "edge_before": None,
        "infra": {
            "degree": 2,
            "device_degree": 1,
            "ip_degree": 1,
            "shared_device_accounts": 0,
            "shared_ip_accounts": 0,
            "max_device_sharing": 1,
            "max_ip_sharing": 1,
        },
        "label": False,
    }
    assert state.behavioral.get_edge("a1", "a2") is not None


def test_replay_rejects_missing_ground_truth(tmp_path: Path):
    events_path, _ = write_world(
        tmp_path,
        [transaction("e1", "2026-01-01T00:00:00Z", "a1", merchant="m1")],
        {},
    )
    with pytest.raises(ValueError, match="Missing ground truth"):
        replay_world(events_path, {}, on_transaction=lambda *_: None)


def test_replay_rejects_non_chronological_events(tmp_path: Path):
    events_path, gt_path = write_world(
        tmp_path,
        [
            transaction("e2", "2026-01-01T01:00:00Z", "a1", merchant="m1"),
            transaction("e1", "2026-01-01T00:00:00Z", "a1", merchant="m2"),
        ],
        {"e1": False, "e2": False},
    )
    from graph.temporal_replay import load_ground_truth

    with pytest.raises(ValueError, match="chronological"):
        replay_world(events_path, load_ground_truth(gt_path), on_transaction=lambda *_: None)
