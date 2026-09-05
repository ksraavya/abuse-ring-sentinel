from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from events.schema import EventType, WorldId
from verifier.world_c_replay import WorldCReplayConfig, WorldCVerifierRunner


TS = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def _account(event_id: str, account: str, device: str, ip: str, ts: datetime) -> dict:
    return {
        "event_id": event_id,
        "event_type": EventType.ACCOUNT_CREATED.value,
        "world_id": WorldId.WORLD_C.value,
        "timestamp": ts.isoformat().replace("+00:00", "Z"),
        "account_id": account,
        "device_id": device,
        "ip_prefix": ip,
    }


def _p2p(event_id: str, sender: str, receiver: str, ts: datetime) -> dict:
    return {
        "event_id": event_id,
        "event_type": EventType.TRANSACTION.value,
        "world_id": WorldId.WORLD_C.value,
        "timestamp": ts.isoformat().replace("+00:00", "Z"),
        "account_id": sender,
        "merchant_id": None,
        "counterparty_account_id": receiver,
        "amount": 100.0,
        "channel": "upi",
        "device_id": "dev-1",
        "ip_prefix": "10.0.0",
    }


def _merchant(event_id: str, account: str, merchant: str, ts: datetime, amount: float = 100.0) -> dict:
    return {
        "event_id": event_id,
        "event_type": EventType.TRANSACTION.value,
        "world_id": WorldId.WORLD_C.value,
        "timestamp": ts.isoformat().replace("+00:00", "Z"),
        "account_id": account,
        "merchant_id": merchant,
        "counterparty_account_id": None,
        "amount": amount,
        "channel": "upi",
        "device_id": "dev-1",
        "ip_prefix": "10.0.0",
    }


def _fake_artifact(monkeypatch, threshold: float = 0.01):
    class FakeModel:
        def predict(self, x):
            return np.asarray([0.8] * len(x), dtype=np.float32)

        def num_feature(self):
            return 26

    metadata = {
        "threshold": {"value": threshold},
        "feature_list": list(__import__("models.temporal", fromlist=["FEATURE_COLUMNS"]).FEATURE_COLUMNS),
        "training": {"world": "world_a"},
        "feature_contract": {"total": 26, "infrastructure_raw_columns_in_model": []},
        "information_boundary": {"future_state": False, "ground_truth_visible_to_detector": False},
    }
    monkeypatch.setattr(
        "verifier.world_c_replay.load_artifact",
        lambda _path: (FakeModel(), metadata),
    )
    monkeypatch.setattr(
        "verifier.world_c_replay._sha256",
        lambda _path: "model-sha",
    )
    monkeypatch.setattr(
        "verifier.world_c_replay.json",
        json,
    )
    return metadata


def test_world_c_replay_is_alert_only_and_runs_all_investigators(tmp_path, monkeypatch):
    _fake_artifact(monkeypatch)
    events = tmp_path / "events.jsonl"
    rows = [
        _account("a1", "acc-1", "dev-1", "10.0.0", TS),
        _account("a2", "acc-2", "dev-1", "10.0.0", TS + timedelta(seconds=1)),
        _account("a3", "acc-3", "dev-1", "10.0.0", TS + timedelta(seconds=2)),
        _account("a4", "acc-4", "dev-1", "10.0.0", TS + timedelta(seconds=3)),
        _p2p("t1", "acc-2", "acc-1", TS + timedelta(hours=1)),
        _p2p("t2", "acc-3", "acc-1", TS + timedelta(hours=2)),
        _p2p("t3", "acc-1", "acc-2", TS + timedelta(hours=3)),
        _merchant("t4", "acc-1", "m-1", TS + timedelta(hours=4), amount=100.0),
    ]
    events.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    alerts = tmp_path / "alerts.jsonl"
    summary = tmp_path / "summary.json"
    runner = WorldCVerifierRunner(
        WorldCReplayConfig(
            artifact_dir=str(tmp_path / "artifact"),
            output_alerts=str(alerts),
            output_summary=str(summary),
        )
    )

    # Fake freeze manifest required by the artifact validator.
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "model.lgbm").write_bytes(b"fake")
    (artifact_dir / "metadata.json").write_text("{}")
    (artifact_dir / "freeze_manifest.json").write_text(
        json.dumps({"model_sha256": "model-sha", "threshold": 0.01, "feature_list": []})
    )
    # The monkeypatched loader supplies the real feature list; freeze manifest
    # is patched below to match it.
    from models.temporal import FEATURE_COLUMNS
    (artifact_dir / "freeze_manifest.json").write_text(
        json.dumps({"model_sha256": "model-sha", "threshold": 0.01, "feature_list": list(FEATURE_COLUMNS)})
    )

    seen = []
    result = runner.replay(events_path=events, on_alert=seen.append)

    assert result["transactions_replayed"] == 4
    assert result["detector_alerts"] == 4
    assert result["causal_contract"]["ground_truth_loaded_by_replay"] is False
    assert alerts.is_file()
    assert len(seen) == 4
    first = json.loads(alerts.read_text(encoding="utf-8").splitlines()[0])
    assert "is_fraud" not in first
    assert "ring_id" not in first
    assert "evidence" in first
    agents = {item["source_agent"] for item in first["evidence"]}
    assert "infrastructure-investigator" in agents
    # The later transaction has enough behavioral history to exercise ring and context agents.
    later = [json.loads(line) for line in alerts.read_text(encoding="utf-8").splitlines() if json.loads(line)["event_id"] == "t4"][0]
    later_agents = {item["source_agent"] for item in later["evidence"]}
    assert "ring-investigator" in later_agents
    assert "context-investigator" in later_agents


def test_world_c_rejects_world_d(tmp_path):
    try:
        WorldCReplayConfig(world="world_d")
    except ValueError as exc:
        assert "world_c" in str(exc)
    else:
        raise AssertionError("World D must never be accepted by the World C development runner")
