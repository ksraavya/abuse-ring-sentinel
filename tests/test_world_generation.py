from __future__ import annotations

import hashlib
import json

from worlds.generator import WorldGenerator
from worlds.schema import WorldConfig


def small_config(seed: int = 42) -> WorldConfig:
    return WorldConfig(
        world_id="world_a",
        seed=seed,
        duration_days=60,
        legitimate_accounts=200,
        hard_negative_family_clusters=3,
        hard_negative_hostel_clusters=1,
        hard_negative_corporate_clusters=1,
        fast_forming_rings=2,
        slow_burn_rings=1,
        obfuscated_rings=1,
        topology_distribution={"distributed": 0.5, "star": 0.2, "chain": 0.15, "cluster": 0.15},
        organic_transactions=8000,
        organic_p2p_fraction=0.20,
        activity_sigma=1.0,
        ring_size_min=5,
        ring_size_max=8,
        precursor_interactions_per_ring=8,
        precursor_merchant_events_per_ring=6,
        ring_cover_events_per_member=2,
        precursor_acceleration_events_per_member=2,
        fraud_events_per_participant_min=3,
        fraud_events_per_participant_max=5,
        fraud_participation_probability=0.65,
    )


def test_disk_generation_is_reproducible(tmp_path):
    cfg = small_config()
    first = WorldGenerator(cfg).generate_to_disk(tmp_path / "first")
    second = WorldGenerator(cfg).generate_to_disk(tmp_path / "second")

    a = (tmp_path / "first" / "events.jsonl").read_bytes()
    b = (tmp_path / "second" / "events.jsonl").read_bytes()
    assert hashlib.sha256(a).digest() == hashlib.sha256(b).digest()
    assert first["event_count"] == second["event_count"]


def test_disk_events_are_chronological_and_ground_truth_is_separate(tmp_path):
    out = tmp_path / "world"
    manifest = WorldGenerator(small_config()).generate_to_disk(out)

    event_lines = (out / "events.jsonl").read_text().splitlines()
    gt_lines = (out / "ground_truth.jsonl").read_text().splitlines()
    events = [json.loads(x) for x in event_lines]
    ground_truth = [json.loads(x) for x in gt_lines]

    timestamps = [x["timestamp"] for x in events]
    assert timestamps == sorted(timestamps)
    assert len(events) == manifest["event_count"]
    assert len(ground_truth) == len([x for x in events if x["event_type"] == "transaction"])
    assert all("is_fraud" not in x for x in events)
    assert all("is_fraud" in x for x in ground_truth)
    assert manifest["account_update_count"] >= 0


def test_disk_world_contains_realistic_event_mix(tmp_path):
    out = tmp_path / "world"
    manifest = WorldGenerator(small_config()).generate_to_disk(out)
    events = [json.loads(x) for x in (out / "events.jsonl").read_text().splitlines()]
    txns = [x for x in events if x["event_type"] == "transaction"]
    p2p = [x for x in txns if x["counterparty_account_id"] is not None]
    merchant = [x for x in txns if x["merchant_id"] is not None]
    assert len(txns) > 5000
    assert p2p
    assert merchant
    assert manifest["fraud_transaction_count"] > 0
    assert manifest["fraud_transaction_count"] / len(txns) < 0.02
