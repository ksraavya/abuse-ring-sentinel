from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
import json

import pytest

from events.schema import AccountCreatedEvent, AccountUpdatedEvent, TransactionEvent
from worlds.generator import WorldGenerator
from worlds.schema import WorldConfig


def config(seed: int = 42, **overrides) -> WorldConfig:
    values = {
        "world_id": "world_a",
        "seed": seed,
        "duration_days": 60,
        "legitimate_accounts": 120,
        "hard_negative_family_clusters": 4,
        "hard_negative_hostel_clusters": 2,
        "hard_negative_corporate_clusters": 2,
        "fast_forming_rings": 2,
        "slow_burn_rings": 2,
        "obfuscated_rings": 2,
        "topology_distribution": {
            "distributed": 0.50,
            "star": 0.20,
            "chain": 0.15,
            "cluster": 0.15,
        },
        "organic_transactions": 4000,
        "organic_p2p_fraction": 0.20,
        "activity_sigma": 1.0,
        "activity_min_weight": 0.10,
        "ring_size_min": 5,
        "ring_size_max": 7,
        "ring_cover_events_per_member": 2,
        "precursor_interactions_per_ring": 8,
        "precursor_merchant_events_per_ring": 6,
        "precursor_acceleration_events_per_member": 2,
        "fraud_events_per_participant_min": 3,
        "fraud_events_per_participant_max": 5,
        "hard_negative_p2p_rate_per_account_day": 0.20,
        "hard_negative_activity_multiplier": 1.35,
        "fraud_participation_probability": 0.65,
        "precursor_probability": 1.0,
    }
    values.update(overrides)
    return WorldConfig(**values)


def test_same_seed_is_deterministic():
    a = WorldGenerator(config()).generate()
    b = WorldGenerator(config()).generate()

    assert [x.model_dump(mode="json") for x in a] == [x.model_dump(mode="json") for x in b]


def test_different_seed_changes_world():
    a = WorldGenerator(config(seed=42)).generate()
    b = WorldGenerator(config(seed=4242)).generate()

    assert [x.model_dump(mode="json") for x in a] != [x.model_dump(mode="json") for x in b]


def test_events_are_sorted_and_inside_window():
    cfg = config()
    events = WorldGenerator(cfg).generate()
    timestamps = [r.event.timestamp for r in events]
    assert timestamps == sorted(timestamps)
    end = WorldGenerator.START_TIME + timedelta(days=cfg.duration_days)
    assert all(WorldGenerator.START_TIME <= t < end for t in timestamps)



def test_event_ids_are_unique():
    ids = [r.event.event_id for r in WorldGenerator(config()).generate()]
    assert len(ids) == len(set(ids))


def test_accounts_are_unique():
    ids = [r.event.account_id for r in WorldGenerator(config()).generate() if isinstance(r.event, AccountCreatedEvent)]
    assert len(ids) == len(set(ids))


def test_transactions_reference_existing_accounts():
    events = WorldGenerator(config()).generate()
    accounts = {r.event.account_id for r in events if isinstance(r.event, AccountCreatedEvent)}
    txns = [r.event for r in events if isinstance(r.event, TransactionEvent)]
    assert txns
    assert all(tx.account_id in accounts for tx in txns)
    assert all(tx.counterparty_account_id is None or tx.counterparty_account_id in accounts for tx in txns)

def test_counterparties_exist_before_event_time():
    events = WorldGenerator(config()).generate()
    created = {r.event.account_id: r.event.timestamp for r in events if isinstance(r.event, AccountCreatedEvent)}
    for r in events:
        if isinstance(r.event, TransactionEvent):
            assert created[r.event.account_id] <= r.event.timestamp
            if r.event.counterparty_account_id:
                assert created[r.event.counterparty_account_id] <= r.event.timestamp


def test_ground_truth_is_separate_from_account_events():
    assert all(
        r.ground_truth is None
        for r in WorldGenerator(config()).generate()
        if isinstance(r.event, (AccountCreatedEvent, AccountUpdatedEvent))
    )

def test_fraud_is_merchant_directed_and_p2p_is_not_fraud():
    records = WorldGenerator(config()).generate()
    fraud = [r for r in records if isinstance(r.event, TransactionEvent) and r.ground_truth and r.ground_truth.is_fraud]
    p2p = [r for r in records if isinstance(r.event, TransactionEvent) and r.event.counterparty_account_id]
    assert fraud
    assert all(r.event.merchant_id is not None and r.event.counterparty_account_id is None for r in fraud)
    assert all(not r.ground_truth.is_fraud for r in p2p)


def test_fraud_rate_is_low():
    records = WorldGenerator(config()).generate()
    txns = [r for r in records if isinstance(r.event, TransactionEvent)]
    fraud = sum(bool(r.ground_truth and r.ground_truth.is_fraud) for r in txns)
    assert fraud > 0
    assert fraud / len(txns) < 0.02


def test_rings_have_precursors_and_fraud():
    generated = WorldGenerator(config()).generate_with_manifest()
    cfg = config()
    assert generated.manifest["ring_count"] == cfg.total_rings
    for ring in generated.manifest["rings"]:
        assert ring["coordination_start_day"] < ring["coordination_end_day"] < ring["activation_day"]
        assert ring["precursor_emitted"] is True
        assert ring["behavioral_edge_count"] >= 1
        assert ring["precursor_interaction_count"] >= 1
        assert ring["cover_event_count"] >= 1
        assert ring["acceleration_event_count"] >= 1
        assert ring["fraud_transaction_count"] >= 1


def test_ring_accounts_exist_before_coordination():
    generated = WorldGenerator(config()).generate_with_manifest()
    created = {r.event.account_id: r.event.timestamp for r in generated.records if isinstance(r.event, AccountCreatedEvent)}
    start = WorldGenerator.START_TIME
    for ring in generated.manifest["rings"]:
        coordination = start + timedelta(days=ring["coordination_start_day"])
        assert all(created[a] < coordination for a in ring["account_ids"])


def test_ring_topologies_follow_config_and_are_heterogeneous():
    generated = WorldGenerator(config()).generate_with_manifest()
    topologies = {r["topology"] for r in generated.manifest["rings"]}
    assert len(topologies) >= 2
    assert topologies <= {"distributed", "star", "chain", "cluster"}


def test_distributed_rings_have_no_shared_infrastructure():
    cfg = config(topology_distribution={"distributed": 1.0, "star": 0.0, "chain": 0.0, "cluster": 0.0})
    generated = WorldGenerator(cfg).generate_with_manifest()
    account_events = {r.event.account_id: r.event for r in generated.records if isinstance(r.event, AccountCreatedEvent)}
    for ring in generated.manifest["rings"]:
        devices = [account_events[a].device_id for a in ring["account_ids"]]
        ips = [account_events[a].ip_prefix for a in ring["account_ids"]]
        assert len(devices) == len(set(devices))
        assert len(ips) == len(set(ips))
        assert ring["behavioral_edge_count"] >= 1


def test_non_distributed_rings_have_observable_infrastructure_overlap():
    generated = WorldGenerator(config(topology_distribution={"star": 1.0, "distributed": 0.0, "chain": 0.0, "cluster": 0.0})).generate_with_manifest()
    account_events = {r.event.account_id: r.event for r in generated.records if isinstance(r.event, AccountCreatedEvent)}
    for ring in generated.manifest["rings"]:
        devices = [account_events[a].device_id for a in ring["account_ids"]]
        ips = [account_events[a].ip_prefix for a in ring["account_ids"]]
        assert len(devices) > len(set(devices)) or len(ips) > len(set(ips))


def test_account_churn_is_real_and_has_update_event():
    generated = WorldGenerator(config(topology_distribution={"distributed": 1.0, "star": 0.0, "chain": 0.0, "cluster": 0.0})).generate_with_manifest()
    updates = [r.event for r in generated.records if isinstance(r.event, AccountUpdatedEvent)]
    assert updates
    for update in updates:
        assert update.old_device_id != update.new_device_id
        assert update.old_ip_prefix != update.new_ip_prefix
        created = next(r.event for r in generated.records if isinstance(r.event, AccountCreatedEvent) and r.event.account_id == update.account_id)
        assert created.timestamp < update.timestamp


def test_churn_is_not_restricted_to_one_ring_kind():
    generated = WorldGenerator(config()).generate_with_manifest()
    kind_by_account = {a: r["kind"] for r in generated.manifest["rings"] for a in r["account_ids"]}
    churned_kinds = {kind_by_account[a] for r in generated.manifest["rings"] for a in r["churned_account_ids"]}
    assert churned_kinds >= {"fast", "slow_burn"} or len(churned_kinds) >= 2


def test_churned_account_uses_new_identity_after_update():
    generated = WorldGenerator(config()).generate_with_manifest()
    updates = {r.event.account_id: r.event for r in generated.records if isinstance(r.event, AccountUpdatedEvent)}
    for record in generated.records:
        if isinstance(record.event, TransactionEvent) and record.event.account_id in updates:
            update = updates[record.event.account_id]
            if record.event.timestamp >= update.timestamp:
                assert record.event.device_id == update.new_device_id
                assert record.event.ip_prefix == update.new_ip_prefix


def test_behavioral_precursors_precede_ring_fraud():
    generated = WorldGenerator(config()).generate_with_manifest()
    account_to_ring = {a: r["ring_id"] for r in generated.manifest["rings"] for a in r["account_ids"]}
    p2p_times = defaultdict(list)
    fraud_times = defaultdict(list)
    for record in generated.records:
        if not isinstance(record.event, TransactionEvent):
            continue
        ring_id = record.ground_truth.ring_id if record.ground_truth and record.ground_truth.is_fraud else None
        if ring_id:
            fraud_times[ring_id].append(record.event.timestamp)
        elif record.event.counterparty_account_id:
            ring_id = account_to_ring.get(record.event.account_id)
            if ring_id:
                p2p_times[ring_id].append(record.event.timestamp)
    for ring in generated.manifest["rings"]:
        rid = ring["ring_id"]
        assert p2p_times[rid]
        assert fraud_times[rid]
        assert min(p2p_times[rid]) < min(fraud_times[rid])


def test_activity_accelerates_toward_coordination_end():
    generated = WorldGenerator(config()).generate_with_manifest()
    start = WorldGenerator.START_TIME
    by_account = defaultdict(list)
    for record in generated.records:
        if isinstance(record.event, TransactionEvent):
            by_account[record.event.account_id].append(record.event.timestamp)

    checked = 0
    for ring in generated.manifest["rings"]:
        coord_start = start + timedelta(days=ring["coordination_start_day"])
        coord_end = start + timedelta(days=ring["coordination_end_day"])
        midpoint = coord_start + (coord_end - coord_start) * 0.5
        last_half = 0
        first_half = 0
        for account_id in ring["account_ids"]:
            times = by_account[account_id]
            first_half += sum(coord_start <= t < midpoint for t in times)
            last_half += sum(midpoint <= t <= coord_end for t in times)
        if first_half + last_half > 0:
            assert last_half >= first_half
            checked += 1
    assert checked >= 3


def test_hard_negative_clusters_have_structured_p2p():
    generated = WorldGenerator(config()).generate_with_manifest()
    cluster_map = {a: c["cluster_id"] for c in generated.manifest["hard_negative_clusters"] for a in c["account_ids"]}
    p2p = [
        r for r in generated.records
        if isinstance(r.event, TransactionEvent)
        and r.event.counterparty_account_id
        and cluster_map.get(r.event.account_id) == cluster_map.get(r.event.counterparty_account_id)
    ]
    assert len(p2p) > 10


def test_activity_is_heterogeneous():
    records = WorldGenerator(config()).generate()
    counts = defaultdict(int)
    for r in records:
        if isinstance(r.event, TransactionEvent):
            counts[r.event.account_id] += 1
    values = sorted(counts.values())
    assert len(values) > 30
    assert values[-1] > values[0]
    assert values[-1] >= 2 * values[len(values) // 2]


def test_serialization_keeps_ground_truth_separate(tmp_path):
    generated = WorldGenerator(config()).generate_with_manifest()
    events_path = tmp_path / "events.jsonl"
    gt_path = tmp_path / "ground_truth.jsonl"
    WorldGenerator.write_jsonl(generated.records, events_path)
    WorldGenerator.write_ground_truth(generated.records, gt_path)
    event_lines = [json.loads(x) for x in events_path.read_text().splitlines()]
    gt_lines = [json.loads(x) for x in gt_path.read_text().splitlines()]
    assert event_lines and gt_lines
    assert all("ground_truth" not in x and "is_fraud" not in x for x in event_lines)
    assert all("is_fraud" in x for x in gt_lines)


def test_full_scale_config_is_three_million_class():
    full = WorldConfig(
        world_id="world_a", seed=42, duration_days=180, legitimate_accounts=100000,
        hard_negative_family_clusters=100, hard_negative_hostel_clusters=30, hard_negative_corporate_clusters=20,
        fast_forming_rings=18, slow_burn_rings=12, obfuscated_rings=10,
        topology_distribution={"distributed": 0.85, "star": 0.05, "chain": 0.05, "cluster": 0.05},
        organic_transactions=2900000,
    )
    assert full.legitimate_accounts == 100000
    assert full.organic_transactions == 2900000
    assert full.duration_days == 180
