from __future__ import annotations

from datetime import timedelta

import pytest

from events.schema import AccountCreatedEvent, TransactionEvent
from worlds.generator import WorldGenerator
from worlds.schema import WorldConfig


def config(seed: int = 42, **overrides) -> WorldConfig:
    values = {
        "world_id": "world_a",
        "seed": seed,
        "duration_days": 180,
        "legitimate_accounts": 100,
        "hard_negative_family_clusters": 2,
        "hard_negative_hostel_clusters": 1,
        "hard_negative_corporate_clusters": 1,
        "fast_forming_rings": 0,
        "slow_burn_rings": 0,
        "obfuscated_rings": 0,
        "topology_distribution": {
            "distributed": 0.85,
            "star": 0.05,
            "chain": 0.05,
            "cluster": 0.05,
        },
    }
    values.update(overrides)
    return WorldConfig(**values)


def test_same_seed_is_deterministic():
    a = WorldGenerator(config()).generate()
    b = WorldGenerator(config()).generate()

    assert [x.model_dump() for x in a] == [x.model_dump() for x in b]


def test_different_seed_changes_world():
    a = WorldGenerator(config(seed=42)).generate()
    b = WorldGenerator(config(seed=4242)).generate()

    assert [x.model_dump() for x in a] != [x.model_dump() for x in b]


def test_events_are_sorted():
    events = WorldGenerator(config()).generate()
    timestamps = [record.event.timestamp for record in events]

    assert timestamps == sorted(timestamps)


def test_event_ids_are_unique():
    events = WorldGenerator(config()).generate()
    event_ids = [record.event.event_id for record in events]

    assert len(event_ids) == len(set(event_ids))


def test_account_ids_are_unique():
    events = WorldGenerator(config()).generate()
    account_ids = [
        record.event.account_id
        for record in events
        if isinstance(record.event, AccountCreatedEvent)
    ]

    assert len(account_ids) == len(set(account_ids))


def test_transactions_reference_existing_accounts():
    events = WorldGenerator(config()).generate()

    accounts = {
        record.event.account_id
        for record in events
        if isinstance(record.event, AccountCreatedEvent)
    }
    transactions = [
        record.event
        for record in events
        if isinstance(record.event, TransactionEvent)
    ]

    assert transactions
    assert all(tx.account_id in accounts for tx in transactions)


def test_ground_truth_is_separate_from_account_events():
    events = WorldGenerator(config()).generate()

    for record in events:
        if isinstance(record.event, AccountCreatedEvent):
            assert record.ground_truth is None


def test_transaction_ground_truth_matches_event():
    events = WorldGenerator(config()).generate()

    for record in events:
        if isinstance(record.event, TransactionEvent):
            assert record.ground_truth is not None
            assert record.ground_truth.event_id == record.event.event_id


def test_events_stay_inside_world_window():
    cfg = config()
    events = WorldGenerator(cfg).generate()

    start = WorldGenerator.START_TIME
    end = start + timedelta(days=cfg.duration_days)

    assert all(start <= record.event.timestamp < end for record in events)


def test_unknown_topology_is_rejected():
    with pytest.raises(ValueError, match="unknown topology types"):
        config(
            topology_distribution={
                "distributed": 0.85,
                "star": 0.05,
                "chain": 0.05,
                "cluster": 0.04,
                "unknown": 0.01,
            }
        )


def test_negative_topology_weight_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        config(
            topology_distribution={
                "distributed": -0.01,
                "star": 0.05,
                "chain": 0.05,
                "cluster": 0.90,
            }
        )


def test_empty_topology_distribution_is_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        config(topology_distribution={})


def test_zero_sum_topology_distribution_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        config(
            topology_distribution={
                "distributed": 0.0,
                "star": 0.0,
                "chain": 0.0,
                "cluster": 0.0,
            }
        )


def test_total_rings():
    cfg = config(
        fast_forming_rings=2,
        slow_burn_rings=3,
        obfuscated_rings=4,
    )

    assert cfg.total_rings == 9
