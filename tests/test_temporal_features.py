from datetime import datetime, timezone

import pytest

from events.schema import TransactionChannel, TransactionEvent, WorldId
from features.temporal import TEMPORAL_FEATURE_NAMES, TemporalFeatureState
from graph.behavioral_state import BehavioralState


def tx(
    event_id: str,
    timestamp: str,
    account_id: str = "a1",
    counterparty: str | None = None,
    merchant: str | None = None,
    amount: float = 100.0,
) -> TransactionEvent:
    return TransactionEvent(
        event_id=event_id,
        world_id=WorldId.WORLD_A,
        timestamp=datetime.fromisoformat(timestamp.replace("Z", "+00:00")),
        account_id=account_id,
        merchant_id=merchant,
        counterparty_account_id=counterparty,
        amount=amount,
        channel=TransactionChannel.UPI,
        device_id="d1",
        ip_prefix="10.0.0.0/24",
    )


def add(history, graph, event):
    history.update(event)
    if event.counterparty_account_id is not None:
        graph.update(
            event.account_id,
            event.counterparty_account_id,
            event.timestamp,
            event.amount,
        )


def features_at(history, graph, event):
    return history.extract_features(event, event.timestamp, graph)


def test_feature_schema_has_exactly_the_frozen_16_features():
    assert len(TEMPORAL_FEATURE_NAMES) == 16
    assert TEMPORAL_FEATURE_NAMES == (
        "p2p_txn_count_24h",
        "p2p_unique_neighbors_24h",
        "p2p_unique_neighbors_14d",
        "p2p_amount_24h",
        "activity_count_6h",
        "activity_count_24h",
        "days_since_last_txn",
        "new_neighbors_14d",
        "new_neighbors_24h",
        "edge_creation_acceleration",
        "reciprocal_neighbors_14d",
        "neighbor_activity_6h",
        "neighbor_activity_24h",
        "cluster_activity_synchrony_6h",
        "neighbor_merchant_overlap_24h",
        "activity_acceleration_6h_vs_7d",
    )


def test_empty_history_returns_zero_vector():
    history = TemporalFeatureState()
    graph = BehavioralState()
    current = tx("current", "2026-01-01T10:00:00Z", counterparty="a2")

    result = features_at(history, graph, current)

    assert set(result) == set(TEMPORAL_FEATURE_NAMES)
    assert all(value == 0.0 for value in result.values())


def test_current_transaction_is_excluded_from_all_relevant_features():
    history = TemporalFeatureState()
    graph = BehavioralState()
    current = tx(
        "current",
        "2026-01-01T10:00:00Z",
        counterparty="a2",
        amount=500.0,
    )

    result = features_at(history, graph, current)

    assert result["p2p_txn_count_24h"] == 0.0
    assert result["p2p_unique_neighbors_24h"] == 0.0
    assert result["p2p_amount_24h"] == 0.0
    assert result["activity_count_6h"] == 0.0
    assert result["activity_count_24h"] == 0.0
    assert result["new_neighbors_24h"] == 0.0
    assert result["new_neighbors_14d"] == 0.0
    assert result["days_since_last_txn"] == 0.0


def test_exact_timestamp_boundary_is_excluded():
    history = TemporalFeatureState()
    graph = BehavioralState()
    add(history, graph, tx("same", "2026-01-01T10:00:00Z", counterparty="a2"))

    current = tx("current", "2026-01-01T10:00:00Z", counterparty="a3")
    result = features_at(history, graph, current)

    assert result["p2p_txn_count_24h"] == 0.0
    assert result["p2p_unique_neighbors_24h"] == 0.0
    assert result["activity_count_6h"] == 0.0


def test_lower_window_boundary_is_included():
    history = TemporalFeatureState()
    graph = BehavioralState()
    add(history, graph, tx("boundary", "2026-01-01T10:00:00Z", counterparty="a2"))

    current = tx("current", "2026-01-02T10:00:00Z", counterparty="a3")
    result = features_at(history, graph, current)

    assert result["p2p_txn_count_24h"] == 1.0


def test_future_event_is_excluded():
    history = TemporalFeatureState()
    graph = BehavioralState()
    history.update(tx("future", "2026-01-01T11:00:00Z", counterparty="a2"))

    current = tx("current", "2026-01-01T10:00:00Z", counterparty="a3")
    result = features_at(history, graph, current)

    assert result["p2p_txn_count_24h"] == 0.0
    assert result["activity_count_24h"] == 0.0


def test_p2p_metrics_count_only_outgoing_account_to_account_transactions():
    history = TemporalFeatureState()
    graph = BehavioralState()

    add(history, graph, tx("p2p", "2026-01-01T09:00:00Z", counterparty="a2", amount=120))
    history.update(tx("merchant", "2026-01-01T09:30:00Z", merchant="m1", amount=300))

    current = tx("current", "2026-01-01T10:00:00Z", counterparty="a3")
    result = features_at(history, graph, current)

    assert result["p2p_txn_count_24h"] == 1.0
    assert result["p2p_unique_neighbors_24h"] == 1.0
    assert result["p2p_amount_24h"] == 120.0


def test_merchant_transaction_does_not_create_behavioral_edge():
    history = TemporalFeatureState()
    graph = BehavioralState()
    history.update(tx("merchant", "2026-01-01T09:00:00Z", merchant="m1"))

    assert graph.get_neighbors("a1") == set()

    current = tx("current", "2026-01-01T10:00:00Z", counterparty="a2")
    result = features_at(history, graph, current)
    assert result["p2p_unique_neighbors_24h"] == 0.0


def test_receiver_activity_is_counted_as_activity_involving_the_account():
    history = TemporalFeatureState()
    graph = BehavioralState()
    add(
        history,
        graph,
        tx(
            "incoming",
            "2026-01-01T09:00:00Z",
            account_id="a2",
            counterparty="a1",
        ),
    )

    current = tx("current", "2026-01-01T10:00:00Z", account_id="a1", counterparty="a3")
    result = features_at(history, graph, current)

    assert result["activity_count_6h"] == 1.0
    assert result["activity_count_24h"] == 1.0
    assert result["p2p_txn_count_24h"] == 0.0


def test_new_neighbors_14d_counts_edge_first_seen_not_recent_reuse():
    history = TemporalFeatureState()
    graph = BehavioralState()
    add(history, graph, tx("old", "2025-12-01T10:00:00Z", counterparty="a2"))
    add(history, graph, tx("recent", "2026-01-01T09:00:00Z", counterparty="a2"))

    current = tx("current", "2026-01-01T10:00:00Z", counterparty="a3")
    result = features_at(history, graph, current)

    assert result["new_neighbors_14d"] == 0.0
    assert result["new_neighbors_24h"] == 0.0


def test_new_neighbors_24h_and_14d_distinguish_their_windows():
    history = TemporalFeatureState()
    graph = BehavioralState()
    add(history, graph, tx("older", "2025-12-25T10:00:00Z", counterparty="a2"))
    add(history, graph, tx("recent", "2026-01-01T09:00:00Z", counterparty="a3"))

    current = tx("current", "2026-01-01T10:00:00Z", counterparty="a4")
    result = features_at(history, graph, current)

    assert result["new_neighbors_14d"] == 2.0
    assert result["new_neighbors_24h"] == 1.0


def test_p2p_unique_neighbors_14d_counts_distinct_recent_counterparties():
    history = TemporalFeatureState()
    graph = BehavioralState()
    add(history, graph, tx("older-c", "2025-12-25T10:00:00Z", counterparty="a4"))
    add(history, graph, tx("recent-a", "2026-01-01T08:00:00Z", counterparty="a2"))
    add(history, graph, tx("recent-a-again", "2026-01-01T09:00:00Z", counterparty="a2"))
    add(history, graph, tx("recent-b", "2026-01-01T09:30:00Z", counterparty="a3"))

    current = tx("current", "2026-01-01T10:00:00Z", counterparty="a5")
    result = features_at(history, graph, current)

    assert result["p2p_unique_neighbors_24h"] == 2.0
    assert result["p2p_unique_neighbors_14d"] == 3.0


def test_new_neighbor_exact_14d_boundary_is_included():
    history = TemporalFeatureState()
    graph = BehavioralState()
    add(history, graph, tx("boundary", "2025-12-18T10:00:00Z", counterparty="a2"))

    current = tx("current", "2026-01-01T10:00:00Z", counterparty="a3")
    result = features_at(history, graph, current)

    assert result["new_neighbors_14d"] == 1.0


def test_edge_creation_acceleration_is_one_when_recent_and_baseline_rates_match():
    history = TemporalFeatureState()
    graph = BehavioralState()

    # One distinct new edge per day for the preceding 13 days.
    for i in range(13):
        day = 19 + i
        add(
            history,
            graph,
            tx(
                f"baseline-{i}",
                f"2025-12-{day:02d}T10:00:00Z",
                counterparty=f"b{i}",
            ),
        )

    # One new edge in the latest 24h => same one/day rate.
    add(history, graph, tx("recent", "2026-01-01T10:00:00Z", counterparty="recent"))

    current = tx("current", "2026-01-01T11:00:00Z", counterparty="next")
    result = features_at(history, graph, current)

    assert result["new_neighbors_24h"] == 1.0
    assert result["edge_creation_acceleration"] == pytest.approx(1.0)


def test_edge_creation_acceleration_is_greater_than_one_for_a_recent_burst():
    history = TemporalFeatureState()
    graph = BehavioralState()

    add(history, graph, tx("baseline", "2025-12-20T10:00:00Z", counterparty="b0"))
    for i, hour in enumerate((8, 9, 10), start=1):
        add(
            history,
            graph,
            tx(
                f"burst-{i}",
                f"2026-01-01T{hour:02d}:00:00Z",
                counterparty=f"b{i}",
            ),
        )

    current = tx("current", "2026-01-01T11:00:00Z", counterparty="next")
    result = features_at(history, graph, current)

    assert result["new_neighbors_24h"] == 3.0
    assert result["edge_creation_acceleration"] > 1.0


def test_edge_creation_acceleration_is_zero_when_no_new_edges_are_recent():
    history = TemporalFeatureState()
    graph = BehavioralState()
    add(history, graph, tx("old", "2025-12-20T10:00:00Z", counterparty="a2"))
    add(history, graph, tx("reuse", "2026-01-01T09:00:00Z", counterparty="a2"))

    current = tx("current", "2026-01-01T10:00:00Z", counterparty="a3")
    result = features_at(history, graph, current)

    assert result["new_neighbors_24h"] == 0.0
    assert result["edge_creation_acceleration"] == 0.0


def test_reciprocal_neighbors_require_recent_reverse_interaction():
    history = TemporalFeatureState()
    graph = BehavioralState()
    add(history, graph, tx("out", "2026-01-01T08:00:00Z", counterparty="a2"))
    add(
        history,
        graph,
        tx("reverse", "2026-01-01T09:00:00Z", account_id="a2", counterparty="a1"),
    )

    current = tx("current", "2026-01-01T10:00:00Z", counterparty="a3")
    result = features_at(history, graph, current)

    assert result["reciprocal_neighbors_14d"] == 1.0


def test_old_reverse_interaction_outside_14d_is_not_reciprocal():
    history = TemporalFeatureState()
    graph = BehavioralState()
    add(history, graph, tx("out", "2025-12-25T08:00:00Z", counterparty="a2"))
    add(
        history,
        graph,
        tx("reverse-old", "2025-12-25T09:00:00Z", account_id="a2", counterparty="a1"),
    )

    current = tx("current", "2026-01-10T10:00:00Z", counterparty="a3")
    result = features_at(history, graph, current)

    assert result["reciprocal_neighbors_14d"] == 0.0


def test_neighbor_activity_counts_transactions_of_pre_t_behavioral_neighbors():
    history = TemporalFeatureState()
    graph = BehavioralState()

    add(history, graph, tx("link", "2026-01-01T08:00:00Z", counterparty="a2"))
    add(history, graph, tx("neighbor-txn", "2026-01-01T09:00:00Z", account_id="a2", merchant="m1"))

    current = tx("current", "2026-01-01T10:00:00Z", counterparty="a3")
    result = features_at(history, graph, current)

    assert result["neighbor_activity_6h"] == 2.0
    assert result["neighbor_activity_24h"] == 2.0


def test_new_current_event_is_not_a_neighbor_before_it_creates_an_edge():
    history = TemporalFeatureState()
    graph = BehavioralState()
    add(history, graph, tx("link", "2026-01-01T08:00:00Z", counterparty="a2"))

    current = tx("current", "2026-01-01T10:00:00Z", counterparty="a3")
    result = features_at(history, graph, current)

    assert "a3" not in graph.get_neighbors("a1")
    assert result["neighbor_activity_6h"] == 1.0


def test_cluster_activity_synchrony_is_fraction_of_known_neighbors_active_in_6h():
    history = TemporalFeatureState()
    graph = BehavioralState()

    add(history, graph, tx("link1", "2026-01-01T03:00:00Z", counterparty="a2"))
    add(history, graph, tx("link2", "2026-01-01T03:30:00Z", counterparty="a3"))
    history.update(tx("a2-active", "2026-01-01T09:00:00Z", account_id="a2", merchant="m1"))

    current = tx("current", "2026-01-01T10:00:00Z", counterparty="a4")
    result = features_at(history, graph, current)

    assert result["cluster_activity_synchrony_6h"] == pytest.approx(0.5)


def test_merchant_overlap_counts_common_merchants_between_account_and_neighbors():
    history = TemporalFeatureState()
    graph = BehavioralState()

    add(history, graph, tx("link", "2026-01-01T07:00:00Z", counterparty="a2"))
    history.update(tx("account-m1", "2026-01-01T08:00:00Z", merchant="m1"))
    history.update(tx("account-m2", "2026-01-01T08:30:00Z", merchant="m2"))
    history.update(tx("neighbor-m1", "2026-01-01T09:00:00Z", account_id="a2", merchant="m1"))
    history.update(tx("neighbor-m3", "2026-01-01T09:30:00Z", account_id="a2", merchant="m3"))

    current = tx("current", "2026-01-01T10:00:00Z", counterparty="a3")
    result = features_at(history, graph, current)

    assert result["neighbor_merchant_overlap_24h"] == 1.0


def test_activity_acceleration_uses_preceding_7_days_not_current_6h():
    history = TemporalFeatureState()
    graph = BehavioralState()

    # Four transactions in the preceding 162h => expected 6h rate 4/27.
    baseline_times = (
        "2025-12-25T12:00:00Z",
        "2025-12-27T12:00:00Z",
        "2025-12-29T12:00:00Z",
        "2025-12-31T12:00:00Z",
    )
    for i, timestamp in enumerate(baseline_times):
        history.update(tx(f"base-{i}", timestamp, merchant=f"m{i}"))

    # Two current-window transactions.
    history.update(tx("recent-1", "2026-01-01T08:00:00Z", merchant="r1"))
    history.update(tx("recent-2", "2026-01-01T09:00:00Z", merchant="r2"))

    current = tx("current", "2026-01-01T10:00:00Z", counterparty="a2")
    result = features_at(history, graph, current)

    assert result["activity_count_6h"] == 2.0
    assert result["activity_acceleration_6h_vs_7d"] == pytest.approx(13.5)


def test_days_since_last_transaction_is_measured_from_pre_t_history():
    history = TemporalFeatureState()
    graph = BehavioralState()
    history.update(tx("last", "2026-01-01T04:00:00Z", merchant="m1"))

    current = tx("current", "2026-01-01T10:00:00Z", counterparty="a2")
    result = features_at(history, graph, current)

    assert result["days_since_last_txn"] == pytest.approx(0.25)


def test_timestamp_argument_must_match_event_timestamp():
    history = TemporalFeatureState()
    graph = BehavioralState()
    current = tx("e1", "2026-01-01T10:00:00Z", counterparty="a2")

    with pytest.raises(ValueError, match="exactly match"):
        history.extract_features(
            current,
            datetime(2026, 1, 1, 10, 1, tzinfo=timezone.utc),
            graph,
        )


def test_update_rejects_out_of_order_replay():
    history = TemporalFeatureState()
    history.update(tx("later", "2026-01-01T11:00:00Z", counterparty="a2"))

    with pytest.raises(ValueError, match="chronological"):
        history.update(tx("earlier", "2026-01-01T10:00:00Z", counterparty="a3"))


def test_self_transaction_is_rejected():
    history = TemporalFeatureState()
    with pytest.raises(ValueError, match="self-transactions"):
        history.update(
            tx("self", "2026-01-01T10:00:00Z", account_id="a1", counterparty="a1")
        )


def test_invalid_transaction_without_merchant_or_counterparty_is_rejected():
    history = TemporalFeatureState()
    with pytest.raises(ValueError, match="merchant or counterparty"):
        history.update(tx("invalid", "2026-01-01T10:00:00Z"))
