from graph.behavioral_state import BehavioralState


def test_directed_edge_metadata_accumulates():
    state = BehavioralState()

    state.update("a1", "a2", "2026-01-01T10:00:00Z", 100.0)
    state.update("a1", "a2", "2026-01-01T11:00:00Z", 50.0)

    edge = state.get_edge("a1", "a2")
    assert edge is not None
    assert edge.count == 2
    assert edge.total_amount == 150.0
    assert edge.first_seen == "2026-01-01T10:00:00Z"
    assert edge.last_seen == "2026-01-01T11:00:00Z"


def test_direction_is_preserved():
    state = BehavioralState()
    state.update("a1", "a2", "2026-01-01T10:00:00Z", 100.0)

    assert state.get_edge("a1", "a2") is not None
    assert state.get_edge("a2", "a1") is None


def test_neighbors_include_incoming_and_outgoing_accounts():
    state = BehavioralState()
    state.update("a1", "a2", "2026-01-01T10:00:00Z", 100.0)
    state.update("a3", "a1", "2026-01-01T10:01:00Z", 25.0)
    state.update("a1", "a4", "2026-01-01T10:02:00Z", 75.0)

    assert state.get_neighbors("a1") == {"a2", "a3", "a4"}


def test_unseen_account_has_no_neighbors_or_edge():
    state = BehavioralState()

    assert state.get_edge("a1", "a2") is None
    assert state.get_neighbors("a1") == set()


def test_same_pair_updates_existing_edge_not_new_edge():
    state = BehavioralState()
    state.update("a1", "a2", "2026-01-01T11:00:00Z", 50.0)
    state.update("a1", "a2", "2026-01-01T09:00:00Z", 25.0)

    assert len(state.edges) == 1
    edge = state.get_edge("a1", "a2")
    assert edge is not None
    assert edge.count == 2
    assert edge.total_amount == 75.0
    assert edge.first_seen == "2026-01-01T09:00:00Z"
    assert edge.last_seen == "2026-01-01T11:00:00Z"


def test_invalid_self_transaction_is_rejected():
    state = BehavioralState()

    try:
        state.update("a1", "a1", "2026-01-01T10:00:00Z", 100.0)
    except ValueError as exc:
        assert "different" in str(exc)
    else:
        raise AssertionError("self-transaction should be rejected")

def test_update_after_read_does_not_affect_prior_read():
    state = BehavioralState()
    state.update("a1", "a2", "2026-01-01T10:00:00Z", 100.0)
    
    # Read state before update
    neighbors_before = state.get_neighbors("a1")
    
    # Update happens after read (simulating post-score update)
    state.update("a1", "a3", "2026-01-01T11:00:00Z", 50.0)
    
    # The prior read is unaffected
    assert "a3" not in neighbors_before
    assert "a3" in state.get_neighbors("a1")
