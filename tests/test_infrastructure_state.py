from graph.infrastructure_state import InfrastructureState


def test_shared_device_and_ip_features():
    state = InfrastructureState()
    state.add_or_update("a1", "d1", "ip1")
    state.add_or_update("a2", "d1", "ip2")
    state.add_or_update("a3", "d3", "ip1")

    assert state.features("a1")["shared_device_accounts"] == 1
    assert state.features("a1")["shared_ip_accounts"] == 1
    assert state.features("a1")["degree"] == 2


def test_update_removes_old_identity():
    state = InfrastructureState()
    state.add_or_update("a1", "d1", "ip1")
    state.add_or_update("a2", "d1", "ip2")
    state.add_or_update("a1", "d9", "ip9")

    assert state.features("a2")["shared_device_accounts"] == 0
    assert state.features("a2")["shared_ip_accounts"] == 0
    assert state.features("a1")["degree"] == 2


def test_transaction_features_read_pre_event_state():
    state = InfrastructureState()
    state.add_or_update("a1", "d1", "ip1")
    state.add_or_update("a2", "d1", "ip2")

    # The current transaction cannot change infrastructure state.
    before = state.features("a1")
    assert before["shared_device_accounts"] == 1
    assert before["shared_ip_accounts"] == 0
