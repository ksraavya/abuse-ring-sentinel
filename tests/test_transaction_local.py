from features.transaction_local import FEATURE_COLUMNS, flat_event_to_feature_row


def test_feature_order_is_fixed():
    assert FEATURE_COLUMNS == (
        "amount",
        "amount_log",
        "hour",
        "day_of_week",
        "is_weekend",
        "is_night",
        "channel_upi",
        "channel_card",
        "channel_wallet",
        "channel_netbanking",
    )


def test_transaction_local_features():
    raw = {
        "event_id": "t1",
        "event_type": "transaction",
        "world_id": "world_a",
        "timestamp": "2026-01-03T23:00:00Z",
        "account_id": "a1",
        "merchant_id": "m1",
        "amount": 100.0,
        "channel": "upi",
        "device_id": "d1",
        "ip_prefix": "10.0.0.0/24",
    }
    row = flat_event_to_feature_row(raw)
    assert row[0] == 100.0
    assert row[2] == 23.0
    assert row[4] == 1.0
    assert row[5] == 1.0
    assert row[6] == 1.0
    assert sum(row[6:]) == 1.0
