from events.schema import TransactionChannel, TransactionEvent, WorldId, EventRecord
from streaming.serialization import event_to_kafka_bytes, event_from_kafka_bytes

def test_kafka_payload_contains_event_only():
    event = TransactionEvent(
        event_id="tx-1", world_id=WorldId.WORLD_A,
        timestamp="2026-01-02T12:00:00Z", account_id="acc-1",
        merchant_id="m-1", amount=100.0,
        channel=TransactionChannel.UPI, device_id="dev-1", ip_prefix="10.0.0")
    payload = event_from_kafka_bytes(event_to_kafka_bytes(EventRecord(event=event)))
    assert "event_id" in payload
    assert "ground_truth" not in payload
    assert payload["event_id"] == "tx-1"

def test_kafka_payload_round_trips_event_fields():
    event = TransactionEvent(
        event_id="tx-2", world_id=WorldId.WORLD_B,
        timestamp="2026-01-03T13:00:00Z", account_id="acc-2",
        merchant_id="m-2", amount=250.5,
        channel=TransactionChannel.CARD, device_id="dev-2", ip_prefix="10.1.1")
    payload = event_from_kafka_bytes(event_to_kafka_bytes(EventRecord(event=event)))
    assert payload["event_id"] == event.event_id
    assert payload["world_id"] == event.world_id.value
    assert payload["amount"] == event.amount
    assert payload["merchant_id"] == event.merchant_id
