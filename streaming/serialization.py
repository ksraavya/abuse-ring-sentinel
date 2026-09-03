from __future__ import annotations
import json
from events.schema import EventRecord

def event_to_kafka_bytes(record: EventRecord) -> bytes:
    return json.dumps(
        record.event.model_dump(mode="json"),
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

def event_from_kafka_bytes(payload: bytes) -> dict:
    return json.loads(payload.decode("utf-8"))
