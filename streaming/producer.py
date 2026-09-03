from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from confluent_kafka import Producer

from events.schema import (
    AccountCreatedEvent,
    AccountUpdatedEvent,
    EventRecord,
    EventType,
    TransactionEvent,
)
from streaming.config import KafkaConfig
from streaming.serialization import event_to_kafka_bytes


def _parse_event(raw: dict):
    """Validate a flat event from events.jsonl against its concrete event schema."""
    event_type = raw.get("event_type")

    if event_type == EventType.ACCOUNT_CREATED:
        return AccountCreatedEvent.model_validate(raw)

    if event_type == EventType.ACCOUNT_UPDATED:
        return AccountUpdatedEvent.model_validate(raw)

    if event_type == EventType.TRANSACTION:
        return TransactionEvent.model_validate(raw)

    raise ValueError(f"Unknown event_type: {event_type!r}")


class WorldReplayProducer:
    def __init__(self, config: KafkaConfig | None = None) -> None:
        self.config = config or KafkaConfig()
        self.producer = Producer(
            {
                "bootstrap.servers": self.config.bootstrap_servers,
                "client.id": self.config.client_id,
                "acks": self.config.acks,
                "enable.idempotence": True,
            }
        )

    def replay(
        self,
        events_path: str | Path,
        world_id: str,
        rate: float = 0.0,
        limit: int | None = None,
    ) -> int:
        """Replay generated events into the world's Kafka topic.

        The source events.jsonl contains flat model-visible events.
        Ground truth is intentionally stored separately and is never
        published to Kafka.
        """
        path = Path(events_path)
        if not path.exists():
            raise FileNotFoundError(f"Events file not found: {path}")

        topic = self.config.topic(world_id)
        count = 0

        interval = 1.0 / rate if rate > 0 else 0.0
        next_send_time = time.monotonic()

        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue

                raw = json.loads(line)

                event = _parse_event(raw)
                record = EventRecord(event=event)

                payload = event_to_kafka_bytes(record)

                self.producer.produce(
                    topic,
                    key=event.event_id.encode("utf-8"),
                    value=payload,
                )
                self.producer.poll(0)

                count += 1

                if interval > 0:
                    next_send_time += interval
                    sleep_for = next_send_time - time.monotonic()
                    if sleep_for > 0:
                        time.sleep(sleep_for)

                if limit is not None and count >= limit:
                    break

        remaining = self.producer.flush(timeout=30)

        if remaining:
            raise RuntimeError(
                f"Kafka producer still has {remaining} message(s) "
                "queued after flush timeout."
            )

        return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a generated world event stream into Kafka."
    )
    parser.add_argument(
        "--world",
        required=True,
        choices=["world_a", "world_b"],
        help="World whose Kafka topic should receive the events.",
    )
    parser.add_argument(
        "--events",
        required=True,
        help="Path to the generated events.jsonl file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of events to replay.",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=0.0,
        help="Replay rate in events/second. 0 means as fast as possible.",
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than 0")

    if args.rate < 0:
        parser.error("--rate must be greater than or equal to 0")

    count = WorldReplayProducer().replay(
        args.events,
        args.world,
        args.rate,
        args.limit,
    )

    print(
        f"Replay complete: {count} event(s) produced "
        f"to {args.world}.events"
    )


if __name__ == "__main__":
    main()