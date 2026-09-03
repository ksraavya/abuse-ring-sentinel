from __future__ import annotations
import json
import os
import time
from confluent_kafka import Consumer, Producer
from confluent_kafka.admin import AdminClient, NewTopic


def _wait_for_broker(bootstrap: str, timeout: float = 30.0) -> None:
    admin = AdminClient({"bootstrap.servers": bootstrap})
    deadline = time.time() + timeout
    while time.time() < deadline:
        meta = admin.list_topics(timeout=2)
        if meta.brokers:
            time.sleep(1.5)
            return
        time.sleep(1.0)
    raise RuntimeError(f"Kafka broker at {bootstrap} not ready after {timeout}s")


def _reset_topic(bootstrap: str, topic: str) -> None:
    """Delete and recreate the topic so there are no stale messages."""
    admin = AdminClient({"bootstrap.servers": bootstrap})
    existing = admin.list_topics(timeout=5).topics
    if topic in existing:
        futures = admin.delete_topics([topic], operation_timeout=15)
        futures[topic].result()
        # Wait for deletion to propagate
        deadline = time.time() + 15
        while time.time() < deadline:
            if topic not in admin.list_topics(timeout=2).topics:
                break
            time.sleep(0.5)
    # Recreate with a single partition so order is guaranteed
    futures = admin.create_topics([NewTopic(topic, num_partitions=1, replication_factor=1)])
    futures[topic].result()
    time.sleep(0.5)


def main() -> None:
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic = "world_a.smoke"
    group = "risk-manager-smoke-test"

    _wait_for_broker(bootstrap)
    _reset_topic(bootstrap, topic)

    producer = Producer({
        "bootstrap.servers": bootstrap,
        "client.id": "risk-manager-smoke-producer",
        "acks": "all",
        "enable.idempotence": True,
    })
    expected = [{"event_id": f"smoke-{i}", "sequence": i} for i in range(25)]
    for event in expected:
        producer.produce(topic, key=event["event_id"].encode(),
                         value=json.dumps(event).encode())
        producer.poll(0)
    producer.flush(timeout=30)

    consumer = Consumer({
        "bootstrap.servers": bootstrap,
        "group.id": group,
        "auto.offset.reset": "earliest",
    })
    consumer.subscribe([topic])
    received = []
    deadline = time.time() + 15
    try:
        while len(received) < len(expected) and time.time() < deadline:
            msg = consumer.poll(1)
            if msg is None:
                continue
            if msg.error():
                raise RuntimeError(msg.error())
            received.append(json.loads(msg.value()))
    finally:
        consumer.close()

    assert len(received) == len(expected), f"Got {len(received)}/{len(expected)} messages"
    assert received == expected, "Messages received out of order"
    print("Kafka smoke test passed: 25/25 messages, order preserved.")


if __name__ == "__main__":
    main()