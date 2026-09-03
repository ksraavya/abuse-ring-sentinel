from __future__ import annotations
import argparse
import json
from confluent_kafka import Consumer, KafkaException
from streaming.config import KafkaConfig

def consume(world_id: str, group_id: str, max_messages: int | None = None,
            timeout: float = 2.0) -> list[dict]:
    cfg = KafkaConfig()
    consumer = Consumer({
        "bootstrap.servers": cfg.bootstrap_servers,
        "group.id": group_id,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
    })
    consumer.subscribe([cfg.topic(world_id)])
    messages = []
    try:
        while max_messages is None or len(messages) < max_messages:
            msg = consumer.poll(timeout)
            if msg is None:
                break
            if msg.error():
                raise KafkaException(msg.error())
            messages.append(json.loads(msg.value().decode("utf-8")))
    finally:
        consumer.close()
    return messages

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", choices=["world_a", "world_b"], required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--max-messages", type=int, default=20)
    args = parser.parse_args()
    messages = consume(args.world, args.group, args.max_messages)
    for message in messages:
        print(json.dumps(message, indent=2, sort_keys=True))
    print(f"Consumed {len(messages):,} messages")

if __name__ == "__main__":
    main()
