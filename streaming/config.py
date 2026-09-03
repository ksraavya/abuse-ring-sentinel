from __future__ import annotations
from dataclasses import dataclass
import os

@dataclass(frozen=True)
class KafkaConfig:
    bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    client_id: str = os.getenv("KAFKA_CLIENT_ID", "risk-manager")
    topic_prefix: str = os.getenv("KAFKA_TOPIC_PREFIX", "")
    acks: str = os.getenv("KAFKA_ACKS", "all")

    def topic(self, world_id: str) -> str:
        return f"{self.topic_prefix}{world_id}.events"
