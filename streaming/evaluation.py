from __future__ import annotations

import time
from pathlib import Path

from streaming.config import KafkaConfig


EVALUATION_WORLD = "world_b"
EVALUATION_PARTITIONS = 1


def _admin_client(config: KafkaConfig):
    # Import lazily so pure unit tests and offline rubric code do not require a
    # running Kafka installation merely to import this module.
    from confluent_kafka.admin import AdminClient

    return AdminClient({"bootstrap.servers": config.bootstrap_servers})


def _wait_for_topic_state(admin, topic: str, exists: bool, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        topics = admin.list_topics(timeout=2).topics
        if (topic in topics) == exists:
            return
        time.sleep(0.5)
    state = "exist" if exists else "be deleted"
    raise RuntimeError(f"Kafka topic {topic!r} did not {state} within {timeout}s")


def reset_evaluation_topic(config: KafkaConfig | None = None) -> str:
    """Reset World B to one partition so chronological causal order is global.

    The local Docker broker defaults to multiple partitions. A multi-partition
    World B topic would permit consumers to observe events from different
    partitions in an order that is not globally chronological, which is not
    acceptable for the event-time Temporal detector. The held-out evaluation
    therefore uses one partition deliberately; production-scale partitioning
    requires an explicit event-time ordering/watermark design and is outside
    this experiment.
    """
    cfg = config or KafkaConfig()
    topic = cfg.topic(EVALUATION_WORLD)
    admin = _admin_client(cfg)

    existing = admin.list_topics(timeout=5).topics
    if topic in existing:
        futures = admin.delete_topics([topic], operation_timeout=30)
        futures[topic].result()
        _wait_for_topic_state(admin, topic, exists=False)

    from confluent_kafka.admin import NewTopic

    futures = admin.create_topics(
        [NewTopic(topic, num_partitions=EVALUATION_PARTITIONS, replication_factor=1)]
    )
    futures[topic].result()
    _wait_for_topic_state(admin, topic, exists=True)
    return topic


def replay_world_b_once(
    events_path: str | Path,
    config: KafkaConfig | None = None,
    *,
    rate: float = 0.0,
) -> int:
    """Reset the held-out topic and publish World B exactly once.

    The canonical event payload contains no ground truth. Independent model
    consumer groups can subsequently read the same retained topic from the
    beginning, making Baseline A/B and Temporal evaluation uniform.
    """
    from streaming.producer import WorldReplayProducer

    cfg = config or KafkaConfig()
    reset_evaluation_topic(cfg)
    return WorldReplayProducer(cfg).replay(
        events_path=events_path,
        world_id=EVALUATION_WORLD,
        rate=rate,
    )
