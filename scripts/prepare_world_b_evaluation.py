from __future__ import annotations

import argparse
from pathlib import Path

from streaming.evaluation import replay_world_b_once


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset the held-out World B Kafka topic and publish World B once."
    )
    parser.add_argument(
        "--events",
        default="data/generated/world_b/events.jsonl",
        help="Path to the canonical World B events JSONL.",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=0.0,
        help="Replay rate in events/second. 0 means as fast as possible.",
    )
    args = parser.parse_args()
    if args.rate < 0:
        parser.error("--rate must be greater than or equal to 0")

    count = replay_world_b_once(Path(args.events), rate=args.rate)
    print("World B evaluation stream prepared.")
    print("  Topic:     world_b.events")
    print("  Partitions: 1")
    print(f"  Events:    {count:,}")
    print("  Ground truth: not published to Kafka")
    print("  Next step: start model-specific consumers with independent groups.")


if __name__ == "__main__":
    main()
