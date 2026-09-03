from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np
import yaml

from events.schema import EventRecord
from .behavior import GeneratedWorld, generate_world, write_world
from .population import build_hard_negative_clusters, build_legitimate_population
from .rings import build_ring_plans
from .schema import WorldConfig


class WorldGenerator:
    START_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __init__(self, config: WorldConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def _build_components(self):
        prefix = self.config.world_id.value
        legitimate = build_legitimate_population(
            self.rng, prefix, self.config.legitimate_accounts, self.START_TIME,
            self.config.duration_days, activity_sigma=self.config.activity_sigma,
            activity_min_weight=self.config.activity_min_weight,
        )
        counts = {
            "family": self.config.hard_negative_family_clusters,
            "hostel": self.config.hard_negative_hostel_clusters,
            "corporate": self.config.hard_negative_corporate_clusters,
        }
        hard_negatives, clusters = build_hard_negative_clusters(
            self.rng, prefix, len(legitimate), counts, self.START_TIME, self.config.duration_days,
        )
        ring_count = self.config.total_rings
        ring_capacity = max(1, ring_count * self.config.ring_size_max)
        ring_accounts = build_legitimate_population(
            self.rng, prefix + "-ring", ring_capacity, self.START_TIME,
            self.config.duration_days, cohort="ring", activity_sigma=self.config.activity_sigma,
            activity_min_weight=self.config.activity_min_weight,
        )
        plans = build_ring_plans(self.rng, prefix, self.config, ring_accounts, start_day=0.5, start=self.START_TIME)
        adjusted = {a.account_id: a for plan in plans for a in plan.accounts}
        ring_accounts = [adjusted[a.account_id] for a in ring_accounts if a.account_id in adjusted]
        return legitimate, hard_negatives, ring_accounts, clusters, plans

    def generate(self) -> list[EventRecord]:
        parts = self._build_components()
        return generate_world(
            rng=self.rng, world_id=self.config.world_id, start=self.START_TIME, config=self.config,
            legitimate_accounts=parts[0], hard_negative_accounts=parts[1], ring_accounts=parts[2],
            hard_negative_clusters=parts[3], ring_plans=parts[4],
        ).records

    def generate_with_manifest(self) -> GeneratedWorld:
        parts = self._build_components()
        return generate_world(
            rng=self.rng, world_id=self.config.world_id, start=self.START_TIME, config=self.config,
            legitimate_accounts=parts[0], hard_negative_accounts=parts[1], ring_accounts=parts[2],
            hard_negative_clusters=parts[3], ring_plans=parts[4],
        )

    def generate_to_disk(self, output_dir: Path) -> dict:
        parts = self._build_components()
        return write_world(
            rng=self.rng, world_id=self.config.world_id, start=self.START_TIME, config=self.config,
            legitimate_accounts=parts[0], hard_negative_accounts=parts[1], ring_accounts=parts[2],
            hard_negative_clusters=parts[3], ring_plans=parts[4], output_dir=output_dir,
        )

    @staticmethod
    def write_jsonl(records: list[EventRecord], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.event.model_dump(mode="json"), separators=(",", ":")) + "\n")

    @staticmethod
    def write_ground_truth(records: list[EventRecord], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                if record.ground_truth is not None:
                    handle.write(json.dumps(record.ground_truth.model_dump(mode="json"), separators=(",", ":")) + "\n")


def load_config(path: Path, world_id: str) -> WorldConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    section = raw.get(world_id, raw)
    return WorldConfig(world_id=world_id, **section)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one complete synthetic world")
    parser.add_argument("--world", choices=["world_a", "world_b"], required=True)
    parser.add_argument("--config-dir", type=Path, default=Path("config"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated"))
    args = parser.parse_args()

    cfg = load_config(args.config_dir / f"{args.world}.yaml", args.world)
    manifest = WorldGenerator(cfg).generate_to_disk(args.output_dir / args.world)
    print(f"Generated {manifest['event_count']:,} events for {args.world}")
    print(f"Accounts: {manifest['account_count']:,}")
    print(f"Transactions: {manifest['transaction_count']:,}")
    print(f"Fraud transactions: {manifest['fraud_transaction_count']:,}")
    print(f"Fraud rate: {manifest['fraud_transaction_count'] / max(1, manifest['transaction_count']):.4%}")
    print(f"Account updates: {manifest['account_update_count']:,}")
    print(f"Output: {(args.output_dir / args.world).resolve()}")


if __name__ == "__main__":
    main()
