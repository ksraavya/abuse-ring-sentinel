from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


WORLDS = ("world_c", "world_d")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_and_freeze(root: Path, world: str, output_root: Path, config_root: Path) -> None:
    output_dir = output_root / world
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Refusing to overwrite existing {world} at {output_dir}. "
            "World C/D are frozen experiment inputs. Remove the directory only if you intentionally restart the experiment."
        )

    subprocess.run(
        [sys.executable, "-m", "worlds.generator", "--world", world, "--config-dir", str(config_root), "--output-dir", str(output_root)],
        cwd=root,
        check=True,
    )

    files = {
        "events.jsonl": sha256_file(output_dir / "events.jsonl"),
        "ground_truth.jsonl": sha256_file(output_dir / "ground_truth.jsonl"),
        "manifest.json": sha256_file(output_dir / "manifest.json"),
        "config.yaml": sha256_file(config_root / f"{world}.yaml"),
    }
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    freeze = {
        "freeze_version": 1,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "world_id": world,
        "seed": manifest["seed"],
        "duration_days": manifest["duration_days"],
        "event_count": manifest["event_count"],
        "transaction_count": manifest["transaction_count"],
        "account_count": manifest["account_count"],
        "fraud_transaction_count": manifest["fraud_transaction_count"],
        "ring_count": manifest["ring_count"],
        "files_sha256": files,
        "ground_truth_separate_from_events": True,
        "purpose": "verifier/responder development" if world == "world_c" else "final end-to-end held-out evaluation",
    }
    (output_dir / "freeze_manifest.json").write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen {world}: {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and freeze the independent World C and World D datasets")
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--config-dir", type=Path, default=Path("config"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    for world in WORLDS:
        generate_and_freeze(root, world, args.output_dir, args.config_dir)


if __name__ == "__main__":
    main()
