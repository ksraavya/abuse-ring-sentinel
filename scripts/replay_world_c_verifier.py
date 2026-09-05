from __future__ import annotations

import argparse
from pathlib import Path

from verifier.world_c_replay import WorldCReplayConfig, WorldCVerifierRunner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay frozen Temporal detector and verifier stack on World C."
    )
    parser.add_argument("--events", default="data/generated/world_c/events.jsonl")
    parser.add_argument("--artifact-dir", default="artifacts/temporal")
    parser.add_argument(
        "--output-alerts",
        default="artifacts/verifier/world_c/verification_records.jsonl",
    )
    parser.add_argument(
        "--output-summary",
        default="artifacts/verifier/world_c/replay_summary.json",
    )
    args = parser.parse_args()

    runner = WorldCVerifierRunner(
        WorldCReplayConfig(
            artifact_dir=args.artifact_dir,
            output_alerts=args.output_alerts,
            output_summary=args.output_summary,
        )
    )
    summary = runner.replay(events_path=Path(args.events))
    print("World C verifier replay complete.")
    print(f"  Transactions: {summary['transactions_replayed']:,}")
    print(f"  Alerts:       {summary['detector_alerts']:,}")
    print(f"  Alert rate:   {summary['alert_rate']:.6%}")
    print(f"  Evidence:     {summary['evidence_items']:,}")
    print(f"  Actions:      {summary['policy_actions']}")
    print(f"  Records:      {summary['outputs']['verification_records']}")


if __name__ == "__main__":
    main()
