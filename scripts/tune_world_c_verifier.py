from __future__ import annotations

import argparse

from verifier.world_c_tuning import main_cli


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune and freeze verifier/policy on World C.")
    parser.add_argument("--records", default="artifacts/verifier/world_c/verification_records.jsonl")
    parser.add_argument("--ground-truth", default="data/generated/world_c/ground_truth.jsonl")
    parser.add_argument("--manifest", default="data/generated/world_c/manifest.json")
    parser.add_argument("--output-dir", default="artifacts/verifier/world_c/tuning")
    parser.add_argument("--min-block-precision", type=float, default=0.70)
    parser.add_argument("--events", default="data/generated/world_c/events.jsonl")
    args = parser.parse_args()
    main_cli(args)


if __name__ == "__main__":
    main()