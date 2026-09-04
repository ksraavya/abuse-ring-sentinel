from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from models.temporal import EXPECTED_FEATURE_COUNT, FEATURE_COLUMNS, load_artifact


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frozen_artifact(artifact_dir: Path) -> dict[str, Any]:
    model_path = artifact_dir / "model.lgbm"
    metadata_path = artifact_dir / "metadata.json"
    threshold_path = artifact_dir / "threshold_search.csv"
    validation_path = artifact_dir / "validation_predictions.csv"

    selection_model_path = artifact_dir / "selection_model.lgbm"
    required = [model_path, selection_model_path, metadata_path, threshold_path, validation_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Temporal World A artifact is incomplete; missing: " + ", ".join(missing)
        )

    booster, metadata = load_artifact(artifact_dir)
    if booster.num_feature() != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Frozen model has {booster.num_feature()} features; "
            f"expected {EXPECTED_FEATURE_COUNT}"
        )
    if metadata.get("feature_list") != list(FEATURE_COLUMNS):
        raise ValueError("Frozen metadata feature list does not match the locked contract")

    contract = metadata.get("feature_contract", {})
    if contract.get("total") != EXPECTED_FEATURE_COUNT:
        raise ValueError("Frozen metadata feature_contract.total is not 26")
    if contract.get("infrastructure_raw_columns_in_model") != []:
        raise ValueError("Temporal artifact unexpectedly contains raw infrastructure columns")

    boundary = metadata.get("information_boundary", {})
    if boundary.get("future_state") is not False:
        raise ValueError("Temporal artifact does not explicitly assert future_state=false")
    if boundary.get("ground_truth_visible_to_detector") is not False:
        raise ValueError("Temporal artifact does not explicitly assert GT is hidden")

    training = metadata.get("training", {})
    if training.get("world") != "world_a":
        raise ValueError("Frozen Temporal artifact was not trained on World A")

    threshold = metadata.get("threshold", {}).get("value")
    allowed_thresholds = {round(i / 100.0, 2) for i in range(1, 100)}
    if not isinstance(threshold, (int, float)) or round(float(threshold), 2) not in allowed_thresholds:
        raise ValueError(f"Frozen threshold is invalid or outside the locked grid: {threshold!r}")

    validation_metrics = metadata.get("validation_metrics")
    if not isinstance(validation_metrics, dict) or "precision" not in validation_metrics or "recall" not in validation_metrics:
        raise ValueError("Frozen metadata is missing validation metrics")

    return {
        "artifact_version": metadata.get("artifact_version"),
        "detector": metadata.get("detector"),
        "feature_count": booster.num_feature(),
        "threshold": float(threshold),
        "validation_metrics": metadata.get("validation_metrics", {}),
        "model_sha256": sha256(model_path),
        "selection_model_sha256": sha256(selection_model_path),
        "metadata_sha256": sha256(metadata_path),
        "threshold_search_sha256": sha256(threshold_path),
        "validation_predictions_sha256": sha256(validation_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and freeze the completed World A Temporal artifact."
    )
    parser.add_argument("--artifact-dir", default="artifacts/temporal")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Output manifest path; defaults to <artifact-dir>/freeze_manifest.json",
    )
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    manifest_path = (
        Path(args.manifest)
        if args.manifest is not None
        else artifact_dir / "freeze_manifest.json"
    )

    manifest = validate_frozen_artifact(artifact_dir)
    manifest["artifact_dir"] = str(artifact_dir)
    manifest["feature_list"] = list(FEATURE_COLUMNS)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("Temporal World A artifact verified and frozen.")
    print(f"  Features:  {manifest['feature_count']}")
    print(f"  Threshold: {manifest['threshold']:.2f}")
    print(f"  Model SHA: {manifest['model_sha256']}")
    print(f"  Manifest:  {manifest_path}")


if __name__ == "__main__":
    main()
