from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from graph.temporal_replay import TemporalReplay, TemporalReplayState
from models.temporal import EXPECTED_FEATURE_COUNT, FEATURE_COLUMNS, load_artifact

from .context_investigator import ContextInvestigator
from .contracts import EvidenceBundle, EvidenceContext, EvidenceItem, VerificationRequest
from .evidence_fusion import DeterministicEvidenceFusion, FusionBreakdown
from .infrastructure_investigator import InfrastructureInvestigator
from .policy import AutoResponderPolicy, PolicyDecision
from .ring_investigator import RingInvestigator


@dataclass(frozen=True)
class WorldCReplayConfig:
    """Development-only wiring for chronological World C verifier replay.

    The detector artifact and threshold are loaded from the already-frozen
    World A Temporal artifact. Investigator/fusion/policy parameters are
    supplied by their normal constructors and may be changed during 12A/12B.
    World D is never accepted by this runner.
    """

    world: str = "world_c"
    detector_model_name: str = "temporal-world-a-frozen"
    verifier_version: str = "12a-world-c-dev-v1"
    output_alerts: str = "artifacts/verifier/world_c/verification_records.jsonl"
    output_summary: str = "artifacts/verifier/world_c/replay_summary.json"
    artifact_dir: str = "artifacts/temporal"

    def __post_init__(self) -> None:
        if self.world != "world_c":
            raise ValueError("WorldCReplayConfig only permits world_c")
        if not self.verifier_version.strip():
            raise ValueError("verifier_version must be non-empty")


@dataclass(frozen=True)
class VerificationRecord:
    """Serializable result for one detector alert, with no ground truth."""

    event_id: str
    timestamp: str
    account_id: str
    detector_probability: float
    detector_threshold: float
    detector_model: str
    verifier_version: str
    evidence: tuple[dict[str, Any], ...]
    fusion: dict[str, Any]
    verification_confidence: float
    policy: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "account_id": self.account_id,
            "detector_probability": self.detector_probability,
            "detector_threshold": self.detector_threshold,
            "detector_model": self.detector_model,
            "verifier_version": self.verifier_version,
            "evidence": list(self.evidence),
            "fusion": self.fusion,
            "verification_confidence": self.verification_confidence,
            "policy": self.policy,
        }


class WorldCVerifierRunner:
    """Replay World C through the frozen detector and live verifier stack.

    Ground truth is deliberately not loaded or required. TemporalReplay invokes
    the callback at the causal pre-update boundary, so all investigators see
    only state strictly before the current transaction. Alert records contain
    verifier outputs only; evaluation labels are joined later by a separate
    evaluation process.
    """

    def __init__(
        self,
        config: WorldCReplayConfig | None = None,
        *,
        ring_investigator: RingInvestigator | None = None,
        infrastructure_investigator: InfrastructureInvestigator | None = None,
        context_investigator: ContextInvestigator | None = None,
        fusion: DeterministicEvidenceFusion | None = None,
        policy: AutoResponderPolicy | None = None,
    ) -> None:
        self.config = config or WorldCReplayConfig()
        self.ring_investigator = ring_investigator or RingInvestigator()
        self.infrastructure_investigator = infrastructure_investigator or InfrastructureInvestigator()
        self.context_investigator = context_investigator or ContextInvestigator()
        self.fusion = fusion or DeterministicEvidenceFusion()
        self.policy = policy or AutoResponderPolicy()

    def replay(
        self,
        *,
        events_path: Path,
        output_alerts: Path | None = None,
        output_summary: Path | None = None,
        on_alert: Callable[[VerificationRecord], None] | None = None,
    ) -> dict[str, Any]:
        if not events_path.is_file():
            raise FileNotFoundError(events_path)

        model, metadata, model_sha = self._load_frozen_detector(Path(self.config.artifact_dir))
        threshold = float(metadata["threshold"]["value"])
        model_name = self.config.detector_model_name

        alerts_path = Path(output_alerts or self.config.output_alerts)
        summary_path = Path(output_summary or self.config.output_summary)
        alerts_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        replay = TemporalReplay(TemporalReplayState())
        transactions = 0
        alerts = 0
        evidence_items = 0
        evidence_by_type: dict[str, int] = {}
        actions = {"allow": 0, "review": 0, "block": 0}
        max_detector = 0.0

        with alerts_path.open("w", encoding="utf-8") as output:
            # We need the typed current event inside the callback while keeping
            # TemporalReplay's pre-update boundary. This helper installs the
            # actual callback per transaction below.
            with events_path.open("r", encoding="utf-8") as handle:
                for _line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    raw = json.loads(line)
                    if raw.get("event_type") != "transaction":
                        replay.process_event(raw)
                        continue

                    event = replay._parse_event(raw)

                    def score_current(row: tuple[float, ...], event=event) -> None:
                        nonlocal alerts, evidence_items, max_detector
                        if len(row) != EXPECTED_FEATURE_COUNT:
                            raise RuntimeError(
                                f"Temporal replay produced {len(row)} features; expected {EXPECTED_FEATURE_COUNT}"
                            )
                        probability = float(
                            model.predict(np.asarray([row], dtype=np.float32))[0]
                        )
                        max_detector = max(max_detector, probability)
                        if probability < threshold:
                            return

                        request = VerificationRequest(
                            alert_event=event,
                            detector_probability=probability,
                            detector_threshold=threshold,
                            detector_model=model_name,
                            alerted_at=event.timestamp,
                        )
                        context = EvidenceContext(
                            as_of=event.timestamp,
                            state={
                                "infrastructure_state": replay.state.infrastructure,
                                "behavioral_state": replay.state.behavioral,
                                "temporal_feature_state": replay.state.temporal_features,
                            },
                        )

                        # All three investigators run before TemporalReplay
                        # commits this transaction to state.
                        items: list[EvidenceItem] = []
                        items.extend(self.ring_investigator.collect(request, context))
                        items.extend(self.infrastructure_investigator.collect(request, context))
                        items.extend(self.context_investigator.collect(request, context))

                        breakdown: FusionBreakdown = self.fusion.explain(request, items)
                        confidence = breakdown.fused_confidence
                        decision: PolicyDecision = self.policy.decide(
                            request, EvidenceBundle(
                                alert_event_id=event.event_id,
                                decision_time=request.decision_time,
                                verifier_version=self.config.verifier_version,
                                items=tuple(items),
                            ), confidence,
                        )

                        record = VerificationRecord(
                            event_id=event.event_id,
                            timestamp=event.timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                            account_id=str(event.account_id),
                            detector_probability=probability,
                            detector_threshold=threshold,
                            detector_model=model_name,
                            verifier_version=self.config.verifier_version,
                            evidence=tuple(item.model_dump(mode="json") for item in items),
                            fusion=self._fusion_dict(breakdown),
                            verification_confidence=confidence,
                            policy=self._policy_dict(decision),
                        )
                        output.write(json.dumps(record.to_dict(), separators=(",", ":")) + "\n")
                        alerts += 1
                        evidence_items += len(items)
                        for item in items:
                            key = item.evidence_type.value
                            evidence_by_type[key] = evidence_by_type.get(key, 0) + 1
                        actions[decision.action.value] += 1
                        if on_alert is not None:
                            on_alert(record)

                    replay.process_event(raw, score_callback=score_current)
                    transactions += 1
                    if transactions % 100_000 == 0:
                        print(f"  World C transactions replayed: {transactions:,}")

        summary = {
            "world": "world_c",
            "purpose": "verifier/responder development",
            "transactions_replayed": transactions,
            "detector_alerts": alerts,
            "alert_rate": alerts / transactions if transactions else 0.0,
            "average_evidence_items_per_alert": evidence_items / alerts if alerts else 0.0,
            "evidence_items": evidence_items,
            "evidence_by_type": dict(sorted(evidence_by_type.items())),
            "policy_actions": actions,
            "max_detector_probability": max_detector,
            "frozen_detector": {
                "artifact_dir": str(Path(self.config.artifact_dir)),
                "model_sha256": model_sha,
                "threshold": threshold,
                "feature_count": len(FEATURE_COLUMNS),
                "feature_list": list(FEATURE_COLUMNS),
                "trained_world": metadata.get("training", {}).get("world"),
            },
            "causal_contract": {
                "ground_truth_loaded_by_replay": False,
                "read_pre_event_state": True,
                "score_before_current_transaction_state_update": True,
                "transaction_mutates_infrastructure_state": False,
                "raw_infrastructure_columns_in_temporal_model": [],
            },
            "outputs": {
                "verification_records": str(alerts_path),
                "summary": str(summary_path),
            },
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary

    @staticmethod
    def _fusion_dict(breakdown: FusionBreakdown) -> dict[str, Any]:
        return {
            "detector_probability": breakdown.detector_probability,
            "evidence_support": breakdown.evidence_support,
            "agent_coverage": breakdown.agent_coverage,
            "coverage_bonus": breakdown.coverage_bonus,
            "evidence_score": breakdown.evidence_score,
            "fused_confidence": breakdown.fused_confidence,
            "strongest_by_type": {key.value: value for key, value in breakdown.strongest_by_type.items()},
            "evidence_count": breakdown.evidence_count,
            "contributing_agent_names": list(breakdown.contributing_agent_names),
        }

    @staticmethod
    def _policy_dict(decision: PolicyDecision) -> dict[str, Any]:
        return {
            "action": decision.action.value,
            "risk_tier": decision.risk_tier.value,
            "policy_version": decision.policy_version,
            "agent_coverage": decision.agent_coverage,
            "evidence_count": decision.evidence_count,
            "strong_evidence_count": decision.strong_evidence_count,
            "reason_codes": list(decision.reason_codes),
            "rationale": decision.rationale,
        }

    @staticmethod
    def _load_frozen_detector(artifact_dir: Path):
        model_path = artifact_dir / "model.lgbm"
        metadata_path = artifact_dir / "metadata.json"
        freeze_path = artifact_dir / "freeze_manifest.json"
        for path in (model_path, metadata_path, freeze_path):
            if not path.is_file():
                raise FileNotFoundError(f"Frozen Temporal artifact is incomplete; missing {path}")

        model, metadata = load_artifact(artifact_dir)
        if model.num_feature() != EXPECTED_FEATURE_COUNT:
            raise ValueError("Frozen Temporal model feature count is not 26")
        if tuple(metadata.get("feature_list", ())) != FEATURE_COLUMNS:
            raise ValueError("Frozen Temporal feature list does not match the locked contract")
        if metadata.get("training", {}).get("world") != "world_a":
            raise ValueError("12A detector must remain frozen on World A")
        threshold = metadata.get("threshold", {}).get("value")
        if not isinstance(threshold, (int, float)) or not 0.01 <= float(threshold) <= 0.99:
            raise ValueError(f"Frozen Temporal threshold is invalid: {threshold!r}")
        contract = metadata.get("feature_contract", {})
        if contract.get("total") != EXPECTED_FEATURE_COUNT:
            raise ValueError("Frozen Temporal feature contract total is not 26")
        if contract.get("infrastructure_raw_columns_in_model") != []:
            raise ValueError("Temporal model contains raw infrastructure columns")
        boundary = metadata.get("information_boundary", {})
        if boundary.get("future_state") is not False or boundary.get("ground_truth_visible_to_detector") is not False:
            raise ValueError("Frozen Temporal artifact violates the information boundary")

        manifest = json.loads(freeze_path.read_text(encoding="utf-8"))
        model_sha = _sha256(model_path)
        if manifest.get("model_sha256") != model_sha:
            raise ValueError("Frozen Temporal model SHA does not match freeze manifest")
        if float(manifest.get("threshold")) != float(threshold):
            raise ValueError("Frozen Temporal threshold does not match freeze manifest")
        if manifest.get("feature_list") != list(FEATURE_COLUMNS):
            raise ValueError("Freeze manifest feature list does not match locked contract")
        return model, metadata, model_sha


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["VerificationRecord", "WorldCReplayConfig", "WorldCVerifierRunner"]
