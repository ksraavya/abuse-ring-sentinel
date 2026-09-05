from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from events.schema import AccountCreatedEvent, AccountUpdatedEvent, TransactionEvent
from graph.temporal_replay import TemporalReplay, TemporalReplayState
from models.temporal import EXPECTED_FEATURE_COUNT, FEATURE_COLUMNS, load_artifact
from verifier.action_execution import ActionExecutor, ActionExecutionReceipt, InMemoryActionExecutionBackend
from verifier.audit_trail import AuditTrailRecorder, InMemoryAuditTrail
from verifier.context_investigator import ContextInvestigator
from verifier.contracts import EvidenceBundle, EvidenceContext, EvidenceItem, EvidenceStrength, EvidenceType, VerificationRequest
from verifier.evidence_fusion import (
    DEFAULT_STRENGTH_MULTIPLIERS,
    DEFAULT_TYPE_WEIGHTS,
    DeterministicEvidenceFusion,
    EvidenceFusionConfig,
    FusionBreakdown,
)
from verifier.infrastructure_investigator import InfrastructureInvestigator
from verifier.policy import AutoResponderPolicy, AutoResponderPolicyConfig, PolicyAction, PolicyDecision
from verifier.ring_investigator import RingInvestigator


MODEL_NAME = "temporal-world-a-frozen"
EXPECTED_PARTITIONS = 1
FALSE_POSITIVE_COST = 500.0
FALSE_NEGATIVE_COST = 5000.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_time(value: Any) -> datetime:
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return dt.astimezone(timezone.utc)


def _load_manifest(path: Path) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rings: dict[str, set[str]] = {}
    account_to_rings: dict[str, set[str]] = {}
    for entry in raw.get("rings", []):
        ring_id = str(entry.get("ring_id", ""))
        members = entry.get("account_ids")
        if not ring_id or not isinstance(members, list):
            raise ValueError("World D manifest contains an invalid ring entry")
        if ring_id in rings:
            raise ValueError(f"Duplicate World D ring_id: {ring_id}")
        member_set = {str(x) for x in members}
        rings[ring_id] = member_set
        for account_id in member_set:
            account_to_rings.setdefault(account_id, set()).add(ring_id)
    if not rings:
        raise ValueError("World D manifest contains no rings")
    return rings, account_to_rings


def _load_frozen_detector(artifact_dir: Path) -> tuple[Any, dict[str, Any], str]:
    model_path = artifact_dir / "model.lgbm"
    metadata_path = artifact_dir / "metadata.json"
    freeze_path = artifact_dir / "freeze_manifest.json"
    for path in (model_path, metadata_path, freeze_path):
        if not path.is_file():
            raise FileNotFoundError(f"Frozen Temporal artifact incomplete; missing {path}")

    model, metadata = load_artifact(artifact_dir)
    if model.num_feature() != EXPECTED_FEATURE_COUNT:
        raise ValueError("Frozen Temporal model does not contain exactly 26 features")
    if tuple(metadata.get("feature_list", ())) != FEATURE_COLUMNS:
        raise ValueError("Frozen Temporal metadata feature list does not match locked contract")
    if metadata.get("training", {}).get("world") != "world_a":
        raise ValueError("World D must use the Temporal model frozen from World A")

    threshold = metadata.get("threshold", {}).get("value")
    if not isinstance(threshold, (int, float)) or not 0.0 < float(threshold) < 1.0:
        raise ValueError(f"Invalid frozen detector threshold: {threshold!r}")

    contract = metadata.get("feature_contract", {})
    if contract.get("total") != EXPECTED_FEATURE_COUNT:
        raise ValueError("Frozen Temporal feature contract is not 26")
    if contract.get("infrastructure_raw_columns_in_model") != []:
        raise ValueError("Temporal classifier unexpectedly contains raw infrastructure columns")

    boundary = metadata.get("information_boundary", {})
    if boundary.get("future_state") is not False:
        raise ValueError("Frozen Temporal artifact does not assert future_state=false")
    if boundary.get("ground_truth_visible_to_detector") is not False:
        raise ValueError("Frozen Temporal artifact does not assert GT-hidden detector input")

    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    model_sha = _sha256(model_path)
    if freeze.get("model_sha256") != model_sha:
        raise ValueError("Temporal model SHA does not match freeze manifest")
    if float(freeze.get("threshold")) != float(threshold):
        raise ValueError("Temporal threshold does not match freeze manifest")
    if freeze.get("feature_list") != list(FEATURE_COLUMNS):
        raise ValueError("Temporal freeze-manifest feature list does not match contract")

    return model, metadata, model_sha


def _load_frozen_verifier(path: Path) -> tuple[DeterministicEvidenceFusion, AutoResponderPolicy, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Frozen World C verifier config not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("world") != "world_c":
        raise ValueError("World D verifier must load the World C frozen configuration")
    if raw.get("purpose") != "verifier_and_policy_development_then_freeze":
        raise ValueError("Verifier config is not the expected World C freeze artifact")
    if raw.get("ground_truth_used_only_for_selection") is not True:
        raise ValueError("World C freeze must record GT as selection-only")

    fusion_raw = raw.get("fusion", {})
    type_weights = {
        EvidenceType(key): float(value)
        for key, value in fusion_raw.get("type_weights", {}).items()
    }
    strength_multipliers = {
        EvidenceStrength(key): float(value)
        for key, value in fusion_raw.get("strength_multipliers", {}).items()
    }
    fusion = EvidenceFusionConfig(
        detector_weight=float(fusion_raw["detector_weight"]),
        coverage_bonus=float(fusion_raw["coverage_bonus"]),
        expected_agent_names=tuple(fusion_raw["expected_agent_names"]),
        type_weights=type_weights,
        strength_multipliers=strength_multipliers,
    )

    policy_raw = raw.get("policy", {})
    policy = AutoResponderPolicyConfig(
        detector_alert_threshold=float(policy_raw["detector_alert_threshold"]),
        block_detector_threshold=float(policy_raw["block_detector_threshold"]),
        block_verifier_threshold=float(policy_raw["block_verifier_threshold"]),
        review_verifier_threshold=float(policy_raw["review_verifier_threshold"]),
        min_block_agent_coverage=int(policy_raw["min_block_agent_coverage"]),
        require_strong_evidence_for_block=bool(policy_raw["require_strong_evidence_for_block"]),
        policy_version=str(policy_raw["policy_version"]),
    )
    return DeterministicEvidenceFusion(fusion), AutoResponderPolicy(policy), raw


def _read_truth_row(handle, line_number: int) -> tuple[str, bool, str | None]:
    line = handle.readline()
    if not line:
        raise RuntimeError(
            f"World D ground truth ended before transaction {line_number}"
        )
    raw = json.loads(line)
    event_id = str(raw.get("event_id", ""))
    if not event_id:
        raise ValueError(f"World D ground-truth line {line_number} has no event_id")
    return event_id, bool(raw["is_fraud"]), (
        str(raw["ring_id"]) if raw.get("ring_id") is not None else None
    )


def _confusion(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=bool)
    tp = int(np.sum(predictions & (labels == 1)))
    fp = int(np.sum(predictions & (labels == 0)))
    tn = int(np.sum(~predictions & (labels == 0)))
    fn = int(np.sum(~predictions & (labels == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0
    return {
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "fnr": fnr,
        "economic_cost": FALSE_POSITIVE_COST * fp + FALSE_NEGATIVE_COST * fn,
    }


def _lead_time_stats(
    first_fraud: dict[str, datetime],
    first_detection: dict[str, datetime],
) -> tuple[int, dict[str, float | None]]:
    lead_times = sorted(
        (first_fraud[r] - first_detection[r]).total_seconds() / 86400.0
        for r in first_fraud
        if r in first_detection and first_detection[r] < first_fraud[r]
    )
    if not lead_times:
        return 0, {
            "mean_days": None,
            "median_days": None,
            "min_days": None,
            "max_days": None,
        }
    return len(lead_times), {
        "mean_days": float(np.mean(lead_times)),
        "median_days": float(np.median(lead_times)),
        "min_days": float(min(lead_times)),
        "max_days": float(max(lead_times)),
    }


def _ring_metrics(
    fraud_rings: set[str],
    detected_rings: set[str],
    first_fraud: dict[str, datetime],
    first_detection: dict[str, datetime],
) -> dict[str, Any]:
    pre_count, lead = _lead_time_stats(first_fraud, first_detection)
    return {
        "fraud_bearing_rings": len(fraud_rings),
        "detected_rings": len(detected_rings & fraud_rings),
        "ring_detection_recall": (
            len(detected_rings & fraud_rings) / len(fraud_rings) if fraud_rings else 0.0
        ),
        "rings_with_abuse": len(first_fraud),
        "rings_detected_pre_abuse": pre_count,
        "pre_abuse_detection_recall": pre_count / len(first_fraud) if first_fraud else 0.0,
        "lead_time_days_mean": lead["mean_days"],
        "lead_time_days_median": lead["median_days"],
        "lead_time_days_min": lead["min_days"],
        "lead_time_days_max": lead["max_days"],
    }


def evaluate_world_d(
    *,
    events_path: Path,
    ground_truth_path: Path,
    manifest_path: Path,
    artifact_dir: Path,
    verifier_config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    for path in (events_path, ground_truth_path, manifest_path, artifact_dir, verifier_config_path):
        if not path.exists():
            raise FileNotFoundError(path)

    model, metadata, model_sha = _load_frozen_detector(artifact_dir)
    threshold = float(metadata["threshold"]["value"])
    fusion, policy, verifier_freeze = _load_frozen_verifier(verifier_config_path)
    rings, account_to_rings = _load_manifest(manifest_path)

    # These are all computed in one chronological replay. Ground truth is read
    # only after the detector/verifier/responder path has completed for the
    # current transaction, so labels cannot influence any decision path.
    replay = TemporalReplay(TemporalReplayState())
    execution_backend = InMemoryActionExecutionBackend()
    executor = ActionExecutor(execution_backend)
    audit_trail = InMemoryAuditTrail()
    audit = AuditTrailRecorder(audit_trail)

    ring_investigator = RingInvestigator()
    infrastructure_investigator = InfrastructureInvestigator()
    context_investigator = ContextInvestigator()

    with ground_truth_path.open("r", encoding="utf-8") as count_handle:
        expected = sum(1 for line in count_handle if line.strip())
    if expected == 0:
        raise ValueError("World D ground truth is empty")

    probabilities = np.empty(expected, dtype=np.float32)
    labels = np.empty(expected, dtype=np.int8)
    detector_predictions = np.zeros(expected, dtype=bool)
    block_predictions = np.zeros(expected, dtype=bool)

    detector_tp = detector_fp = detector_tn = detector_fn = 0
    block_tp = block_fp = block_tn = block_fn = 0
    transactions = alerts = account_events = evidence_items = 0
    action_counts = {"allow": 0, "review": 0, "block": 0}
    execution_status_counts = {"executed": 0, "idempotent_replay": 0}
    evidence_by_type: dict[str, int] = {}
    verifier_confidence_sum = 0.0
    blocked_fraud_amount = 0.0
    detector_alerted_fraud_amount = 0.0
    fraud_exposure = 0.0
    first_fraud: dict[str, datetime] = {}
    detector_detected_rings: set[str] = set()
    detector_first_ring_member_alert: dict[str, datetime] = {}
    blocked_ring_ids: set[str] = set()
    block_first_ring_member: dict[str, datetime] = {}
    fraud_rings: set[str] = set()
    seen_truth_ids: set[str] = set()

    started = time.perf_counter()
    with events_path.open("r", encoding="utf-8") as events_handle, ground_truth_path.open(
        "r", encoding="utf-8"
    ) as gt_handle:
        truth_line = 0
        for line_number, line in enumerate(events_handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            event_type = raw.get("event_type")

            if event_type == "account_created":
                replay.process_event(raw)
                account_events += 1
                continue
            if event_type == "account_updated":
                replay.process_event(raw)
                account_events += 1
                continue
            if event_type != "transaction":
                raise ValueError(f"Unknown World D event_type {event_type!r}")

            event = TransactionEvent.model_validate(raw)
            event_timestamp = _parse_time(raw["timestamp"])
            event_id = event.event_id

            def score_current(row: tuple[float, ...]) -> None:
                nonlocal transactions, alerts, evidence_items
                nonlocal detector_tp, detector_fp, detector_tn, detector_fn
                nonlocal block_tp, block_fp, block_tn, block_fn
                nonlocal verifier_confidence_sum, blocked_fraud_amount
                nonlocal detector_alerted_fraud_amount, fraud_exposure
                nonlocal truth_line

                if len(row) != EXPECTED_FEATURE_COUNT:
                    raise RuntimeError(
                        f"Temporal replay produced {len(row)} features; expected {EXPECTED_FEATURE_COUNT}"
                    )

                # IMPORTANT: everything below the detector score is still
                # label-blind. Ground truth is joined only after this path has
                # produced its decision and execution result.
                probability = float(model.predict(np.asarray([row], dtype=np.float32))[0])
                predicted = probability >= threshold
                idx = transactions
                probabilities[idx] = probability
                detector_predictions[idx] = predicted

                if predicted:
                    for ring_id in account_to_rings.get(event.account_id, set()):
                        detector_detected_rings.add(ring_id)
                        prior = detector_first_ring_member_alert.get(ring_id)
                        if prior is None or event_timestamp < prior:
                            detector_first_ring_member_alert[ring_id] = event_timestamp
                    alerts += 1

                    request = VerificationRequest(
                        alert_event=event,
                        detector_probability=probability,
                        detector_threshold=threshold,
                        detector_model=MODEL_NAME,
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
                    items: list[EvidenceItem] = []
                    items.extend(ring_investigator.collect(request, context))
                    items.extend(infrastructure_investigator.collect(request, context))
                    items.extend(context_investigator.collect(request, context))
                    evidence_items += len(items)
                    for item in items:
                        evidence_by_type[item.evidence_type.value] = evidence_by_type.get(
                            item.evidence_type.value, 0
                        ) + 1

                    bundle = EvidenceBundle(
                        alert_event_id=event.event_id,
                        decision_time=event.timestamp,
                        verifier_version=str(verifier_freeze["policy"]["policy_version"]),
                        items=tuple(items),
                    )
                    breakdown: FusionBreakdown = fusion.explain(request, items)
                    confidence = breakdown.fused_confidence
                    verifier_confidence_sum += confidence
                    decision = policy.decide(request, bundle, confidence)
                    action_counts[decision.action.value] += 1

                    if decision.action is PolicyAction.BLOCK:
                        block_predictions[idx] = True
                        for ring_id in account_to_rings.get(event.account_id, set()):
                            blocked_ring_ids.add(ring_id)
                            prior = block_first_ring_member.get(ring_id)
                            if prior is None or event_timestamp < prior:
                                block_first_ring_member[ring_id] = event_timestamp

                    receipt: ActionExecutionReceipt = executor.execute(
                        request, decision, executed_at=event.timestamp
                    )
                    audit.record_decision(
                        request, bundle, decision, recorded_at=event.timestamp
                    )
                    audit.record_execution(
                        request, decision, receipt, recorded_at=event.timestamp
                    )
                    execution_status_counts[receipt.status.value] += 1
                else:
                    action_counts["allow"] += 1
                    block_predictions[idx] = False

                # Evaluation-only join. No ground-truth value has been used by
                # the detector, investigators, fusion, policy, execution, or
                # audit trail above.
                truth_line += 1
                truth_id, event_is_fraud, event_ring_id = _read_truth_row(
                    gt_handle, truth_line
                )
                if truth_id != event_id:
                    raise ValueError(
                        "World D event/ground-truth order mismatch: "
                        f"events={event_id}, ground_truth={truth_id}"
                    )
                if truth_id in seen_truth_ids:
                    raise ValueError(f"Duplicate World D ground-truth event_id: {truth_id}")
                seen_truth_ids.add(truth_id)
                labels[idx] = int(event_is_fraud)

                if predicted and event_is_fraud:
                    detector_tp += 1
                elif predicted and not event_is_fraud:
                    detector_fp += 1
                elif not predicted and not event_is_fraud:
                    detector_tn += 1
                else:
                    detector_fn += 1

                if block_predictions[idx] and event_is_fraud:
                    block_tp += 1
                    blocked_fraud_amount += float(event.amount)
                elif block_predictions[idx] and not event_is_fraud:
                    block_fp += 1
                elif not block_predictions[idx] and event_is_fraud:
                    block_fn += 1
                else:
                    block_tn += 1

                if event_is_fraud:
                    fraud_exposure += float(event.amount)
                    if event_ring_id:
                        fraud_rings.add(event_ring_id)
                        prior = first_fraud.get(event_ring_id)
                        if prior is None or event_timestamp < prior:
                            first_fraud[event_ring_id] = event_timestamp
                if predicted and event_is_fraud:
                    detector_alerted_fraud_amount += float(event.amount)

                transactions += 1

            replay.process_event(raw, score_callback=score_current)
            if transactions and transactions % 100_000 == 0:
                elapsed = time.perf_counter() - started
                print(f"  Evaluated {transactions:,}/{expected:,} transactions ({elapsed/60:.1f} min)")

        extra = next((line for line in gt_handle if line.strip()), None)
        if extra is not None:
            raise RuntimeError("World D ground truth contains more records than event transactions")

    if transactions != expected:
        raise RuntimeError(
            f"World D transaction count mismatch: events={transactions}, ground_truth={expected}"
        )

    detector_metrics = _confusion(labels, detector_predictions)
    block_metrics = _confusion(labels, block_predictions)

    detector_ring = _ring_metrics(
        fraud_rings, detector_detected_rings, first_fraud, detector_first_ring_member_alert
    )
    block_ring = _ring_metrics(
        fraud_rings, blocked_ring_ids, first_fraud, block_first_ring_member
    )

    y = labels[:transactions]
    p = probabilities[:transactions]
    detector_pr_auc = float(average_precision_score(y, p))
    detector_roc_auc = float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "world": "world_d",
        "evaluation_type": "final_held_out_end_to_end",
        "transactions": transactions,
        "account_events": account_events,
        "fraud_transactions": int(y.sum()),
        "fraud_rate": float(y.mean()),
        "frozen_detector": {
            "model": MODEL_NAME,
            "artifact_dir": str(artifact_dir),
            "model_sha256": model_sha,
            "threshold": threshold,
            "feature_count": EXPECTED_FEATURE_COUNT,
            "feature_list": list(FEATURE_COLUMNS),
            "training_world": metadata.get("training", {}).get("world"),
        },
        "detector_only": {
            **detector_metrics,
            "pr_auc": detector_pr_auc,
            "roc_auc": detector_roc_auc,
            "ring_detection": detector_ring,
            "exposure": {
                "fraud_exposure": fraud_exposure,
                "alerted_fraud_amount": detector_alerted_fraud_amount,
                "exposure_if_detector_alerts_were_blocked_pct": (
                    detector_alerted_fraud_amount / fraud_exposure if fraud_exposure else 0.0
                ),
            },
        },
        "verifier_responder": {
            **block_metrics,
            "alerts": alerts,
            "reviews": action_counts["review"],
            "blocks": action_counts["block"],
            "allows": action_counts["allow"],
            "average_verification_confidence": verifier_confidence_sum / alerts if alerts else 0.0,
            "ring_detection": block_ring,
            "exposure": {
                "fraud_exposure": fraud_exposure,
                "blocked_fraud_amount": blocked_fraud_amount,
                "observed_fraud_exposure_blocked_pct": (
                    blocked_fraud_amount / fraud_exposure if fraud_exposure else 0.0
                ),
            },
            "evidence": {
                "total_items": evidence_items,
                "by_type": dict(sorted(evidence_by_type.items())),
                "average_items_per_alert": evidence_items / alerts if alerts else 0.0,
            },
            "execution": {
                "status_counts": execution_status_counts,
                "backend": "in_memory_reference_backend",
                "executed_alert_actions": alerts,
            },
            "audit": {
                "records": len(audit_trail.records),
                "last_hash": audit_trail.last_hash,
                "hash_chain_genesis": audit_trail.GENESIS_HASH,
            },
        },
        "comparison": {
            "detector_recall": detector_metrics["recall"],
            "block_recall": block_metrics["recall"],
            "recall_change_absolute": block_metrics["recall"] - detector_metrics["recall"],
            "detector_precision": detector_metrics["precision"],
            "block_precision": block_metrics["precision"],
            "economic_cost_detector": detector_metrics["economic_cost"],
            "economic_cost_blocking": block_metrics["economic_cost"],
            "economic_cost_reduction_pct": (
                (detector_metrics["economic_cost"] - block_metrics["economic_cost"])
                / detector_metrics["economic_cost"] * 100.0
                if detector_metrics["economic_cost"] else 0.0
            ),
        },
        "verifier_config": {
            "path": str(verifier_config_path),
            "sha256": _sha256(verifier_config_path),
            "world_c_selection": verifier_freeze.get("selection", {}),
            "fusion": verifier_freeze.get("fusion", {}),
            "policy": verifier_freeze.get("policy", {}),
        },
        "causal_contract": {
            "ground_truth_visible_to_detector": False,
            "ground_truth_used_only_for_evaluation": True,
            "read_pre_event_state": True,
            "score_before_current_transaction_state_update": True,
            "transaction_mutates_infrastructure_state": False,
            "temporal_model_feature_count": EXPECTED_FEATURE_COUNT,
            "temporal_model_raw_infrastructure_columns": [],
        },
        "evaluation_contract": {
            "world_d_is_final_held_out": True,
            "detector_retrained_on_world_d": False,
            "detector_threshold_tuned_on_world_d": False,
            "verifier_tuned_on_world_d": False,
            "policy_tuned_on_world_d": False,
            "events_source": str(events_path),
            "ground_truth_source": str(ground_truth_path),
            "manifest_source": str(manifest_path),
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print("WORLD D — FINAL HELD-OUT END-TO-END EVALUATION")
    print("=" * 72)
    print(f"Transactions:                 {transactions:,}")
    print(f"Fraud transactions:           {int(y.sum()):,} ({float(y.mean()):.4%})")
    print("\n[1] TEMPORAL DETECTOR ONLY")
    print(f"Threshold:                    {threshold:.4f}")
    print(f"Precision:                    {detector_metrics['precision']:.6f}")
    print(f"Recall:                       {detector_metrics['recall']:.6f}")
    print(f"FPR:                          {detector_metrics['fpr']:.6f}")
    print(f"FNR:                          {detector_metrics['fnr']:.6f}")
    print(f"PR-AUC:                       {detector_pr_auc:.6f}")
    print(f"ROC-AUC:                      {detector_roc_auc:.6f}" if detector_roc_auc is not None else "ROC-AUC:                      None")
    print(f"TP / FP / FN:                 {detector_metrics['true_positives']:,} / {detector_metrics['false_positives']:,} / {detector_metrics['false_negatives']:,}")
    print(f"Economic cost:                ₹{detector_metrics['economic_cost']:,.0f}")
    print(f"Eventual rings:                {detector_ring['detected_rings']}/{detector_ring['fraud_bearing_rings']} ({detector_ring['ring_detection_recall']:.2%})")
    print(f"Pre-abuse rings:               {detector_ring['rings_detected_pre_abuse']}/{detector_ring['rings_with_abuse']} ({detector_ring['pre_abuse_detection_recall']:.2%})")
    print(f"Median lead time:              {detector_ring['lead_time_days_median']} days")
    print(f"Alerted fraud exposure:        ₹{detector_alerted_fraud_amount:,.2f} ({detector_ring and (detector_alerted_fraud_amount / fraud_exposure if fraud_exposure else 0.0):.2%})")
    print("\n[2] TEMPORAL + VERIFIER + RESPONDER")
    print(f"Alerts:                       {alerts:,}")
    print(f"ALLOW / REVIEW / BLOCK:        {action_counts['allow']:,} / {action_counts['review']:,} / {action_counts['block']:,}")
    print(f"Block precision:               {block_metrics['precision']:.6f}")
    print(f"Block recall:                  {block_metrics['recall']:.6f}")
    print(f"Block FPR:                     {block_metrics['fpr']:.6f}")
    print(f"Block FNR:                     {block_metrics['fnr']:.6f}")
    print(f"Block TP / FP / FN:            {block_metrics['true_positives']:,} / {block_metrics['false_positives']:,} / {block_metrics['false_negatives']:,}")
    print(f"Economic cost:                ₹{block_metrics['economic_cost']:,.0f}")
    print(f"Eventual rings blocked:        {block_ring['detected_rings']}/{block_ring['fraud_bearing_rings']} ({block_ring['ring_detection_recall']:.2%})")
    print(f"Pre-abuse rings blocked:       {block_ring['rings_detected_pre_abuse']}/{block_ring['rings_with_abuse']} ({block_ring['pre_abuse_detection_recall']:.2%})")
    print(f"Median block lead time:        {block_ring['lead_time_days_median']} days")
    print(f"Blocked fraud exposure:        ₹{blocked_fraud_amount:,.2f} ({(blocked_fraud_amount / fraud_exposure if fraud_exposure else 0.0):.2%})")
    print(f"Evidence items / alert:        {evidence_items / alerts if alerts else 0.0:.2f}")
    print(f"Audit records:                 {len(audit_trail.records):,}")
    print("\n[3] DIRECT COMPARISON")
    print(f"Recall:                        {detector_metrics['recall']:.4%} → {block_metrics['recall']:.4%}")
    print(f"Precision:                     {detector_metrics['precision']:.4%} → {block_metrics['precision']:.4%}")
    print(f"Economic cost:                ₹{detector_metrics['economic_cost']:,.0f} → ₹{block_metrics['economic_cost']:,.0f}")
    print(f"Cost reduction:                {metrics['comparison']['economic_cost_reduction_pct']:.2f}%")
    print("\n[4] EVIDENCE BREAKDOWN")
    for key, value in sorted(evidence_by_type.items()):
        print(f"  {key:28s} {value:,}")
    print("\n[5] FREEZE / INTEGRITY")
    print(f"Temporal model SHA256:         {model_sha}")
    print(f"Verifier config SHA256:        {metrics['verifier_config']['sha256']}")
    print("World D tuning after freeze:   FALSE")
    print(f"Results:                       {output_dir / 'metrics.json'}")
    print("=" * 72)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Final held-out World D evaluation: Temporal detector + frozen verifier/responder."
    )
    parser.add_argument("--events", default="data/generated/world_d/events.jsonl")
    parser.add_argument("--ground-truth", default="data/generated/world_d/ground_truth.jsonl")
    parser.add_argument("--manifest", default="data/generated/world_d/manifest.json")
    parser.add_argument("--artifact-dir", default="artifacts/temporal")
    parser.add_argument(
        "--verifier-config",
        default="artifacts/verifier/world_c/tuning/freeze_config.json",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/evaluation/world_d",
    )
    args = parser.parse_args()
    evaluate_world_d(
        events_path=Path(args.events),
        ground_truth_path=Path(args.ground_truth),
        manifest_path=Path(args.manifest),
        artifact_dir=Path(args.artifact_dir),
        verifier_config_path=Path(args.verifier_config),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
