from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from verifier.evidence_fusion import (
    DEFAULT_STRENGTH_MULTIPLIERS,
    DEFAULT_TYPE_WEIGHTS,
    EvidenceFusionConfig,
)
from verifier.policy import AutoResponderPolicyConfig, PolicyAction
from verifier.contracts import EvidenceStrength, EvidenceType


@dataclass(frozen=True)
class TuningCase:
    event_id: str
    timestamp: datetime
    account_id: str
    amount: float
    detector_probability: float
    verification_confidence: float
    agent_coverage: int
    strong_evidence_count: int
    evidence: tuple[dict[str, Any], ...]
    is_fraud: bool
    ring_id: str | None


@dataclass(frozen=True)
class TuningResult:
    fusion: EvidenceFusionConfig
    policy: AutoResponderPolicyConfig
    transactions: int
    alerts: int
    blocks: int
    reviews: int
    blocked_fraud_transactions: int
    blocked_fraud_amount: float
    fraud_amount: float
    false_positive_blocks: int
    blocked_ring_ids: frozenset[str]
    pre_abuse_ring_ids: frozenset[str]
    fraud_ring_ids: frozenset[str]
    pre_abuse_recall: float
    ring_recall: float
    exposure_prevented_pct: float
    block_precision: float
    block_recall: float
    block_fpr: float
    economic_cost: float

    def score_key(self) -> tuple[float, float, float, float, float]:
        # Selection priority:
        # 1. Eventual ring recall — preserve broad ring coverage
        # 2. Pre-abuse ring recall — prefer earlier blocking among equals
        # 3. Block precision — prefer accurate intervention
        # 4. Exposure prevented
        # 5. Fewer false-positive blocks
        return (
            self.ring_recall,
            self.pre_abuse_recall,
            self.block_precision,
            self.exposure_prevented_pct,
            -float(self.false_positive_blocks),
        )


def _parse_time(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return dt.astimezone(timezone.utc)


def load_cases(records_path: Path, ground_truth_path: Path, events_path: Path) -> tuple[TuningCase, ...]:
    truth: dict[str, tuple[bool, str | None]] = {}
    with ground_truth_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            event_id = str(raw.get("event_id", ""))
            if not event_id:
                raise ValueError(f"ground truth line {line_no} has no event_id")
            if event_id in truth:
                raise ValueError(f"duplicate ground-truth event_id: {event_id}")
            truth[event_id] = (bool(raw["is_fraud"]), raw.get("ring_id"))

    alert_ids: set[str] = set()
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            event_id = str(raw.get("event_id", ""))
            if event_id:
                alert_ids.add(event_id)

    amounts: dict[str, float] = {}
    with events_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            if raw.get("event_type") != "transaction":
                continue
            event_id = str(raw.get("event_id", ""))
            if event_id in alert_ids:
                amounts[event_id] = float(raw["amount"])
                if len(amounts) == len(alert_ids):
                    break

    missing = alert_ids - amounts.keys()
    if missing:
        raise ValueError(
            f"Missing transaction amounts for {len(missing)} alert events"
        )

    cases: list[TuningCase] = []
    seen: set[str] = set()
    with records_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            event_id = str(raw.get("event_id", ""))
            if not event_id or event_id in seen:
                raise ValueError(f"invalid or duplicate alert event_id at line {line_no}: {event_id!r}")
            seen.add(event_id)
            if event_id not in truth:
                raise ValueError(f"alert {event_id} missing from World C ground truth")
            policy = raw.get("policy", {})
            fusion = raw.get("fusion", {})
            evidence = tuple(raw.get("evidence", ()))
            fraud, ring_id = truth[event_id]
            cases.append(TuningCase(
                event_id=event_id,
                timestamp=_parse_time(str(raw["timestamp"])),
                account_id=str(raw["account_id"]),
                amount=amounts[event_id],
                detector_probability=float(raw["detector_probability"]),
                verification_confidence=float(raw["verification_confidence"]),
                agent_coverage=len(fusion.get("contributing_agent_names", ()))
                    if "contributing_agent_names" in fusion
                    else int(policy.get("agent_coverage", 0)),
                strong_evidence_count=int(policy.get("strong_evidence_count", 0)),
                evidence=evidence,
                is_fraud=fraud,
                ring_id=ring_id,
            ))
    return tuple(cases)


def _load_ring_members(manifest_path: Path) -> dict[str, set[str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result: dict[str, set[str]] = {}
    for ring in manifest.get("rings", []):
        ring_id = str(ring.get("ring_id", ""))
        members = ring.get("account_ids")
        if not ring_id or not isinstance(members, list):
            raise ValueError("invalid World C ring manifest entry")
        if ring_id in result:
            raise ValueError(f"duplicate ring_id: {ring_id}")
        result[ring_id] = {str(x) for x in members}
    return result


def _verification_confidence(case: TuningCase, config: EvidenceFusionConfig) -> float:
    weights = config.type_weights
    multipliers = config.strength_multipliers
    assert weights is not None and multipliers is not None
    strongest = {kind: 0.0 for kind in EvidenceType}
    agents: set[str] = set()
    for item in case.evidence:
        kind = EvidenceType(str(item["evidence_type"]))
        strength = EvidenceStrength(str(item["strength"]))
        confidence = float(item["confidence"])
        strongest[kind] = max(strongest[kind], confidence * multipliers[strength])
        agents.add(str(item["source_agent"]))
    support = sum(weights[k] * v for k, v in strongest.items())
    expected = set(config.expected_agent_names)
    coverage = len(agents & expected) / len(expected)
    score = min(1.0, support + config.coverage_bonus * coverage)
    fused = case.detector_probability + (1.0 - case.detector_probability) * (1.0 - config.detector_weight) * score
    return max(0.0, min(1.0, fused))


def _candidate_policy(config: AutoResponderPolicyConfig, *, block_detector: float, block_verifier: float, min_agents: int, require_strong: bool) -> AutoResponderPolicyConfig:
    return replace(config,
        block_detector_threshold=block_detector,
        block_verifier_threshold=block_verifier,
        min_block_agent_coverage=min_agents,
        require_strong_evidence_for_block=require_strong,
        policy_version="12b-candidate",
    )


def evaluate_candidate(
    cases: Iterable[TuningCase],
    *,
    fusion: EvidenceFusionConfig,
    policy: AutoResponderPolicyConfig,
    ring_members: Mapping[str, set[str]],
) -> TuningResult:
    rows = list(cases)
    alerts = len(rows)
    fraud_amount = sum(c.amount for c in rows if c.is_fraud)
    blocks = reviews = blocked_fraud = false_positive_blocks = 0
    blocked_fraud_amount = 0.0
    blocked_ring_ids: set[str] = set()
    fraud_ring_ids: set[str] = {c.ring_id for c in rows if c.is_fraud and c.ring_id}

    blocked_events = []
    for case in rows:
        conf = _verification_confidence(case, fusion)
        eligible = (
            case.detector_probability >= policy.block_detector_threshold
            and conf >= policy.block_verifier_threshold
            and case.agent_coverage >= policy.min_block_agent_coverage
            and ((not policy.require_strong_evidence_for_block) or case.strong_evidence_count >= 1)
        )
        if eligible:
            blocks += 1
            blocked_events.append(case)
            if case.is_fraud:
                blocked_fraud += 1
                blocked_fraud_amount += case.amount
                if case.ring_id:
                    blocked_ring_ids.add(case.ring_id)
            else:
                false_positive_blocks += 1
        else:
            reviews += 1

    first_fraud: dict[str, datetime] = {}
    first_block_member: dict[str, datetime] = {}
    for case in rows:
        if case.is_fraud and case.ring_id:
            prior = first_fraud.get(case.ring_id)
            if prior is None or case.timestamp < prior:
                first_fraud[case.ring_id] = case.timestamp
        if case in blocked_events:
            for ring_id, members in ring_members.items():
                if case.account_id in members:
                    prior = first_block_member.get(ring_id)
                    if prior is None or case.timestamp < prior:
                        first_block_member[ring_id] = case.timestamp

    pre_abuse = {
        ring_id for ring_id, fraud_time in first_fraud.items()
        if ring_id in first_block_member and first_block_member[ring_id] < fraud_time
    }
    rings = set(first_fraud) or fraud_ring_ids
    pre_recall = len(pre_abuse) / len(rings) if rings else 0.0
    ring_recall = len(blocked_ring_ids) / len(fraud_ring_ids) if fraud_ring_ids else 0.0
    exposure_pct = blocked_fraud_amount / fraud_amount if fraud_amount else 0.0
    precision = blocked_fraud / blocks if blocks else 0.0
    recall = blocked_fraud / sum(c.is_fraud for c in rows) if any(c.is_fraud for c in rows) else 0.0
    negatives = sum(not c.is_fraud for c in rows)
    alert_negative_block_rate = false_positive_blocks / negatives if negatives else 0.0
    cost = 500.0 * false_positive_blocks + 5000.0 * (sum(c.is_fraud for c in rows) - blocked_fraud)
    return TuningResult(
        fusion=fusion, policy=policy, transactions=alerts, alerts=alerts,
        blocks=blocks, reviews=reviews, blocked_fraud_transactions=blocked_fraud,
        blocked_fraud_amount=blocked_fraud_amount, fraud_amount=fraud_amount,
        false_positive_blocks=false_positive_blocks,
        blocked_ring_ids=frozenset(blocked_ring_ids), pre_abuse_ring_ids=frozenset(pre_abuse),
        fraud_ring_ids=frozenset(fraud_ring_ids), pre_abuse_recall=pre_recall,
        ring_recall=ring_recall, exposure_prevented_pct=exposure_pct,
        block_precision=precision, block_recall=recall,
        block_fpr=alert_negative_block_rate, economic_cost=cost,
    )


def default_fusion_candidates() -> tuple[EvidenceFusionConfig, ...]:
    # Small, predeclared interpretable grid. Type weights are intentionally
    # kept at their locked values in 12B; we tune scalar fusion behavior first.
    return tuple(
        EvidenceFusionConfig(detector_weight=w, coverage_bonus=b)
        for w in (0.55, 0.65, 0.75)
        for b in (0.04, 0.08, 0.12)
    )


def default_policy_candidates() -> tuple[AutoResponderPolicyConfig, ...]:
    base = AutoResponderPolicyConfig()
    return tuple(
        _candidate_policy(base, block_detector=d, block_verifier=v, min_agents=a, require_strong=s)
        for d in (0.10, 0.15, 0.20, 0.25, 0.30)
        for v in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55)
        for a in (1, 2, 3)
        for s in (True, False)
    )


def tune_world_c(
    *, records_path: Path, ground_truth_path: Path, manifest_path: Path,
    output_dir: Path, events_path: Path, min_block_precision: float = 0.70,
) -> TuningResult:
    cases = load_cases(records_path, ground_truth_path, events_path)
    rings = _load_ring_members(manifest_path)
    candidates: list[TuningResult] = []
    for fusion in default_fusion_candidates():
        for policy in default_policy_candidates():
            result = evaluate_candidate(cases, fusion=fusion, policy=policy, ring_members=rings)
            if result.block_precision >= min_block_precision:
                candidates.append(result)
    if not candidates:
        raise RuntimeError(
            f"No World C candidate achieved >= {min_block_precision:.0%} block precision. "
            "Inspect evidence quality before freezing."
        )
    candidates.sort(key=lambda r: r.score_key(), reverse=True)
    winner = candidates[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in candidates:
        rows.append({
            "detector_weight": result.fusion.detector_weight,
            "coverage_bonus": result.fusion.coverage_bonus,
            "block_detector_threshold": result.policy.block_detector_threshold,
            "block_verifier_threshold": result.policy.block_verifier_threshold,
            "min_block_agent_coverage": result.policy.min_block_agent_coverage,
            "require_strong_evidence_for_block": result.policy.require_strong_evidence_for_block,
            "pre_abuse_recall": result.pre_abuse_recall,
            "ring_recall": result.ring_recall,
            "exposure_prevented_pct": result.exposure_prevented_pct,
            "block_precision": result.block_precision,
            "block_recall": result.block_recall,
            "alert_negative_block_rate": result.block_fpr,
            "blocks": result.blocks,
            "reviews": result.reviews,
            "false_positive_blocks": result.false_positive_blocks,
            "economic_cost": result.economic_cost,
        })
    (output_dir / "candidate_sweep.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    freeze = {
        "world": "world_c",
        "purpose": "verifier_and_policy_development_then_freeze",
        "selection": {
            "constraint": f"block_precision >= {min_block_precision:.4f}",
            "priority": ["ring_recall", "pre_abuse_recall", "block_precision", "exposure_prevented_pct", "-false_positive_blocks"],
            "candidate_count_after_constraint": len(candidates),
        },
        "fusion": {
            "detector_weight": winner.fusion.detector_weight,
            "coverage_bonus": winner.fusion.coverage_bonus,
            "type_weights": {k.value: v for k, v in winner.fusion.type_weights.items()},
            "strength_multipliers": {k.value: v for k, v in winner.fusion.strength_multipliers.items()},
            "expected_agent_names": list(winner.fusion.expected_agent_names),
        },
        "policy": {
            "detector_alert_threshold": winner.policy.detector_alert_threshold,
            "block_detector_threshold": winner.policy.block_detector_threshold,
            "block_verifier_threshold": winner.policy.block_verifier_threshold,
            "review_verifier_threshold": winner.policy.review_verifier_threshold,
            "min_block_agent_coverage": winner.policy.min_block_agent_coverage,
            "require_strong_evidence_for_block": winner.policy.require_strong_evidence_for_block,
            "policy_version": "12b-world-c-frozen-v1",
        },
        "metrics": {
            "pre_abuse_recall": winner.pre_abuse_recall,
            "ring_recall": winner.ring_recall,
            "exposure_prevented_pct": winner.exposure_prevented_pct,
            "block_precision": winner.block_precision,
            "block_recall": winner.block_recall,
            "alert_negative_block_rate": winner.block_fpr,
            "blocks": winner.blocks,
            "reviews": winner.reviews,
            "false_positive_blocks": winner.false_positive_blocks,
            "economic_cost": winner.economic_cost,
            "blocked_fraud_transactions": winner.blocked_fraud_transactions,
            "blocked_fraud_amount": winner.blocked_fraud_amount,
            "fraud_amount": winner.fraud_amount,
        },
        "ground_truth_used_only_for_selection": True,
        "detector_retrained_on_world_c": False,
        "detector_threshold_tuned_on_world_c": False,
    }
    (output_dir / "freeze_config.json").write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return winner


def main_cli(args: Any) -> None:
    winner = tune_world_c(
        records_path=Path(args.records), ground_truth_path=Path(args.ground_truth),
        events_path=Path(args.events), manifest_path=Path(args.manifest), output_dir=Path(args.output_dir),
        min_block_precision=float(args.min_block_precision),
    )
    print("World C verifier/policy tuning complete and frozen.")
    print(f"  Candidate blocks:              {winner.blocks:,}")
    print(f"  Candidate reviews:             {winner.reviews:,}")
    print(f"  Pre-abuse recall:              {winner.pre_abuse_recall:.4%}")
    print(f"  Ring recall:                   {winner.ring_recall:.4%}")
    print(f"  Exposure prevented:            {winner.exposure_prevented_pct:.4%}")
    print(f"  Block precision:               {winner.block_precision:.4%}")
    print(f"  Block recall:                  {winner.block_recall:.4%}")
    print(f"  Alert negative block rate:     {winner.block_fpr:.6%}")
    print(f"  Economic cost:                 ₹{winner.economic_cost:,.0f}")
    print(f"  Frozen config:                 {Path(args.output_dir) / 'freeze_config.json'}")


__all__ = ["TuningCase", "TuningResult", "load_cases", "evaluate_candidate", "tune_world_c"]