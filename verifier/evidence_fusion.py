from __future__ import annotations

from dataclasses import dataclass
from datetime import timezone
from math import isfinite
from typing import Mapping

from .contracts import (
    EvidenceItem,
    EvidenceStrength,
    EvidenceType,
    VerificationRequest,
)


DEFAULT_TYPE_WEIGHTS: dict[EvidenceType, float] = {
    EvidenceType.RING_STRUCTURE: 0.24,
    EvidenceType.BEHAVIORAL_ACCELERATION: 0.18,
    EvidenceType.PEER_SYNCHRONY: 0.14,
    EvidenceType.MERCHANT_CONVERGENCE: 0.12,
    EvidenceType.INFRASTRUCTURE_SHARING: 0.12,
    EvidenceType.INFRASTRUCTURE_CHURN: 0.05,
    EvidenceType.ACCOUNT_CONTEXT: 0.07,
    EvidenceType.TEMPORAL_CONTEXT: 0.08,
}

DEFAULT_STRENGTH_MULTIPLIERS: dict[EvidenceStrength, float] = {
    EvidenceStrength.WEAK: 0.35,
    EvidenceStrength.MODERATE: 0.65,
    EvidenceStrength.STRONG: 1.0,
}


@dataclass(frozen=True)
class EvidenceFusionConfig:
    """Frozen, interpretable parameters for evidence fusion.

    These parameters define how evidence is summarized; they are not fraud
    thresholds and are not intervention policy. World C may tune them before
    World D is frozen.
    """

    detector_weight: float = 0.65
    coverage_bonus: float = 0.08
    expected_agent_names: tuple[str, ...] = (
        "ring-investigator",
        "infrastructure-investigator",
        "context-investigator",
    )
    type_weights: Mapping[EvidenceType, float] | None = None
    strength_multipliers: Mapping[EvidenceStrength, float] | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.detector_weight <= 1.0:
            raise ValueError("detector_weight must be in [0, 1]")
        if not 0.0 <= self.coverage_bonus <= 1.0:
            raise ValueError("coverage_bonus must be in [0, 1]")
        if not self.expected_agent_names:
            raise ValueError("expected_agent_names must not be empty")
        if len(self.expected_agent_names) != len(set(self.expected_agent_names)):
            raise ValueError("expected_agent_names must be unique")

        weights = dict(self.type_weights or DEFAULT_TYPE_WEIGHTS)
        multipliers = dict(self.strength_multipliers or DEFAULT_STRENGTH_MULTIPLIERS)

        if set(weights) != set(EvidenceType):
            raise ValueError("type_weights must contain exactly every EvidenceType")
        if any(weight < 0.0 or not isfinite(weight) for weight in weights.values()):
            raise ValueError("type_weights must be finite and non-negative")
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError("type_weights must sum to 1.0")
        if set(multipliers) != set(EvidenceStrength):
            raise ValueError("strength_multipliers must contain exactly every EvidenceStrength")
        if any(value < 0.0 or not isfinite(value) for value in multipliers.values()):
            raise ValueError("strength_multipliers must be finite and non-negative")

        object.__setattr__(self, "type_weights", weights)
        object.__setattr__(self, "strength_multipliers", multipliers)


@dataclass(frozen=True)
class FusionBreakdown:
    """Auditable intermediate values for one fusion call."""

    detector_probability: float
    evidence_support: float
    agent_coverage: float
    coverage_bonus: float
    evidence_score: float
    fused_confidence: float
    strongest_by_type: dict[EvidenceType, float]
    evidence_count: int
    contributing_agent_names: tuple[str, ...]


class DeterministicEvidenceFusion:
    """Combine detector confidence with corroborating evidence.

    The fusion layer deliberately does *not* decide ALLOW/REVIEW/BLOCK and
    does not claim that its output is a calibrated fraud probability. The
    detector score remains visible in the result and contributes directly to
    the verification confidence.

    Evidence is de-duplicated by evidence type for scoring: repeated items of
    the same type cannot stack without bound. For each type, only the strongest
    effective contribution ``confidence * strength_multiplier`` is retained.
    Evidence from multiple investigators is represented separately through a
    small, bounded agent-coverage bonus rather than additive double counting.
    """

    name = "deterministic-evidence-fusion"

    def __init__(self, config: EvidenceFusionConfig | None = None) -> None:
        self.config = config or EvidenceFusionConfig()

    def fuse(self, request: VerificationRequest, items: list[EvidenceItem]) -> float:
        return self.explain(request, items).fused_confidence

    def explain(
        self,
        request: VerificationRequest,
        items: list[EvidenceItem],
    ) -> FusionBreakdown:
        self._validate_items(request, items)
        type_weights = self.config.type_weights
        strength_multipliers = self.config.strength_multipliers
        assert type_weights is not None
        assert strength_multipliers is not None

        strongest: dict[EvidenceType, float] = {
            evidence_type: 0.0 for evidence_type in EvidenceType
        }
        agents: set[str] = set()

        for item in items:
            effective = item.confidence * strength_multipliers[item.strength]
            if effective > strongest[item.evidence_type]:
                strongest[item.evidence_type] = effective
            agents.add(item.source_agent)

        evidence_support = sum(
            type_weights[evidence_type] * contribution
            for evidence_type, contribution in strongest.items()
        )
        expected = set(self.config.expected_agent_names)
        recognized_agents = agents & expected
        coverage = len(recognized_agents) / len(expected)
        bonus = self.config.coverage_bonus * coverage
        evidence_score = min(1.0, evidence_support + bonus)
        # Evidence is corroborative, not a replacement classifier. With no
        # evidence the detector score is unchanged; positive evidence can only
        # move confidence upward toward 1.0.
        fused = request.detector_probability + (
            1.0 - request.detector_probability
        ) * (1.0 - self.config.detector_weight) * evidence_score

        return FusionBreakdown(
            detector_probability=request.detector_probability,
            evidence_support=evidence_support,
            agent_coverage=coverage,
            coverage_bonus=bonus,
            evidence_score=evidence_score,
            fused_confidence=max(0.0, min(1.0, fused)),
            strongest_by_type=dict(strongest),
            evidence_count=len(items),
            contributing_agent_names=tuple(sorted(recognized_agents)),
        )

    @staticmethod
    def _validate_items(request: VerificationRequest, items: list[EvidenceItem]) -> None:
        seen_ids: set[str] = set()
        decision_time = request.decision_time.astimezone(timezone.utc)
        alert_time = request.alert_event.timestamp.astimezone(timezone.utc)

        for item in items:
            if item.evidence_id in seen_ids:
                raise ValueError(f"duplicate evidence_id: {item.evidence_id}")
            seen_ids.add(item.evidence_id)

            observed = item.observed_at.astimezone(timezone.utc)
            if observed > decision_time:
                raise ValueError(
                    f"evidence {item.evidence_id} is newer than decision_time"
                )
            if observed > alert_time:
                raise ValueError(
                    f"evidence {item.evidence_id} is newer than alert event time"
                )


__all__ = [
    "DEFAULT_STRENGTH_MULTIPLIERS",
    "DEFAULT_TYPE_WEIGHTS",
    "DeterministicEvidenceFusion",
    "EvidenceFusionConfig",
    "FusionBreakdown",
]
