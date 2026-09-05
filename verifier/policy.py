from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Sequence

from verifier.contracts import EvidenceBundle, EvidenceItem, EvidenceStrength, VerificationRequest


class PolicyAction(str, Enum):
    """Intervention intent produced by the policy layer."""

    ALLOW = "allow"
    REVIEW = "review"
    BLOCK = "block"


class RiskTier(str, Enum):
    """Coarse policy risk tier for audit and downstream actioning."""

    LOW = "low"
    ELEVATED = "elevated"
    HIGH = "high"


@dataclass(frozen=True)
class AutoResponderPolicyConfig:
    """Explicit, interpretable policy thresholds.

    These are initial development defaults only. They are not detector
    thresholds and are not tuned on World D. World C may tune them in 12B.

    An alert is never blocked from detector score alone. BLOCK additionally
    requires corroborating verifier evidence from at least two investigators,
    with at least one strong evidence item. Alerts that do not meet the block
    gate remain REVIEW rather than being silently allowed.
    """

    detector_alert_threshold: float = 0.01
    block_detector_threshold: float = 0.20
    block_verifier_threshold: float = 0.45
    review_verifier_threshold: float = 0.10
    min_block_agent_coverage: int = 2
    require_strong_evidence_for_block: bool = True
    policy_version: str = "11a-initial-v1"

    def __post_init__(self) -> None:
        for name in (
            "detector_alert_threshold",
            "block_detector_threshold",
            "block_verifier_threshold",
            "review_verifier_threshold",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0 or not isfinite(value):
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if self.block_detector_threshold < self.detector_alert_threshold:
            raise ValueError("block_detector_threshold cannot be below alert threshold")
        if self.review_verifier_threshold > self.block_verifier_threshold:
            raise ValueError("review_verifier_threshold cannot exceed block threshold")
        if self.min_block_agent_coverage < 1:
            raise ValueError("min_block_agent_coverage must be >= 1")
        if not self.policy_version.strip():
            raise ValueError("policy_version must be non-empty")


@dataclass(frozen=True)
class PolicyDecision:
    """Immutable policy output; it requests an action but does not execute it."""

    action: PolicyAction
    risk_tier: RiskTier
    policy_version: str
    detector_probability: float
    verification_confidence: float
    agent_coverage: int
    evidence_count: int
    strong_evidence_count: int
    reason_codes: tuple[str, ...]
    rationale: str


class AutoResponderPolicy:
    """Deterministic policy converting risk evidence into an action intent.

    The policy sits after detection and verification. It never changes the
    detector score, never mutates state, and never performs the requested
    action. Execution belongs to 11B and audit persistence belongs to 11C.

    Non-alert transactions are ALLOW. Once the frozen detector raises an
    alert, the safe default is REVIEW; BLOCK requires explicit corroboration.
    This prevents weak or missing verifier evidence from turning a detector
    alert into an unreviewed allow.
    """

    name = "auto-responder-policy"

    def __init__(self, config: AutoResponderPolicyConfig | None = None) -> None:
        self.config = config or AutoResponderPolicyConfig()

    def decide(
        self,
        request: VerificationRequest,
        bundle: EvidenceBundle,
        verification_confidence: float,
    ) -> PolicyDecision:
        self._validate_inputs(request, bundle, verification_confidence)

        items: Sequence[EvidenceItem] = bundle.items
        agents = {item.source_agent for item in items}
        strong_count = sum(item.strength is EvidenceStrength.STRONG for item in items)
        coverage = len(agents)
        detector = request.detector_probability
        verifier = verification_confidence

        if detector < self.config.detector_alert_threshold:
            return PolicyDecision(
                action=PolicyAction.ALLOW,
                risk_tier=RiskTier.LOW,
                policy_version=self.config.policy_version,
                detector_probability=detector,
                verification_confidence=verifier,
                agent_coverage=coverage,
                evidence_count=len(items),
                strong_evidence_count=strong_count,
                reason_codes=("below_detector_alert_threshold",),
                rationale="Detector score is below the configured alert threshold.",
            )

        block_reasons = [
            detector >= self.config.block_detector_threshold,
            verifier >= self.config.block_verifier_threshold,
            coverage >= self.config.min_block_agent_coverage,
            (not self.config.require_strong_evidence_for_block) or strong_count >= 1,
        ]

        if all(block_reasons):
            return PolicyDecision(
                action=PolicyAction.BLOCK,
                risk_tier=RiskTier.HIGH,
                policy_version=self.config.policy_version,
                detector_probability=detector,
                verification_confidence=verifier,
                agent_coverage=coverage,
                evidence_count=len(items),
                strong_evidence_count=strong_count,
                reason_codes=(
                    "detector_above_block_threshold",
                    "verifier_above_block_threshold",
                    "multi_agent_corroboration",
                    "strong_evidence_present" if self.config.require_strong_evidence_for_block else "strong_evidence_not_required",
                ),
                rationale=(
                    "Detector risk is high and the alert is corroborated by "
                    "independent verifier evidence sufficient for the block gate."
                ),
            )

        reasons = ["detector_alert"]
        if verifier >= self.config.review_verifier_threshold:
            reasons.append("verifier_supports_review")
        else:
            reasons.append("insufficient_verifier_support_for_block")
        if coverage < self.config.min_block_agent_coverage:
            reasons.append("insufficient_agent_corroboration")
        if self.config.require_strong_evidence_for_block and strong_count == 0:
            reasons.append("no_strong_evidence")

        return PolicyDecision(
            action=PolicyAction.REVIEW,
            risk_tier=RiskTier.ELEVATED,
            policy_version=self.config.policy_version,
            detector_probability=detector,
            verification_confidence=verifier,
            agent_coverage=coverage,
            evidence_count=len(items),
            strong_evidence_count=strong_count,
            reason_codes=tuple(reasons),
            rationale=(
                "The detector raised an alert, but the explicit block gate is "
                "not satisfied; route to review rather than silently allowing it."
            ),
        )

    @staticmethod
    def _validate_inputs(
        request: VerificationRequest,
        bundle: EvidenceBundle,
        verification_confidence: float,
    ) -> None:
        if bundle.alert_event_id != request.event_id:
            raise ValueError("evidence bundle belongs to a different alert event")
        if bundle.decision_time != request.decision_time:
            raise ValueError("evidence bundle decision_time must match request")
        if not 0.0 <= verification_confidence <= 1.0 or not isfinite(verification_confidence):
            raise ValueError("verification_confidence must be finite and in [0, 1]")


__all__ = [
    "AutoResponderPolicy",
    "AutoResponderPolicyConfig",
    "PolicyAction",
    "PolicyDecision",
    "RiskTier",
]
