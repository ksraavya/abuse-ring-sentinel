from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from graph.infrastructure_state import InfrastructureState
from graph.behavioral_state import BehavioralState

from .contracts import (
    EvidenceContext,
    EvidenceItem,
    EvidenceStrength,
    EvidenceType,
    VerificationRequest,
)


@dataclass(frozen=True)
class InfrastructureInvestigatorConfig:
    """Conservative evidence-generation thresholds.

    These thresholds describe when infrastructure observations are worth
    emitting as verifier evidence. They are not fraud thresholds and do not
    directly determine ALLOW/REVIEW/BLOCK.
    """

    min_shared_accounts: int = 2
    min_behavioral_overlap: int = 1
    max_accounts_in_evidence: int = 100


class InfrastructureInvestigator:
    """Deterministic investigator for current account/device/IP evidence.

    The investigator uses the exact event-time ``InfrastructureState`` already
    maintained by the replay layer. It reads state before the current
    transaction is committed, never mutates that state, and never sees ground
    truth.

    The current InfrastructureState intentionally contains only the latest
    account -> device/IP mappings. Therefore this investigator does NOT emit
    ``INFRASTRUCTURE_CHURN`` evidence yet. Churn requires lifecycle history
    (with timestamps), which will be added only if development evidence shows
    that it materially improves verification.
    """

    name = "infrastructure-investigator"

    def __init__(self, config: InfrastructureInvestigatorConfig | None = None) -> None:
        self.config = config or InfrastructureInvestigatorConfig()

    def collect(
        self,
        request: VerificationRequest,
        context: EvidenceContext,
    ) -> list[EvidenceItem]:
        self._validate_context_time(request, context)
        infrastructure = self._require_infrastructure_state(context)
        behavioral = self._optional_behavioral_state(context)

        account = str(request.alert_event.account_id)
        device = infrastructure.account_to_device.get(account)
        ip_prefix = infrastructure.account_to_ip.get(account)

        device_accounts = set(infrastructure.device_to_accounts.get(device, set())) if device else set()
        ip_accounts = set(infrastructure.ip_to_accounts.get(ip_prefix, set())) if ip_prefix else set()
        device_accounts.discard(account)
        ip_accounts.discard(account)

        if len(device_accounts) + len(ip_accounts) == 0:
            return []

        behavioral_neighbors = behavioral.get_neighbors(account) if behavioral else set()
        device_behavioral_overlap = device_accounts & behavioral_neighbors
        ip_behavioral_overlap = ip_accounts & behavioral_neighbors

        subjects = {account} | device_accounts | ip_accounts
        subjects = set(sorted(subjects)[: self.config.max_accounts_in_evidence])

        shared_count = len(device_accounts | ip_accounts)
        overlap_count = len(device_behavioral_overlap | ip_behavioral_overlap)

        if shared_count < self.config.min_shared_accounts:
            return []

        strength = self._strength(shared_count, overlap_count)
        confidence = self._confidence(shared_count, overlap_count)
        summary = self._summary(
            device_accounts=device_accounts,
            ip_accounts=ip_accounts,
            overlap_count=overlap_count,
        )

        evidence = EvidenceItem(
            evidence_id=self._evidence_id(request, device_accounts, ip_accounts, overlap_count),
            evidence_type=EvidenceType.INFRASTRUCTURE_SHARING,
            strength=strength,
            source_agent=self.name,
            summary=summary,
            confidence=confidence,
            observed_at=context.as_of,
            source_event_ids=(request.event_id,),
            subject_account_ids=tuple(sorted(subjects)),
            metric_name="shared_infrastructure_accounts",
            metric_value=float(shared_count),
            details={
                "device": device,
                "ip_prefix": ip_prefix,
                "shared_device_accounts": tuple(sorted(device_accounts)),
                "shared_ip_accounts": tuple(sorted(ip_accounts)),
                "device_behavioral_overlap": tuple(sorted(device_behavioral_overlap)),
                "ip_behavioral_overlap": tuple(sorted(ip_behavioral_overlap)),
                "behavioral_overlap_accounts": tuple(sorted(device_behavioral_overlap | ip_behavioral_overlap)),
                "note": "Current infrastructure state only; no post-decision or lifecycle-future state used.",
            },
        )
        return [evidence]

    @staticmethod
    def _validate_context_time(request: VerificationRequest, context: EvidenceContext) -> None:
        if context.as_of > request.decision_time:
            raise ValueError("EvidenceContext.as_of cannot be after verifier decision_time")
        if context.as_of != request.alert_event.timestamp:
            raise ValueError(
                "InfrastructureInvestigator requires EvidenceContext.as_of to equal alert event time"
            )

    @staticmethod
    def _require_infrastructure_state(context: EvidenceContext) -> InfrastructureState:
        state = context.state.get("infrastructure_state")
        if not isinstance(state, InfrastructureState):
            raise TypeError("context.state['infrastructure_state'] must be InfrastructureState")
        return state

    @staticmethod
    def _optional_behavioral_state(context: EvidenceContext) -> BehavioralState | None:
        state = context.state.get("behavioral_state")
        if state is None:
            return None
        if not isinstance(state, BehavioralState):
            raise TypeError("context.state['behavioral_state'] must be BehavioralState when supplied")
        return state

    @staticmethod
    def _strength(shared_count: int, overlap_count: int) -> EvidenceStrength:
        if overlap_count >= 3 or shared_count >= 8:
            return EvidenceStrength.STRONG
        if overlap_count >= 1 or shared_count >= 4:
            return EvidenceStrength.MODERATE
        return EvidenceStrength.WEAK

    @staticmethod
    def _confidence(shared_count: int, overlap_count: int) -> float:
        # Conservative because shared infrastructure is common in benign
        # environments. Behavioral overlap provides stronger corroboration.
        return min(0.90, 0.45 + 0.03 * min(shared_count, 10) + 0.10 * min(overlap_count, 3))

    @staticmethod
    def _summary(
        *, device_accounts: set[str], ip_accounts: set[str], overlap_count: int
    ) -> str:
        parts: list[str] = []
        if device_accounts:
            parts.append(f"{len(device_accounts)} other account(s) share the current device")
        if ip_accounts:
            parts.append(f"{len(ip_accounts)} other account(s) share the current IP prefix")
        if overlap_count:
            parts.append(f"{overlap_count} of those accounts are also behavioral peers")
        return "The alerted account has current shared-infrastructure relationships: " + "; ".join(parts) + "."

    @staticmethod
    def _evidence_id(
        request: VerificationRequest,
        device_accounts: set[str],
        ip_accounts: set[str],
        overlap_count: int,
    ) -> str:
        payload = "|".join(
            [
                request.event_id,
                ",".join(sorted(device_accounts)),
                ",".join(sorted(ip_accounts)),
                str(overlap_count),
            ]
        )
        digest = sha256(payload.encode("utf-8")).hexdigest()[:16]
        return f"infra-{digest}"


__all__ = ["InfrastructureInvestigator", "InfrastructureInvestigatorConfig"]
