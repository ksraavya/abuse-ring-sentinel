from __future__ import annotations

from typing import Protocol

from .contracts import EvidenceContext, EvidenceItem, VerificationRequest


class EvidenceAgent(Protocol):
    """Interface implemented by one deterministic evidence investigator.

    An agent may read only the current alert and pre-decision context. It must
    return evidence items rather than an intervention decision.
    """

    name: str

    def collect(
        self,
        request: VerificationRequest,
        context: EvidenceContext,
    ) -> list[EvidenceItem]:
        """Collect auditable evidence for one alert."""
        ...


class EvidenceFusion(Protocol):
    """Interface for combining evidence without owning intervention policy."""

    name: str

    def fuse(self, request: VerificationRequest, items: list[EvidenceItem]) -> float:
        """Return a normalized verification confidence in [0, 1]."""
        ...


__all__ = ["EvidenceAgent", "EvidenceFusion"]
