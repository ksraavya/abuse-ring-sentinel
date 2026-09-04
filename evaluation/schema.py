from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class PredictionRecord:
    """One model decision joined with evaluation-only ground truth.

    Ground truth is attached only after the detector has produced its decision;
    it is never part of the Kafka event payload or model input.
    """

    model: str
    event_id: str
    timestamp: datetime
    account_id: str
    amount: float
    probability: float
    predicted_fraud: bool
    is_fraud: bool
    ring_id: str | None = None

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("prediction timestamp must be timezone-aware")
        if not self.event_id:
            raise ValueError("event_id must be non-empty")
        if not self.account_id:
            raise ValueError("account_id must be non-empty")
        if self.amount <= 0:
            raise ValueError("amount must be positive")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")

    @property
    def utc_timestamp(self) -> datetime:
        return self.timestamp.astimezone(timezone.utc)
