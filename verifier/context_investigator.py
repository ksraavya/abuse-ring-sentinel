from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Iterable

from events.schema import TransactionEvent
from features.temporal import HistoricalTransaction, TemporalFeatureState

from .contracts import (
    EvidenceContext,
    EvidenceItem,
    EvidenceStrength,
    EvidenceType,
    VerificationRequest,
)


@dataclass(frozen=True)
class ContextInvestigatorConfig:
    """Conservative thresholds for account/context corroboration.

    These values are evidence-generation thresholds, not fraud thresholds.
    They are intentionally exposed so World C can evaluate whether contextual
    evidence actually reduces false positives before any production policy is
    chosen.
    """

    history_window_days: int = 14
    amount_history_min_transactions: int = 3
    amount_deviation_multiplier: float = 3.0
    merchant_novelty_min_history: int = 3
    p2p_context_min_transactions: int = 3
    recent_merchant_min_transactions: int = 2


class ContextInvestigator:
    """Deterministic investigator for transaction/account context.

    This investigator is deliberately different from the Ring Investigator.
    It does not ask whether the account sits inside a suspicious graph. It asks
    whether the *current transaction* is unusual or contextually consistent
    with the account's own pre-decision history.

    Evidence is based only on state strictly before the alert event:
      - whether the current transaction is merchant-directed or P2P;
      - whether the current merchant is historically novel;
      - whether the amount is materially outside the account's recent amount
        distribution;
      - whether recent history is unusually concentrated on merchant activity
        versus P2P activity.

    No ground truth, ring identifier, future event, or account-age shortcut is
    used. The investigator never mutates TemporalFeatureState.
    """

    name = "context-investigator"

    def __init__(self, config: ContextInvestigatorConfig | None = None) -> None:
        self.config = config or ContextInvestigatorConfig()

    def collect(
        self,
        request: VerificationRequest,
        context: EvidenceContext,
    ) -> list[EvidenceItem]:
        self._validate_context_time(request, context)
        temporal = self._require_temporal_state(context)
        event = request.alert_event
        account = str(event.account_id)
        history = temporal.account_events.get(account, [])
        t = context.as_of
        recent = self._window(history, t, timedelta(days=self.config.history_window_days))
        evidence: list[EvidenceItem] = []

        # The current transaction is not inserted into TemporalFeatureState yet,
        # so all calculations below are guaranteed to be pre-event.
        if event.merchant_id is not None:
            merchant_id = str(event.merchant_id)
            prior_merchant_txns = [
                row for row in recent if row.merchant_id == merchant_id
            ]
            if len(recent) >= self.config.merchant_novelty_min_history and not prior_merchant_txns:
                evidence.append(
                    self._item(
                        request,
                        EvidenceType.ACCOUNT_CONTEXT,
                        EvidenceStrength.MODERATE,
                        "The current merchant is novel for this account within the recent historical window.",
                        confidence=0.65,
                        metric_name="merchant_history_count",
                        metric_value=float(len(recent)),
                        details={
                            "merchant_id": merchant_id,
                            "prior_transactions_to_merchant": 0,
                            "window": f"[T-{self.config.history_window_days}d, T)",
                        },
                    )
                )

            merchant_history_amounts = [
                row.amount
                for row in recent
                if row.merchant_id is not None
            ]
            if len(merchant_history_amounts) >= self.config.amount_history_min_transactions:
                baseline = median(merchant_history_amounts)
                if baseline > 0 and event.amount >= baseline * self.config.amount_deviation_multiplier:
                    ratio = float(event.amount / baseline)
                    strength = (
                        EvidenceStrength.STRONG if ratio >= 6.0 else EvidenceStrength.MODERATE
                    )
                    confidence = min(0.92, 0.55 + 0.06 * min(ratio - self.config.amount_deviation_multiplier, 5.0))
                    evidence.append(
                        self._item(
                            request,
                            EvidenceType.TEMPORAL_CONTEXT,
                            strength,
                            "The current merchant transaction amount is materially above the account's recent merchant-transaction baseline.",
                            confidence=confidence,
                            metric_name="amount_to_recent_median_ratio",
                            metric_value=ratio,
                            details={
                                "recent_merchant_transaction_count": len(merchant_history_amounts),
                                "recent_merchant_amount_median": float(baseline),
                                "window": f"[T-{self.config.history_window_days}d, T)",
                            },
                        )
                    )

            if len(recent) >= self.config.recent_merchant_min_transactions:
                merchant_count = sum(1 for row in recent if row.merchant_id is not None)
                p2p_count = sum(1 for row in recent if row.counterparty_account_id is not None)
                if p2p_count >= self.config.p2p_context_min_transactions and merchant_count > 0:
                    p2p_share = p2p_count / len(recent)
                    evidence.append(
                        self._item(
                            request,
                            EvidenceType.ACCOUNT_CONTEXT,
                            EvidenceStrength.MODERATE if p2p_share >= 0.5 else EvidenceStrength.WEAK,
                            "The account's recent history contains substantial peer-to-peer activity before a merchant-directed alert.",
                            confidence=min(0.90, 0.50 + 0.35 * p2p_share),
                            metric_name="recent_p2p_share",
                            metric_value=float(p2p_share),
                            details={
                                "recent_transactions": len(recent),
                                "recent_p2p_transactions": p2p_count,
                                "recent_merchant_transactions": merchant_count,
                                "window": f"[T-{self.config.history_window_days}d, T)",
                            },
                        )
                    )
        else:
            # A P2P alert is contextualized rather than automatically treated
            # as suspicious. This can become useful in World C when hard
            # negatives contain legitimate high-volume P2P clusters.
            if recent:
                p2p_count = sum(1 for row in recent if row.counterparty_account_id is not None)
                p2p_share = p2p_count / len(recent)
                if p2p_count >= self.config.p2p_context_min_transactions and p2p_share >= 0.75:
                    evidence.append(
                        self._item(
                            request,
                            EvidenceType.ACCOUNT_CONTEXT,
                            EvidenceStrength.WEAK,
                            "The current P2P transaction is consistent with a history dominated by peer-to-peer activity.",
                            confidence=0.70,
                            metric_name="recent_p2p_share",
                            metric_value=float(p2p_share),
                            details={
                                "recent_transactions": len(recent),
                                "recent_p2p_transactions": p2p_count,
                                "window": f"[T-{self.config.history_window_days}d, T)",
                            },
                        )
                    )

        return evidence

    @staticmethod
    def _validate_context_time(request: VerificationRequest, context: EvidenceContext) -> None:
        if context.as_of > request.decision_time:
            raise ValueError("EvidenceContext.as_of cannot be after verifier decision_time")
        if context.as_of != request.alert_event.timestamp.astimezone(timezone.utc):
            raise ValueError(
                "ContextInvestigator requires EvidenceContext.as_of to equal alert event time"
            )

    @staticmethod
    def _require_temporal_state(context: EvidenceContext) -> TemporalFeatureState:
        state = context.state.get("temporal_feature_state")
        if not isinstance(state, TemporalFeatureState):
            raise TypeError("context.state['temporal_feature_state'] must be TemporalFeatureState")
        return state

    @staticmethod
    def _window(
        rows: list[HistoricalTransaction],
        end: datetime,
        window: timedelta,
    ) -> list[HistoricalTransaction]:
        start = end - window
        left = bisect_left(rows, start, key=lambda row: row.timestamp)
        right = bisect_left(rows, end, key=lambda row: row.timestamp)
        return rows[left:right]

    @staticmethod
    def _item(
        request: VerificationRequest,
        evidence_type: EvidenceType,
        strength: EvidenceStrength,
        summary: str,
        *,
        confidence: float,
        metric_name: str,
        metric_value: float,
        details: dict[str, object],
    ) -> EvidenceItem:
        return EvidenceItem(
            evidence_id=f"{request.event_id}:{evidence_type.value}:{metric_name}",
            evidence_type=evidence_type,
            strength=strength,
            source_agent=ContextInvestigator.name,
            summary=summary,
            confidence=max(0.0, min(1.0, confidence)),
            observed_at=request.alert_event.timestamp,
            source_event_ids=(request.event_id,),
            subject_account_ids=(str(request.alert_event.account_id),),
            metric_name=metric_name,
            metric_value=metric_value,
            details=details,
        )


__all__ = ["ContextInvestigator", "ContextInvestigatorConfig"]
