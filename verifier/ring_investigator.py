from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from events.schema import TransactionEvent
from graph.behavioral_state import BehavioralEdge, BehavioralState
from features.temporal import HistoricalTransaction, TemporalFeatureState

from .contracts import (
    EvidenceContext,
    EvidenceItem,
    EvidenceStrength,
    EvidenceType,
    VerificationRequest,
)


@dataclass(frozen=True)
class RingInvestigatorConfig:
    """Conservative thresholds for converting graph observations into evidence.

    These are evidence-generation thresholds, not fraud thresholds. They are
    intentionally simple and interpretable so World C can tune them later
    without changing the evidence contract.
    """

    min_neighbors_for_structure: int = 3
    min_internal_edges_for_structure: int = 2
    min_new_edges_24h: int = 2
    min_active_neighbors_6h: int = 2
    min_merchant_overlap: int = 2
    max_neighbors_scanned: int = 250


@dataclass(frozen=True)
class _RingSnapshot:
    account_id: str
    neighbors: frozenset[str]
    internal_edges: int
    internal_density: float
    reciprocal_neighbors: int
    recent_new_edges_24h: int
    recent_new_edges_7d: int
    active_neighbors_6h: int
    merchant_overlap_24h: int


class RingInvestigator:
    """Deterministic investigator for pre-decision behavioral ring evidence.

    The investigator reads only the alert and event-time state supplied through
    ``EvidenceContext``. It never mutates graph/history state and never sees
    ground truth or a ring identifier.

    Expected context state keys:
      - ``behavioral_state``: ``BehavioralState``
      - ``temporal_feature_state``: ``TemporalFeatureState``

    The investigator intentionally does not reproduce the Temporal detector's
    16-feature vector. Instead it produces corroborating structural evidence:
    local induced connectivity, reciprocal structure, graph growth, peer
    synchrony, and merchant convergence.
    """

    name = "ring-investigator"

    def __init__(self, config: RingInvestigatorConfig | None = None) -> None:
        self.config = config or RingInvestigatorConfig()

    def collect(
        self,
        request: VerificationRequest,
        context: EvidenceContext,
    ) -> list[EvidenceItem]:
        self._validate_context_time(request, context)
        behavioral = self._require_behavioral_state(context)
        temporal = self._require_temporal_state(context)

        snapshot = self._snapshot(request.alert_event, context.as_of, behavioral, temporal)
        evidence: list[EvidenceItem] = []

        if (
            len(snapshot.neighbors) >= self.config.min_neighbors_for_structure
            and snapshot.internal_edges >= self.config.min_internal_edges_for_structure
        ):
            strength = self._structure_strength(snapshot)
            evidence.append(
                self._item(
                    request,
                    EvidenceType.RING_STRUCTURE,
                    strength,
                    "The account sits in a connected behavioral neighborhood with internal peer-to-peer links.",
                    confidence=self._structure_confidence(snapshot),
                    metric_name="behavioral_neighbors",
                    metric_value=float(len(snapshot.neighbors)),
                    details={
                        "internal_edges": snapshot.internal_edges,
                        "internal_density": snapshot.internal_density,
                        "reciprocal_neighbors": snapshot.reciprocal_neighbors,
                        "window": "pre-decision graph state",
                    },
                )
            )

        if snapshot.recent_new_edges_24h >= self.config.min_new_edges_24h:
            strength = (
                EvidenceStrength.STRONG
                if snapshot.recent_new_edges_24h >= 4
                else EvidenceStrength.MODERATE
            )
            confidence = min(0.98, 0.55 + 0.08 * snapshot.recent_new_edges_24h)
            evidence.append(
                self._item(
                    request,
                    EvidenceType.BEHAVIORAL_ACCELERATION,
                    strength,
                    "The behavioral neighborhood has acquired multiple new peer relationships recently.",
                    confidence=confidence,
                    metric_name="new_behavioral_edges_24h",
                    metric_value=float(snapshot.recent_new_edges_24h),
                    details={
                        "new_edges_7d": snapshot.recent_new_edges_7d,
                        "window_24h": "[T-24h, T)",
                        "window_7d": "[T-7d, T)",
                    },
                )
            )

        if snapshot.active_neighbors_6h >= self.config.min_active_neighbors_6h:
            strength = (
                EvidenceStrength.STRONG
                if snapshot.active_neighbors_6h >= 4
                else EvidenceStrength.MODERATE
            )
            confidence = min(0.95, 0.55 + 0.08 * snapshot.active_neighbors_6h)
            evidence.append(
                self._item(
                    request,
                    EvidenceType.PEER_SYNCHRONY,
                    strength,
                    "Multiple behavioral peers were active during the same recent six-hour window.",
                    confidence=confidence,
                    metric_name="active_behavioral_neighbors_6h",
                    metric_value=float(snapshot.active_neighbors_6h),
                    details={"window": "[T-6h, T)"},
                )
            )

        if snapshot.merchant_overlap_24h >= self.config.min_merchant_overlap:
            strength = (
                EvidenceStrength.STRONG
                if snapshot.merchant_overlap_24h >= 4
                else EvidenceStrength.MODERATE
            )
            confidence = min(0.95, 0.55 + 0.08 * snapshot.merchant_overlap_24h)
            evidence.append(
                self._item(
                    request,
                    EvidenceType.MERCHANT_CONVERGENCE,
                    strength,
                    "The account and its behavioral peers share multiple recently observed merchants.",
                    confidence=confidence,
                    metric_name="shared_merchants_24h",
                    metric_value=float(snapshot.merchant_overlap_24h),
                    details={"window": "[T-24h, T)"},
                )
            )

        return evidence

    @staticmethod
    def _validate_context_time(
        request: VerificationRequest,
        context: EvidenceContext,
    ) -> None:
        if context.as_of > request.decision_time:
            raise ValueError("EvidenceContext.as_of cannot be after verifier decision_time")
        event_time = request.alert_event.timestamp.astimezone(timezone.utc)
        if context.as_of != event_time:
            raise ValueError(
                "RingInvestigator requires EvidenceContext.as_of to equal alert event time"
            )

    @staticmethod
    def _require_behavioral_state(context: EvidenceContext) -> BehavioralState:
        state = context.state.get("behavioral_state")
        if not isinstance(state, BehavioralState):
            raise TypeError("context.state['behavioral_state'] must be BehavioralState")
        return state

    @staticmethod
    def _require_temporal_state(context: EvidenceContext) -> TemporalFeatureState:
        state = context.state.get("temporal_feature_state")
        if not isinstance(state, TemporalFeatureState):
            raise TypeError(
                "context.state['temporal_feature_state'] must be TemporalFeatureState"
            )
        return state

    def _snapshot(
        self,
        event: TransactionEvent,
        as_of: datetime,
        behavioral: BehavioralState,
        temporal: TemporalFeatureState,
    ) -> _RingSnapshot:
        account = str(event.account_id)
        neighbors = behavioral.get_neighbors(account)
        if len(neighbors) > self.config.max_neighbors_scanned:
            # Deterministic truncation: stable account-id ordering rather than
            # arbitrary set iteration. This bounds worst-case investigation cost.
            neighbors = set(sorted(neighbors)[: self.config.max_neighbors_scanned])

        node_set = neighbors | {account}
        internal_edges = 0
        reciprocal_neighbors = 0
        for sender, receiver in behavioral.edges:
            if sender in node_set and receiver in node_set and sender != receiver:
                internal_edges += 1

        for neighbor in neighbors:
            if behavioral.get_edge(account, neighbor) is not None and behavioral.get_edge(
                neighbor, account
            ) is not None:
                reciprocal_neighbors += 1

        recent_new_edges_24h = self._count_new_edges(
            account, neighbors, behavioral, as_of, timedelta(hours=24)
        )
        recent_new_edges_7d = self._count_new_edges(
            account, neighbors, behavioral, as_of, timedelta(days=7)
        )

        active_neighbors_6h = 0
        account_merchants = self._merchant_set(temporal, account, as_of, timedelta(hours=24))
        neighbor_merchants: set[str] = set()
        for neighbor in neighbors:
            history = temporal.account_events.get(neighbor, [])
            if self._has_recent_activity(history, as_of, timedelta(hours=6)):
                active_neighbors_6h += 1
            neighbor_merchants.update(
                merchant
                for merchant in self._merchant_set(
                    temporal, neighbor, as_of, timedelta(hours=24)
                )
                if merchant is not None
            )

        n = len(node_set)
        possible_directed_edges = n * (n - 1)
        internal_density = (
            internal_edges / possible_directed_edges
            if possible_directed_edges > 0
            else 0.0
        )

        return _RingSnapshot(
            account_id=account,
            neighbors=frozenset(neighbors),
            internal_edges=internal_edges,
            internal_density=internal_density,
            reciprocal_neighbors=reciprocal_neighbors,
            recent_new_edges_24h=recent_new_edges_24h,
            recent_new_edges_7d=recent_new_edges_7d,
            active_neighbors_6h=active_neighbors_6h,
            merchant_overlap_24h=len(account_merchants & neighbor_merchants),
        )

    @staticmethod
    def _count_new_edges(
        account: str,
        neighbors: set[str],
        behavioral: BehavioralState,
        as_of: datetime,
        window: timedelta,
    ) -> int:
        start = as_of - window
        count = 0
        for neighbor in neighbors:
            edge_out = behavioral.get_edge(account, neighbor)
            edge_in = behavioral.get_edge(neighbor, account)
            first_seen = [
                RingInvestigator._parse_state_timestamp(edge.first_seen)
                for edge in (edge_out, edge_in)
                if edge is not None and edge.first_seen is not None
            ]
            if first_seen and min(first_seen) >= start and min(first_seen) < as_of:
                count += 1
        return count

    @staticmethod
    def _has_recent_activity(
        history: list[HistoricalTransaction],
        as_of: datetime,
        window: timedelta,
    ) -> bool:
        start = as_of - window
        # TemporalFeatureState guarantees chronological insertion.
        return bool(history and start <= history[-1].timestamp < as_of)

    @staticmethod
    def _merchant_set(
        temporal: TemporalFeatureState,
        account: str,
        as_of: datetime,
        window: timedelta,
    ) -> set[str]:
        start = as_of - window
        return {
            str(row.merchant_id)
            for row in temporal.merchant_events.get(account, [])
            if row.merchant_id is not None and start <= row.timestamp < as_of
        }

    @staticmethod
    def _parse_state_timestamp(value: str) -> datetime:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("behavioral edge timestamps must be timezone-aware")
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _structure_strength(snapshot: _RingSnapshot) -> EvidenceStrength:
        if snapshot.reciprocal_neighbors >= 2 and snapshot.internal_edges >= 4:
            return EvidenceStrength.STRONG
        if snapshot.internal_edges >= 2:
            return EvidenceStrength.MODERATE
        return EvidenceStrength.WEAK

    @staticmethod
    def _structure_confidence(snapshot: _RingSnapshot) -> float:
        value = 0.45
        value += min(0.20, 0.03 * max(0, len(snapshot.neighbors) - 2))
        value += min(0.20, 0.04 * snapshot.internal_edges)
        value += min(0.15, 0.05 * snapshot.reciprocal_neighbors)
        return min(0.98, value)

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
        details: dict[str, Any],
    ) -> EvidenceItem:
        # 8A/8B state currently stores aggregate graph/history metadata rather
        # than every contributing event ID. The alert event is therefore the
        # auditable anchor for this event-time evidence item; detailed windows
        # and metrics identify the historical state used to derive it.
        return EvidenceItem(
            evidence_id=f"{request.event_id}:{evidence_type.value}",
            evidence_type=evidence_type,
            strength=strength,
            source_agent=RingInvestigator.name,
            summary=summary,
            confidence=max(0.0, min(1.0, confidence)),
            observed_at=request.alert_event.timestamp,
            source_event_ids=(request.event_id,),
            subject_account_ids=(str(request.alert_event.account_id),),
            metric_name=metric_name,
            metric_value=metric_value,
            details=details,
        )


__all__ = ["RingInvestigator", "RingInvestigatorConfig"]
