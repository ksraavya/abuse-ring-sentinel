from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from events.schema import TransactionEvent
from graph.behavioral_state import BehavioralState


# ---------------------------------------------------------------------------
# Frozen Commit 8B temporal feature contract
# ---------------------------------------------------------------------------
# All features are evaluated at the exact event timestamp T and use only
# observations in half-open historical intervals [start, T). The current
# transaction is therefore never available to its own feature vector.
TEMPORAL_FEATURE_NAMES: tuple[str, ...] = (
    "p2p_txn_count_24h",
    "p2p_unique_neighbors_24h",
    "p2p_unique_neighbors_14d",
    "p2p_amount_24h",
    "activity_count_6h",
    "activity_count_24h",
    "days_since_last_txn",
    "new_neighbors_14d",
    "new_neighbors_24h",
    "edge_creation_acceleration",
    "reciprocal_neighbors_14d",
    "neighbor_activity_6h",
    "neighbor_activity_24h",
    "cluster_activity_synchrony_6h",
    "neighbor_merchant_overlap_24h",
    "activity_acceleration_6h_vs_7d",
)

WINDOW_6H = timedelta(hours=6)
WINDOW_24H = timedelta(hours=24)
WINDOW_7D = timedelta(days=7)
WINDOW_14D = timedelta(days=14)
WINDOW_13D = timedelta(days=13)


def _parse_timestamp(value: datetime | str) -> datetime:
    """Parse a timestamp and normalize it to timezone-aware UTC."""
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class HistoricalTransaction:
    """Minimal immutable transaction observation stored by the feature layer."""

    timestamp: datetime
    account_id: str
    merchant_id: str | None
    counterparty_account_id: str | None
    amount: float


@dataclass
class TemporalFeatureState:
    """Time-indexed historical observations used by temporal features.

    This is deliberately separate from BehavioralState. BehavioralState owns
    graph topology and edge metadata. This class owns chronological indexes
    and rolling-window queries. Neither class performs model scoring.

    Commit 8C is responsible for the causal lifecycle:

        extract_features(...) -> score -> update(...)

    In normal replay, update() is therefore called only after the current
    event has been scored.
    """

    # Every transaction involving an account. For P2P, the observation is
    # indexed for both sender and receiver because activity is "involving".
    account_events: dict[str, list[HistoricalTransaction]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # Directed P2P transaction indexes.
    p2p_out_events: dict[str, list[HistoricalTransaction]] = field(
        default_factory=lambda: defaultdict(list)
    )
    p2p_in_events: dict[str, list[HistoricalTransaction]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # Merchant-directed transactions, keyed by sender.
    merchant_events: dict[str, list[HistoricalTransaction]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def update(self, event: TransactionEvent) -> None:
        """Record one completed transaction.

        The method expects chronological replay. It must be called only after
        the event's feature extraction and scoring step.
        """
        timestamp = _parse_timestamp(event.timestamp)
        observation = HistoricalTransaction(
            timestamp=timestamp,
            account_id=event.account_id,
            merchant_id=event.merchant_id,
            counterparty_account_id=event.counterparty_account_id,
            amount=float(event.amount),
        )

        if event.counterparty_account_id is not None:
            receiver = str(event.counterparty_account_id)
            sender = str(event.account_id)
            if sender == receiver:
                raise ValueError("self-transactions are not valid behavioral events")

            _append_sorted(self.account_events[sender], observation)
            _append_sorted(self.account_events[receiver], observation)
            _append_sorted(self.p2p_out_events[sender], observation)
            _append_sorted(self.p2p_in_events[receiver], observation)
            return

        if event.merchant_id is not None:
            _append_sorted(self.account_events[str(event.account_id)], observation)
            _append_sorted(self.merchant_events[str(event.account_id)], observation)
            return

        raise ValueError("transaction must have a merchant or counterparty")

    def extract_features(
        self,
        event: TransactionEvent,
        timestamp: datetime | str,
        behavioral_state: BehavioralState,
    ) -> dict[str, float]:
        """Extract the 16-feature vector from strictly pre-T state.

        ``timestamp`` must equal ``event.timestamp`` exactly. The method is
        read-only: it does not mutate either the temporal state or graph state.
        """
        t = _parse_timestamp(timestamp)
        event_t = _parse_timestamp(event.timestamp)
        if event_t != t:
            raise ValueError("timestamp must exactly match event.timestamp")

        account = str(event.account_id)
        account_history = self.account_events.get(account, [])
        p2p_out = self.p2p_out_events.get(account, [])

        p2p_24h = self._slice(p2p_out, t, WINDOW_24H)
        activity_6h = self._slice(account_history, t, WINDOW_6H)
        activity_24h = self._slice(account_history, t, WINDOW_24H)

        # Historical baseline for account activity excludes the current
        # 6-hour interval. This avoids using the same recent observations in
        # both numerator and denominator.
        activity_prior_7d = self._slice_between(
            account_history,
            t - WINDOW_7D,
            t - WINDOW_6H,
        )

        p2p_14d = self._slice(p2p_out, t, WINDOW_14D)
        p2p_24h_new = {
            str(row.counterparty_account_id)
            for row in p2p_24h
            if row.counterparty_account_id is not None
        }
        p2p_14d_neighbors = {
            str(row.counterparty_account_id)
            for row in p2p_14d
            if row.counterparty_account_id is not None
        }

        # BehavioralState is the source of truth for whether an observed
        # directed behavioral edge exists and when it was first observed.
        new_neighbors_14d = sum(
            1
            for neighbor in p2p_14d_neighbors
            if _edge_first_seen_in_window(
                behavioral_state, account, neighbor, t, WINDOW_14D
            )
        )
        new_neighbors_24h = sum(
            1
            for neighbor in p2p_24h_new
            if _edge_first_seen_in_window(
                behavioral_state, account, neighbor, t, WINDOW_24H
            )
        )

        # Compare recent 24-hour new-edge rate with the preceding 13-day
        # new-edge rate. The two windows are adjacent:
        #   recent: [T-24h, T)
        #   baseline: [T-14d, T-24h)
        # A value of 1 means equal rates; >1 means acceleration; <1 means
        # deceleration. If the historical rate is zero, a nonzero recent
        # count is represented by a finite rate ratio using one virtual
        # historical edge as a conservative floor.
        prior_13d_new_neighbors = sum(
            1
            for neighbor in {
                str(row.counterparty_account_id)
                for row in self._slice_between(
                    p2p_out,
                    t - WINDOW_14D,
                    t - WINDOW_24H,
                )
                if row.counterparty_account_id is not None
            }
            if _edge_first_seen_in_interval(
                behavioral_state,
                account,
                neighbor,
                t - WINDOW_14D,
                t - WINDOW_24H,
            )
        )

        # Since recent is one day and baseline is thirteen days:
        # recent daily rate / baseline daily rate = 13 * recent / baseline.
        edge_creation_acceleration = (
            float(13 * new_neighbors_24h / prior_13d_new_neighbors)
            if prior_13d_new_neighbors > 0
            else (float(new_neighbors_24h) if new_neighbors_24h == 0 else 13.0 * new_neighbors_24h)
        )

        reciprocal_neighbors_14d = 0
        incoming_14d = self._slice(
            self.p2p_in_events.get(account, []),
            t,
            WINDOW_14D,
        )
        incoming_senders = {
            str(row.account_id)
            for row in incoming_14d
        }
        reciprocal_neighbors_14d = len(
            p2p_14d_neighbors & incoming_senders
        )

        # Known behavioral neighbors are read from pre-T graph state. For
        # neighbor activity, both incoming and outgoing transactions count
        # because the feature measures activity involving the peer.
        known_neighbors = behavioral_state.get_neighbors(account)
        neighbor_activity_6h = 0
        neighbor_activity_24h = 0

        account_merchants = {
            row.merchant_id
            for row in self._slice(
                self.merchant_events.get(account, []),
                t,
                WINDOW_24H,
            )
            if row.merchant_id is not None
        }
        neighbor_merchants: set[str] = set()

        active_neighbors_6h = 0
        for neighbor in known_neighbors:
            neighbor_history = self.account_events.get(str(neighbor), [])
            neighbor_6h = self._slice(neighbor_history, t, WINDOW_6H)
            neighbor_24h = self._slice(neighbor_history, t, WINDOW_24H)

            if neighbor_6h:
                active_neighbors_6h += 1
            neighbor_activity_6h += len(neighbor_6h)
            neighbor_activity_24h += len(neighbor_24h)

            neighbor_merchants.update(
                row.merchant_id
                for row in self._slice(
                    self.merchant_events.get(str(neighbor), []),
                    t,
                    WINDOW_24H,
                )
                if row.merchant_id is not None
            )

        # Fraction of known behavioral neighbors that were active in the
        # recent six-hour window. With no known neighbors, synchrony is zero.
        cluster_activity_synchrony_6h = (
            float(active_neighbors_6h / len(known_neighbors))
            if known_neighbors
            else 0.0
        )

        # Activity acceleration compares current 6h activity with the
        # preceding 162h (27 six-hour periods). A one-event floor prevents
        # division by zero while keeping the feature finite and deterministic.
        expected_6h_activity = len(activity_prior_7d) / 27.0
        activity_acceleration = (
            float(len(activity_6h) / expected_6h_activity)
            if expected_6h_activity > 0
            else float(len(activity_6h))
        )

        last_txn = account_history[-1].timestamp if account_history else None
        days_since_last_txn = (
            float((t - last_txn).total_seconds() / 86400.0)
            if last_txn is not None and last_txn < t
            else 0.0
        )

        features = {
            "p2p_txn_count_24h": float(len(p2p_24h)),
            "p2p_unique_neighbors_24h": float(len(p2p_24h_new)),
            "p2p_unique_neighbors_14d": float(len(p2p_14d_neighbors)),
            "p2p_amount_24h": float(sum(row.amount for row in p2p_24h)),
            "activity_count_6h": float(len(activity_6h)),
            "activity_count_24h": float(len(activity_24h)),
            "days_since_last_txn": days_since_last_txn,
            "new_neighbors_14d": float(new_neighbors_14d),
            "new_neighbors_24h": float(new_neighbors_24h),
            "edge_creation_acceleration": float(edge_creation_acceleration),
            "reciprocal_neighbors_14d": float(reciprocal_neighbors_14d),
            "neighbor_activity_6h": float(neighbor_activity_6h),
            "neighbor_activity_24h": float(neighbor_activity_24h),
            "cluster_activity_synchrony_6h": cluster_activity_synchrony_6h,
            "neighbor_merchant_overlap_24h": float(
                len(account_merchants & neighbor_merchants)
            ),
            "activity_acceleration_6h_vs_7d": activity_acceleration,
        }
        validate_temporal_feature_contract(features)
        return features

    @staticmethod
    def _slice(
        rows: list[HistoricalTransaction],
        t: datetime,
        window: timedelta,
    ) -> list[HistoricalTransaction]:
        return TemporalFeatureState._slice_between(rows, t - window, t)

    @staticmethod
    def _slice_between(
        rows: list[HistoricalTransaction],
        start: datetime,
        end: datetime,
    ) -> list[HistoricalTransaction]:
        """Return rows in the half-open interval [start, end)."""
        if not rows:
            return []
        left = bisect_left(rows, start, key=lambda row: row.timestamp)
        right = bisect_left(rows, end, key=lambda row: row.timestamp)
        return rows[left:right]


def _append_sorted(
    rows: list[HistoricalTransaction],
    observation: HistoricalTransaction,
) -> None:
    """Append only in chronological replay order."""
    if rows and observation.timestamp < rows[-1].timestamp:
        raise ValueError(
            "TemporalFeatureState.update requires chronological replay"
        )
    rows.append(observation)


def _edge_first_seen_in_window(
    state: BehavioralState,
    account: str,
    neighbor: str,
    t: datetime,
    window: timedelta,
) -> bool:
    edge = state.get_edge(account, neighbor)
    if edge is None or edge.first_seen is None:
        return False
    first_seen = _parse_timestamp(edge.first_seen)
    return t - window <= first_seen < t


def _edge_first_seen_in_interval(
    state: BehavioralState,
    account: str,
    neighbor: str,
    start: datetime,
    end: datetime,
) -> bool:
    edge = state.get_edge(account, neighbor)
    if edge is None or edge.first_seen is None:
        return False
    first_seen = _parse_timestamp(edge.first_seen)
    return start <= first_seen < end


def validate_temporal_feature_contract(
    features: dict[str, float],
) -> None:
    """Fail fast if the frozen 8B feature schema changes."""
    actual = tuple(features.keys())
    if actual != TEMPORAL_FEATURE_NAMES:
        raise AssertionError(
            "Temporal feature schema changed: "
            f"expected {TEMPORAL_FEATURE_NAMES}, got {actual}"
        )
