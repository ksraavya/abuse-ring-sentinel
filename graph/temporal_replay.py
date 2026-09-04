from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from events.schema import (
    AccountCreatedEvent,
    AccountUpdatedEvent,
    TransactionEvent,
)
from features.temporal import TEMPORAL_FEATURE_NAMES, TemporalFeatureState
from features.transaction_local import FEATURE_COLUMNS as LOCAL_FEATURE_COLUMNS
from features.transaction_local import flat_event_to_feature_row
from graph.behavioral_state import BehavioralState
from graph.infrastructure_state import InfrastructureState

TEMPORAL_MODEL_FEATURE_COLUMNS: tuple[str, ...] = (
    *LOCAL_FEATURE_COLUMNS,
    *TEMPORAL_FEATURE_NAMES,
)

if len(TEMPORAL_MODEL_FEATURE_COLUMNS) != 26:
    raise RuntimeError("Temporal replay must produce exactly 26 model features")


@dataclass
class TemporalReplayState:
    """All event-time state required by the Temporal detector.

    InfrastructureState is deliberately retained even though its seven raw
    Baseline-B columns are not part of the Temporal classifier. It provides
    current device/IP peer context to the broader graph layer.
    """

    infrastructure: InfrastructureState = field(default_factory=InfrastructureState)
    behavioral: BehavioralState = field(default_factory=BehavioralState)
    temporal_features: TemporalFeatureState = field(default_factory=TemporalFeatureState)


class TemporalReplay:
    """Causal event-time replay for Temporal model training/evaluation.

    For every transaction the ordering is structurally fixed as:

        1. read pre-T infrastructure/behavioral/history state
        2. extract 10 local + 16 temporal features
        3. call the score callback, if supplied
        4. update behavioral and temporal state

    Account lifecycle events update InfrastructureState before later
    transactions can observe their current infrastructure context. Transactions
    never mutate InfrastructureState.
    """

    def __init__(self, state: TemporalReplayState | None = None) -> None:
        self.state = state or TemporalReplayState()
        self._previous_order_key: tuple[datetime, str] | None = None

    @staticmethod
    def _parse_event(raw: dict[str, Any]) -> AccountCreatedEvent | AccountUpdatedEvent | TransactionEvent:
        event_type = raw.get("event_type")
        if event_type == "account_created":
            return AccountCreatedEvent.model_validate(raw)
        if event_type == "account_updated":
            return AccountUpdatedEvent.model_validate(raw)
        if event_type == "transaction":
            return TransactionEvent.model_validate(raw)
        raise ValueError(f"Unknown event_type {event_type!r}")

    @staticmethod
    def _utc_iso(value: datetime) -> str:
        if value.tzinfo is None:
            raise ValueError("event timestamp must be timezone-aware")
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    def _check_order(self, event_id: str, timestamp: datetime) -> str:
        ts = self._utc_iso(timestamp)
        key = (timestamp.astimezone(timezone.utc), event_id)  # use datetime not string
        if self._previous_order_key is not None and key < self._previous_order_key:
            raise ValueError(
                "event stream is not chronological by (timestamp, event_id): "
                f"{key!r} follows {self._previous_order_key!r}"
            )
        self._previous_order_key = key
        return ts

    def process_event(
        self,
        raw: dict[str, Any],
        *,
        score_callback: Callable[[tuple[float, ...]], Any] | None = None,
    ) -> None:
        """Process one event and optionally score its pre-update feature row."""
        event = self._parse_event(raw)
        timestamp = self._check_order(event.event_id, event.timestamp)

        if isinstance(event, AccountCreatedEvent):
            self.state.infrastructure.add_or_update(
                event.account_id,
                event.device_id,
                event.ip_prefix,
            )
            return None

        if isinstance(event, AccountUpdatedEvent):
            self.state.infrastructure.add_or_update(
                event.account_id,
                event.new_device_id,
                event.new_ip_prefix,
            )
            return None

        local_features = flat_event_to_feature_row(raw)
        temporal_features = self.state.temporal_features.extract_features(
            event,
            timestamp,
            self.state.behavioral,
        )
        row = local_features + tuple(
            float(temporal_features[name]) for name in TEMPORAL_FEATURE_NAMES
        )
        if len(row) != 26:
            raise RuntimeError(f"Temporal feature row has {len(row)} columns, expected 26")

        # CRITICAL CAUSAL BOUNDARY: score before any current-transaction state
        # mutation. The callback is where a live detector would call predict.
        if score_callback is not None:
            score_callback(row)

        # Only after scoring can the current transaction become historical
        # information for future events.
        self.state.temporal_features.update(event)
        if event.counterparty_account_id is not None:
            self.state.behavioral.update(
                sender=event.account_id,
                receiver=event.counterparty_account_id,
                timestamp=timestamp,
                amount=float(event.amount),
            )

        return None


def load_ground_truth(path: Path) -> dict[str, bool]:
    import json

    labels: dict[str, bool] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            event_id = raw.get("event_id")
            if not event_id:
                raise ValueError(f"Ground truth line {line_number} has no event_id")
            if event_id in labels:
                raise ValueError(f"Duplicate ground-truth event_id: {event_id}")
            labels[event_id] = bool(raw["is_fraud"])
    return labels


def count_transactions(events_path: Path) -> int:
    import json

    count = 0
    with events_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            if raw.get("event_type") == "transaction":
                count += 1
    return count


def replay_world(
    events_path: Path,
    ground_truth: dict[str, bool],
    *,
    on_transaction: Callable[[str, str, tuple[float, ...], bool], None],
    state: TemporalReplayState | None = None,
) -> int:
    """Replay a complete world with a hard pre-update callback boundary.

    ``on_transaction`` is invoked from inside ``process_event``'s scoring
    callback, immediately after the 26 pre-T features are built and before the
    current transaction is committed to behavioral/history state. The replay
    engine then performs the update itself. This makes the causal ordering
    structural rather than a convention the training script can accidentally
    violate.
    """
    import json

    replay = TemporalReplay(state)
    transaction_count = 0
    seen_transaction_ids: set[str] = set()

    with events_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if raw.get("event_type") != "transaction":
                replay.process_event(raw)
                continue

            event_id = raw.get("event_id")
            if not event_id or event_id not in ground_truth:
                raise ValueError(
                    f"Missing ground truth for transaction {event_id!r} at line {line_number}"
                )
            if event_id in seen_transaction_ids:
                raise ValueError(f"Duplicate transaction event_id {event_id}")

            event = replay._parse_event(raw)
            label = ground_truth[event_id]

            def score_and_collect(row: tuple[float, ...]) -> None:
                # This executes before TemporalReplay.process_event mutates
                # behavioral/history state for the current transaction.
                on_transaction(
                    event_id,
                    replay._utc_iso(event.timestamp),
                    row,
                    label,
                )

            replay.process_event(raw, score_callback=score_and_collect)
            seen_transaction_ids.add(event_id)
            transaction_count += 1

    missing = set(ground_truth) - seen_transaction_ids
    if missing:
        raise ValueError(
            f"Ground truth contains {len(missing):,} IDs absent from transactions"
        )
    return transaction_count
