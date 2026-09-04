from __future__ import annotations

from typing import Any

from features.transaction_local import flat_event_to_feature_row
from graph.infrastructure_state import InfrastructureState


INFRASTRUCTURE_FEATURE_COLUMNS: tuple[str, ...] = (
    "degree",
    "device_degree",
    "ip_degree",
    "shared_device_accounts",
    "shared_ip_accounts",
    "max_device_sharing",
    "max_ip_sharing",
)

FEATURE_COLUMNS: tuple[str, ...] = (
    "amount",
    "amount_log",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_night",
    "channel_upi",
    "channel_card",
    "channel_wallet",
    "channel_netbanking",
    *INFRASTRUCTURE_FEATURE_COLUMNS,
)


def event_to_feature_row(
    raw: dict[str, Any],
    state: InfrastructureState,
) -> tuple[float, ...]:
    """Create B features from state strictly before the current transaction.

    The current transaction itself does not mutate InfrastructureState.
    """
    local = flat_event_to_feature_row(raw)
    infra = state.features(str(raw["account_id"]))
    return local + tuple(float(infra[name]) for name in INFRASTRUCTURE_FEATURE_COLUMNS)
