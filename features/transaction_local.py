from __future__ import annotations

from math import log1p
from typing import Any

import numpy as np

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
)

_CHANNEL_INDEX = {
    "upi": 0,
    "card": 1,
    "wallet": 2,
    "netbanking": 3,
}


def flat_event_to_feature_row(raw: dict[str, Any]) -> tuple[float, ...]:
    """Convert one flat transaction event into the canonical Baseline A row.

    Only current-transaction fields are used. No history, graph, or
    infrastructure state is consulted.
    """
    amount = float(raw["amount"])
    timestamp = str(raw["timestamp"])
    hour = int(timestamp[11:13])
    day_of_week = _weekday_from_iso_date(timestamp[:10])
    channel = str(raw["channel"])

    row = [
        amount,
        log1p(amount),
        float(hour),
        float(day_of_week),
        float(day_of_week >= 5),
        float(hour < 6 or hour >= 22),
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    row[6 + _CHANNEL_INDEX[channel]] = 1.0
    return tuple(row)


def _weekday_from_iso_date(date_text: str) -> int:
    """Return Monday=0..Sunday=6 without constructing datetime objects."""
    year = int(date_text[0:4])
    month = int(date_text[5:7])
    day = int(date_text[8:10])

    # Sakamoto's algorithm; convert Sunday=0 to Python Monday=0.
    table = (0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4)
    y = year - (1 if month < 3 else 0)
    sunday_zero = (
        y + y // 4 - y // 100 + y // 400 + table[month - 1] + day
    ) % 7
    return (sunday_zero + 6) % 7


def rows_to_array(rows: list[tuple[float, ...]]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float32)
