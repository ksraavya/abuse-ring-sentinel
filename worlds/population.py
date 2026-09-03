from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta

import numpy as np


@dataclass(frozen=True)
class Account:
    account_id: str
    created_at: datetime
    device_id: str
    ip_prefix: str
    cohort: str
    activity_weight: float = 1.0
    churn_at: datetime | None = None
    post_churn_device_id: str | None = None
    post_churn_ip_prefix: str | None = None

    def identity_at(self, timestamp: datetime) -> tuple[str, str]:
        if self.churn_at is not None and timestamp >= self.churn_at:
            return (
                self.post_churn_device_id or self.device_id,
                self.post_churn_ip_prefix or self.ip_prefix,
            )
        return self.device_id, self.ip_prefix


@dataclass(frozen=True)
class HardNegativeCluster:
    cluster_id: str
    kind: str
    account_ids: tuple[str, ...]


def _account_time(start: datetime, day: float) -> datetime:
    return start + timedelta(days=float(day))


def build_legitimate_population(
    rng: np.random.Generator,
    world_prefix: str,
    count: int,
    start: datetime,
    duration_days: int,
    *,
    cohort: str = "legitimate",
    activity_sigma: float = 1.0,
    activity_min_weight: float = 0.10,
) -> list[Account]:
    """Bulk-generate account attributes; only dataclass construction is row-wise."""
    signup_days = rng.uniform(0.0, duration_days * 0.70, size=count)
    device_indices = rng.integers(0, max(count * 3 // 2, 1), size=count)
    ip_b = rng.integers(1, 223, size=count)
    ip_c = rng.integers(0, 256, size=count)
    weights = np.maximum(rng.lognormal(0.0, activity_sigma, size=count), activity_min_weight)

    return [
        Account(
            account_id=f"{world_prefix}-acct-{i:07d}",
            created_at=_account_time(start, signup_days[i]),
            device_id=f"{world_prefix}-dev-{int(device_indices[i]):07d}",
            ip_prefix=f"10.{int(ip_b[i])}.{int(ip_c[i])}.0/24",
            cohort=cohort,
            activity_weight=float(weights[i]),
        )
        for i in range(count)
    ]


def build_hard_negative_clusters(
    rng: np.random.Generator,
    world_prefix: str,
    start_index: int,
    counts: dict[str, int],
    start: datetime,
    duration_days: int,
) -> tuple[list[Account], list[HardNegativeCluster]]:
    accounts: list[Account] = []
    clusters: list[HardNegativeCluster] = []
    index = start_index
    cluster_sizes = {"family": (4, 7), "hostel": (20, 40), "corporate": (30, 60)}

    for kind, cluster_count in counts.items():
        for c in range(cluster_count):
            size = int(rng.integers(cluster_sizes[kind][0], cluster_sizes[kind][1] + 1))
            member_ids: list[str] = []
            shared_device = f"{world_prefix}-hn-device-{kind}-{c:04d}" if kind == "family" else None
            shared_ip = (
                f"192.168.{int(rng.integers(0, 256))}.0/24"
                if kind == "family"
                else f"172.{int(rng.integers(16, 32))}.{int(rng.integers(0, 256))}.0/24"
            )
            cluster_start = float(rng.uniform(0.0, max(1.0, duration_days * 0.45)))
            cluster_end = min(duration_days * 0.80, cluster_start + {"family": 120, "hostel": 45, "corporate": 90}[kind])

            for _ in range(size):
                day = float(rng.uniform(cluster_start, max(cluster_start + 0.01, cluster_end)))
                account = Account(
                    account_id=f"{world_prefix}-acct-{index:07d}",
                    created_at=_account_time(start, day),
                    device_id=shared_device or f"{world_prefix}-hn-dev-{kind}-{index:07d}",
                    ip_prefix=shared_ip,
                    cohort=f"hard_negative:{kind}",
                    activity_weight=float(max(0.10, rng.lognormal(0.15, 0.75))),
                )
                accounts.append(account)
                member_ids.append(account.account_id)
                index += 1

            clusters.append(HardNegativeCluster(
                cluster_id=f"{world_prefix}-hn-{kind}-{c:04d}",
                kind=kind,
                account_ids=tuple(member_ids),
            ))

    return accounts, clusters
