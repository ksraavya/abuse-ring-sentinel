from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta

import numpy as np

from .population import Account
from .schema import CoordinationStrength, RingKind, RingSpec, WorldConfig


@dataclass(frozen=True)
class RingPlan:
    spec: RingSpec
    accounts: tuple[Account, ...]
    behavioral_pairs: tuple[tuple[str, str], ...]


def _topology_assignments(
    rng: np.random.Generator,
    count: int,
    weights: dict[str, float],
) -> list[str]:
    """Allocate topology counts from configured weights, then shuffle assignments."""
    keys = list(weights)
    positive = [k for k in keys if weights[k] > 0]
    if count <= 0:
        return []
    if not positive:
        raise ValueError("at least one topology must have positive weight")

    raw = np.asarray([weights[k] for k in keys], dtype=float)
    raw /= raw.sum()
    expected = raw * count
    counts = np.floor(expected).astype(int)
    remainder = count - int(counts.sum())
    order = np.argsort(-(expected - counts))
    for idx in order[:remainder]:
        counts[idx] += 1

    # If there are enough rings, positive configured topologies get at least one.
    if count >= len(positive):
        for key in positive:
            idx = keys.index(key)
            if counts[idx] == 0:
                donor = int(np.argmax(counts))
                if counts[donor] <= 1:
                    continue
                counts[donor] -= 1
                counts[idx] += 1

    assignments = [key for key, n in zip(keys, counts) for _ in range(int(n))]
    rng.shuffle(assignments)
    return assignments


def _pair_edges(account_ids: list[str], topology: str, rng: np.random.Generator) -> list[tuple[str, str]]:
    if len(account_ids) < 2:
        return []
    edges: set[tuple[str, str]] = set()

    def add(a: str, b: str) -> None:
        if a != b:
            edges.add(tuple(sorted((a, b))))

    shuffled = list(rng.permutation(account_ids))
    if topology == "star":
        center = shuffled[0]
        for member in shuffled[1:]:
            add(center, member)
        for _ in range(max(1, len(shuffled) // 4)):
            a, b = rng.choice(shuffled[1:], size=2, replace=False)
            add(str(a), str(b))
    elif topology == "chain":
        for a, b in zip(shuffled, shuffled[1:]):
            add(a, b)
        for _ in range(max(1, len(shuffled) // 5)):
            a, b = rng.choice(shuffled, size=2, replace=False)
            add(str(a), str(b))
    elif topology == "cluster":
        groups = np.array_split(shuffled, 3)
        for group in groups:
            group = list(group)
            for i, a in enumerate(group):
                for b in group[i + 1:]:
                    if rng.random() < 0.42:
                        add(a, b)
        # Sparse bridges prevent a complete-clique shortcut.
        for a, b in zip(shuffled, shuffled[1:]):
            if rng.random() < 0.70:
                add(a, b)
    else:  # distributed: sparse random behavioral graph
        target_edges = max(len(shuffled), int(len(shuffled) * 1.35))
        while len(edges) < target_edges:
            a, b = rng.choice(shuffled, size=2, replace=False)
            add(str(a), str(b))
    return sorted(edges)


def _apply_infrastructure(
    rng: np.random.Generator,
    world_prefix: str,
    ring_index: int,
    accounts: list[Account],
    topology: str,
) -> list[Account]:
    """Create observable infrastructure overlap independently from behavioral edges."""
    if topology == "distributed":
        return [
            replace(
                a,
                device_id=f"{world_prefix}-ring-unique-device-{ring_index:04d}-{i:03d}",
                ip_prefix=f"10.250.{ring_index % 256}.{i}/32",
            )
            for i, a in enumerate(accounts)
        ]

    result = list(accounts)
    if topology == "star":
        # A subset shares the hub's device; another subset shares an IP range.
        hub_device = f"{world_prefix}-ring-{ring_index:04d}-hub-device"
        hub_ip = f"100.64.{ring_index % 256}.0/24"
        for i in range(1, len(result)):
            if rng.random() < 0.65:
                result[i] = replace(result[i], device_id=hub_device)
            elif rng.random() < 0.55:
                result[i] = replace(result[i], ip_prefix=hub_ip)
        result[0] = replace(result[0], device_id=hub_device, ip_prefix=hub_ip)

    elif topology == "chain":
        # Adjacent accounts share one infrastructure attribute, alternating
        # between device and IP so the infrastructure graph itself is chain-like.
        for i in range(len(result) - 1):
            if i % 2 == 0:
                shared = f"{world_prefix}-ring-{ring_index:04d}-pair-device-{i // 2:03d}"
                result[i] = replace(result[i], device_id=shared)
                result[i + 1] = replace(result[i + 1], device_id=shared)
            else:
                shared = f"100.65.{ring_index % 256}.{i // 2}/31"
                result[i] = replace(result[i], ip_prefix=shared)
                result[i + 1] = replace(result[i + 1], ip_prefix=shared)

    else:  # cluster
        groups = np.array_split(np.arange(len(result)), 3)
        for g, group in enumerate(groups):
            shared_device = f"{world_prefix}-ring-{ring_index:04d}-cluster-device-{g}"
            shared_ip = f"100.66.{ring_index % 256}.{g}/24"
            for i in group:
                result[int(i)] = replace(result[int(i)], device_id=shared_device, ip_prefix=shared_ip)

    return result

def build_ring_plans(
    rng: np.random.Generator,
    world_prefix: str,
    config: WorldConfig,
    ring_accounts: list[Account],
    start_day: float,
    start: datetime,
) -> list[RingPlan]:
    plans: list[RingPlan] = []
    cursor = 0
    topologies = _topology_assignments(rng, config.total_rings, config.topology_distribution)

    for kind, count in (
        (RingKind.FAST, config.fast_forming_rings),
        (RingKind.SLOW_BURN, config.slow_burn_rings),
        (RingKind.OBFUSCATED, config.obfuscated_rings),
    ):
        for _ in range(count):
            ring_index = len(plans)
            size = int(rng.integers(config.ring_size_min, config.ring_size_max + 1))
            selected = ring_accounts[cursor:cursor + size]
            cursor += size
            if len(selected) < size:
                raise ValueError("not enough ring accounts for configured ring count")

            topology = topologies[ring_index]
            if kind is RingKind.FAST:
                coordination_length = float(rng.uniform(0.25, 1.5))
                activation_gap = float(rng.uniform(0.05, 0.5))
                participation = float(rng.uniform(0.72, 0.96))
                churn = float(rng.uniform(0.02, 0.10))
            elif kind is RingKind.SLOW_BURN:
                coordination_length = float(rng.uniform(7.0, 24.0))
                activation_gap = float(rng.uniform(2.0, 12.0))
                participation = float(rng.uniform(0.55, 0.82))
                churn = float(rng.uniform(0.05, 0.18))
            else:
                coordination_length = float(rng.uniform(3.0, 12.0))
                activation_gap = float(rng.uniform(1.0, 7.0))
                participation = float(rng.uniform(0.50, 0.78))
                churn = float(rng.uniform(0.06, 0.20))

            strength = [
                CoordinationStrength.WEAK,
                CoordinationStrength.MODERATE,
                CoordinationStrength.STRONG,
            ][int(rng.choice(3, p=[0.25, 0.50, 0.25]))]
            pre_coord_gap = float(rng.uniform(0.5, 5.0))
            required = pre_coord_gap + coordination_length + activation_gap + 1.0
            if required >= config.duration_days:
                raise ValueError(
                    f"duration_days={config.duration_days} is too short for {kind.value} ring lifecycle; "
                    f"need at least {required:.1f} days"
                )
            latest_creation = config.duration_days - required
            creation_start = float(start_day + rng.uniform(0.25, max(0.26, latest_creation)))
            coordination_start = creation_start + pre_coord_gap
            coordination_end = coordination_start + coordination_length
            activation_day = coordination_end + activation_gap

            upper_creation = max(0.06, coordination_start - 0.10)
            lower_creation = min(max(0.05, creation_start - 0.25), upper_creation - 0.01)
            adjusted: list[Account] = []
            for account in selected:
                created_day = float(rng.uniform(lower_creation, upper_creation))
                adjusted.append(replace(account, created_at=start + timedelta(days=created_day)))

            adjusted = _apply_infrastructure(rng, world_prefix, ring_index, adjusted, topology)

            # Churn is a real lifecycle event, not merely exclusion from fraud.
            # It is allowed for every ring kind and is independent of topology.
            for i, account in enumerate(adjusted):
                if rng.random() < churn:
                    churn_low = coordination_start + 0.15
                    churn_high = max(churn_low + 0.01, activation_day - 0.15)
                    churn_day = float(rng.uniform(churn_low, churn_high))
                    adjusted[i] = replace(
                        account,
                        churn_at=start + timedelta(days=churn_day),
                        post_churn_device_id=f"{world_prefix}-churn-device-{ring_index:04d}-{i:03d}",
                        post_churn_ip_prefix=f"100.127.{ring_index % 256}.{i}/32",
                    )

            if not any(a.churn_at is not None for a in adjusted):
                # Keep churn a lifecycle variation rather than a topology marker,
                # while ensuring every generated ring kind has at least one
                # observable churn case when that kind is present.
                force_i = int(rng.integers(0, len(adjusted)))
                account = adjusted[force_i]
                churn_low = coordination_start + 0.15
                churn_high = max(churn_low + 0.01, activation_day - 0.15)
                churn_day = float((churn_low + churn_high) / 2.0)
                adjusted[force_i] = replace(
                    account,
                    churn_at=start + timedelta(days=churn_day),
                    post_churn_device_id=f"{world_prefix}-churn-device-{ring_index:04d}-{force_i:03d}",
                    post_churn_ip_prefix=f"100.127.{ring_index % 256}.{force_i}/32",
                )

            pairs = tuple(_pair_edges([a.account_id for a in adjusted], topology, rng))
            spec = RingSpec(
                ring_id=f"{world_prefix}-ring-{ring_index:04d}",
                kind=kind,
                topology=topology,
                account_ids=[a.account_id for a in adjusted],
                creation_start_day=creation_start,
                activation_day=activation_day,
                coordination_start_day=coordination_start,
                coordination_end_day=coordination_end,
                strength=CoordinationStrength(strength),
                churn_probability=churn,
                participant_probability=participation,
            )
            plans.append(RingPlan(spec=spec, accounts=tuple(adjusted), behavioral_pairs=pairs))
    return plans
