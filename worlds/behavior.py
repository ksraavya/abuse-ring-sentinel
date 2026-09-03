from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path

import numpy as np
import orjson

from events.schema import (
    AccountCreatedEvent,
    AccountUpdatedEvent,
    EventRecord,
    TransactionChannel,
    TransactionEvent,
    TransactionGroundTruth,
)
from .population import Account, HardNegativeCluster
from .rings import RingPlan
from .schema import WorldConfig

MERCHANT_COUNT = 1000
CHANNEL_VALUES = np.array([c.value for c in TransactionChannel])
CHANNEL_PROBS = np.array([0.58, 0.22, 0.12, 0.08])
HOUR_WEIGHTS = np.array([
    0.5, 0.3, 0.2, 0.1, 0.1, 0.2,
    0.5, 1.0, 1.2, 1.0, 0.8, 0.9,
    1.5, 1.8, 1.4, 1.2, 1.3, 1.6,
    2.5, 3.5, 3.0, 2.0, 1.2, 0.8,
], dtype=float)
HOUR_WEIGHTS /= HOUR_WEIGHTS.sum()
STRENGTH_FACTOR = {"weak": 0.45, "moderate": 0.75, "strong": 1.0}


@dataclass(frozen=True)
class GeneratedWorld:
    records: list[EventRecord]
    manifest: dict


def _event_id(world_id: str, index: int) -> str:
    return f"{world_id}-event-{index:09d}"


def _iso(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _amounts(rng: np.random.Generator, n: int) -> np.ndarray:
    return np.round(np.maximum(1.0, np.exp(rng.normal(6.5, 1.2, size=n))), 2)


def _build_day_counts(total: int, days: int, rng: np.random.Generator, eligible_days: np.ndarray | None = None) -> np.ndarray:
    base = np.ones(days, dtype=float)
    day_index = np.arange(days)
    weekdays = day_index % 7
    base *= np.where(weekdays < 5, 1.08, 0.90)
    base *= 1.0 + 0.08 * np.sin(2 * np.pi * day_index / 30.0)
    if eligible_days is not None:
        base *= eligible_days.astype(float)
    if base.sum() <= 0:
        return np.zeros(days, dtype=int)
    return rng.multinomial(total, base / base.sum())


def _bulk_organic_day_dicts(
    rng: np.random.Generator,
    *,
    account_ids: np.ndarray,
    device_ids: np.ndarray,
    ip_prefixes: np.ndarray,
    activity_weights: np.ndarray,
    active_indices: np.ndarray,
    n_transactions: int,
    p2p_fraction: float,
    start: datetime,
    day: int,
    event_start: int,
    world_id: str,
) -> list[dict]:
    """High-throughput daily organic generation. Heavy work is bulk NumPy; row assembly is the unavoidable serialization step."""
    if n_transactions <= 0 or active_indices.size == 0:
        return []
    weights = activity_weights[active_indices]
    weights /= weights.sum()
    src_idx = rng.choice(active_indices, size=n_transactions, p=weights)
    is_p2p = rng.random(n_transactions) < p2p_fraction
    dst_idx = rng.choice(active_indices, size=n_transactions)
    if active_indices.size > 1:
        conflicts = is_p2p & (dst_idx == src_idx)
        while conflicts.any():
            dst_idx[conflicts] = rng.choice(active_indices, size=int(conflicts.sum()))
            conflicts = is_p2p & (dst_idx == src_idx)

    hours = rng.choice(24, size=n_transactions, p=HOUR_WEIGHTS)
    minutes = rng.integers(0, 60, size=n_transactions)
    seconds = rng.integers(0, 60, size=n_transactions)
    base = start + timedelta(days=day)
    timestamps = [base + timedelta(hours=int(h), minutes=int(m), seconds=int(s)) for h, m, s in zip(hours, minutes, seconds)]
    amounts = _amounts(rng, n_transactions)
    channels = rng.choice(CHANNEL_VALUES, size=n_transactions, p=CHANNEL_PROBS)
    merchant_idx = rng.integers(0, MERCHANT_COUNT, size=n_transactions)

    rows: list[dict] = []
    for i in range(n_transactions):
        src = int(src_idx[i])
        p2p = bool(is_p2p[i])
        rows.append({
            "event_id": _event_id(world_id, event_start + i),
            "event_type": "transaction",
            "world_id": world_id,
            "timestamp": timestamps[i],
            "_sort": timestamps[i],
            "account_id": str(account_ids[src]),
            "merchant_id": None if p2p else f"merchant-{int(merchant_idx[i]):04d}",
            "counterparty_account_id": str(account_ids[int(dst_idx[i])]) if p2p else None,
            "amount": float(amounts[i]),
            "channel": str(channels[i]),
            "device_id": str(device_ids[src]),
            "ip_prefix": str(ip_prefixes[src]),
            "_is_fraud": False,
            "_ring_id": None,
        })
    return rows


def _hard_negative_events(
    rng: np.random.Generator,
    *,
    clusters: list[HardNegativeCluster],
    lookup: dict[str, Account],
    config: WorldConfig,
    start: datetime,
) -> list[dict]:
    """Structured but legitimate P2P activity inside family/hostel/corporate clusters."""
    events: list[dict] = []
    end = start + timedelta(days=config.duration_days)
    for cluster in clusters:
        members = [lookup[a] for a in cluster.account_ids]
        if len(members) < 2:
            continue
        rate = config.hard_negative_p2p_rate_per_account_day
        if cluster.kind == "corporate":
            rate *= 0.75
        candidate_ids = np.asarray([a.account_id for a in members], dtype=object)
        for account in members:
            active_days = max(1.0, (end - account.created_at).total_seconds() / 86400)
            n = int(rng.poisson(rate * active_days))
            if n <= 0:
                continue
            for _ in range(n):
                idx = int(rng.integers(0, len(candidate_ids) - 1))
                candidates = candidate_ids[candidate_ids != account.account_id]
                counterparty_id = str(candidates[idx])
                counterparty = lookup[counterparty_id]
                lower = max(account.created_at, counterparty.created_at) + timedelta(seconds=60)
                if lower >= end:
                    continue
                ts = lower + timedelta(seconds=float(rng.uniform(0, (end - lower).total_seconds())))
                device, ip = account.identity_at(ts)
                events.append({
                    "timestamp": ts,
                    "account_id": account.account_id,
                    "device_id": device,
                    "ip_prefix": ip,
                    "amount": float(_amounts(rng, 1)[0]),
                    "channel": str(rng.choice(CHANNEL_VALUES, p=CHANNEL_PROBS)),
                    "merchant_id": None,
                    "counterparty_account_id": counterparty_id,
                    "is_fraud": False,
                    "ring_id": None,
                })
    return events


def _ring_events(
    rng: np.random.Generator,
    *,
    plans: list[RingPlan],
    account_lookup: dict[str, Account],
    config: WorldConfig,
    start: datetime,
) -> tuple[list[dict], list[dict]]:
    events: list[dict] = []
    ring_manifest: list[dict] = []

    for plan in plans:
        spec = plan.spec
        factor = STRENGTH_FACTOR[spec.strength.value]
        active = [a for a in plan.accounts if rng.random() < spec.participant_probability]
        if len(active) < 3:
            active = list(plan.accounts[:3])

        precursor_emitted = bool(rng.random() < config.precursor_probability)
        if config.precursor_probability >= 1.0:
            precursor_emitted = True

        preferred = [f"merchant-{int(x):04d}" for x in rng.choice(MERCHANT_COUNT, size=5, replace=False)]
        precursor_count = 0
        acceleration_count = 0
        merchant_precursor_count = 0
        cover_count = 0
        churned_ids = [a.account_id for a in plan.accounts if a.churn_at is not None]

        if precursor_emitted:
            # Cover history: normal-looking merchant activity before coordination.
            for account in active:
                n_cover = max(1, int(rng.poisson(config.ring_cover_events_per_member)))
                creation_day = (account.created_at - start).total_seconds() / 86400
                available = max(0.05, spec.coordination_start_day - creation_day)
                for _ in range(n_cover):
                    day = float(creation_day + rng.uniform(0.01, available))
                    ts = start + timedelta(days=day)
                    device, ip = account.identity_at(ts)
                    events.append({
                        "timestamp": ts, "account_id": account.account_id,
                        "device_id": device, "ip_prefix": ip,
                        "amount": float(_amounts(rng, 1)[0]),
                        "channel": str(rng.choice(CHANNEL_VALUES, p=CHANNEL_PROBS)),
                        "merchant_id": f"merchant-{int(rng.integers(0, MERCHANT_COUNT)):04d}",
                        "counterparty_account_id": None, "is_fraud": False, "ring_id": None,
                    })
                    cover_count += 1

            # Timing convergence: interactions concentrate around the latter part of coordination.
            pairs = list(plan.behavioral_pairs)
            rng.shuffle(pairs)
            take = min(len(pairs), max(2, int(config.precursor_interactions_per_ring * (0.55 + 0.6 * factor))))
            for a_id, b_id in pairs[:take]:
                center = spec.coordination_start_day + (spec.coordination_end_day - spec.coordination_start_day) * (0.45 + 0.45 * factor)
                spread = max(0.03, (spec.coordination_end_day - spec.coordination_start_day) * (0.22 - 0.15 * factor))
                day = float(np.clip(rng.normal(center, spread), spec.coordination_start_day, spec.coordination_end_day))
                a = account_lookup[a_id]
                b = account_lookup[b_id]
                ts = start + timedelta(days=day)
                device, ip = a.identity_at(ts)
                events.append({
                    "timestamp": ts, "account_id": a.account_id,
                    "device_id": device, "ip_prefix": ip,
                    "amount": float(_amounts(rng, 1)[0]),
                    "channel": str(rng.choice(CHANNEL_VALUES, p=CHANNEL_PROBS)),
                    "merchant_id": None, "counterparty_account_id": b.account_id,
                    "is_fraud": False, "ring_id": None,
                })
                precursor_count += 1

            # Merchant convergence: overlap rises probabilistically, but not everybody targets one merchant.
            convergence_probability = 0.45 + 0.35 * factor
            for _ in range(config.precursor_merchant_events_per_ring):
                account = active[int(rng.integers(0, len(active)))]
                day = float(rng.uniform(spec.coordination_start_day, spec.coordination_end_day))
                merchant = preferred[int(rng.integers(0, len(preferred)))] if rng.random() < convergence_probability else f"merchant-{int(rng.integers(0, MERCHANT_COUNT)):04d}"
                ts = start + timedelta(days=day)
                device, ip = account.identity_at(ts)
                events.append({
                    "timestamp": ts, "account_id": account.account_id,
                    "device_id": device, "ip_prefix": ip,
                    "amount": float(_amounts(rng, 1)[0]),
                    "channel": str(rng.choice(CHANNEL_VALUES, p=CHANNEL_PROBS)),
                    "merchant_id": merchant, "counterparty_account_id": None,
                    "is_fraud": False, "ring_id": None,
                })
                merchant_precursor_count += 1

            # Activity acceleration: event intensity increases toward coordination end.
            for account in active:
                n = max(1, int(rng.poisson(config.precursor_acceleration_events_per_member * (0.6 + 0.7 * factor))))
                for _ in range(n):
                    phase = float(rng.beta(3.0 + 3.0 * factor, 1.8))
                    day = spec.coordination_start_day + phase * (spec.coordination_end_day - spec.coordination_start_day)
                    ts = start + timedelta(days=float(day))
                    device, ip = account.identity_at(ts)
                    events.append({
                        "timestamp": ts, "account_id": account.account_id,
                        "device_id": device, "ip_prefix": ip,
                        "amount": float(_amounts(rng, 1)[0]),
                        "channel": str(rng.choice(CHANNEL_VALUES, p=CHANNEL_PROBS)),
                        "merchant_id": f"merchant-{int(rng.integers(0, MERCHANT_COUNT)):04d}",
                        "counterparty_account_id": None, "is_fraud": False, "ring_id": None,
                    })
                    acceleration_count += 1

        abuse_candidates = [a for a in active if a.account_id not in churned_ids]
        fraud_members = [a for a in abuse_candidates if rng.random() < config.fraud_participation_probability]
        if not fraud_members:
            fraud_members = [abuse_candidates[0]]

        fraud_count = 0
        for account in fraud_members:
            n = int(rng.integers(config.fraud_events_per_participant_min, config.fraud_events_per_participant_max + 1))
            for _ in range(n):
                day = float(max(spec.activation_day, rng.normal(spec.activation_day + 0.35, 0.45)))
                day = min(config.duration_days - 0.01, day)
                merchant = preferred[int(rng.integers(0, len(preferred)))] if rng.random() < 0.70 else f"merchant-{int(rng.integers(0, MERCHANT_COUNT)):04d}"
                ts = start + timedelta(days=day)
                device, ip = account.identity_at(ts)
                events.append({
                    "timestamp": ts, "account_id": account.account_id,
                    "device_id": device, "ip_prefix": ip,
                    "amount": float(_amounts(rng, 1)[0]),
                    "channel": str(rng.choice(CHANNEL_VALUES, p=CHANNEL_PROBS)),
                    "merchant_id": merchant, "counterparty_account_id": None,
                    "is_fraud": True, "ring_id": spec.ring_id,
                })
                fraud_count += 1

        ring_manifest.append({
            **spec.model_dump(mode="json"),
            "precursor_emitted": precursor_emitted,
            "preferred_merchant_count": len(preferred),
            "behavioral_edge_count": len(plan.behavioral_pairs),
            "precursor_interaction_count": precursor_count,
            "merchant_precursor_count": merchant_precursor_count,
            "cover_event_count": cover_count,
            "acceleration_event_count": acceleration_count,
            "churned_account_ids": churned_ids,
            "fraud_participant_count": len(fraud_members),
            "fraud_transaction_count": fraud_count,
        })

    return events, ring_manifest


def _raw_to_record(row: dict, world_id: str) -> EventRecord:
    if row["event_type"] == "account_created":
        return EventRecord(event=AccountCreatedEvent(**{k: v for k, v in row.items() if not k.startswith("_")}))
    if row["event_type"] == "account_updated":
        return EventRecord(event=AccountUpdatedEvent(**{k: v for k, v in row.items() if not k.startswith("_")}))
    event = TransactionEvent(
        event_id=row["event_id"], event_type="transaction", world_id=world_id,
        timestamp=row["timestamp"], account_id=row["account_id"],
        merchant_id=row["merchant_id"], counterparty_account_id=row["counterparty_account_id"],
        amount=row["amount"], channel=row["channel"], device_id=row["device_id"], ip_prefix=row["ip_prefix"],
    )
    gt = TransactionGroundTruth(
        event_id=row["event_id"], world_id=world_id,
        is_fraud=bool(row.get("_is_fraud", False)), ring_id=row.get("_ring_id"),
    )
    return EventRecord(event=event, ground_truth=gt)


def _manifest(config: WorldConfig, accounts: list[Account], clusters: list[HardNegativeCluster], rings: list[dict], event_count: int) -> dict:
    fraud_count = sum(int(r["fraud_transaction_count"]) for r in rings)
    return {
        "world_id": config.world_id.value,
        "seed": config.seed,
        "duration_days": config.duration_days,
        "event_count": event_count,
        "account_count": len(accounts),
        "legitimate_account_count": sum(a.cohort == "legitimate" for a in accounts),
        "hard_negative_account_count": sum(a.cohort.startswith("hard_negative:") for a in accounts),
        "ring_account_count": sum(a.cohort == "ring" for a in accounts),
        "organic_transaction_target": config.organic_transactions,
        "ring_count": len(rings),
        "fraud_transaction_count": fraud_count,
        "hard_negative_clusters": [
            {"cluster_id": c.cluster_id, "kind": c.kind, "account_ids": list(c.account_ids)} for c in clusters
        ],
        "rings": rings,
    }


def _component_data(
    rng: np.random.Generator,
    world_id,
    start: datetime,
    config: WorldConfig,
    legitimate_accounts: list[Account],
    hard_negative_accounts: list[Account],
    ring_accounts: list[Account],
    hard_negative_clusters: list[HardNegativeCluster],
    ring_plans: list[RingPlan],
):
    all_accounts = legitimate_accounts + hard_negative_accounts + ring_accounts
    lookup = {a.account_id: a for a in all_accounts}
    hard_rows = _hard_negative_events(rng, clusters=hard_negative_clusters, lookup=lookup, config=config, start=start)
    ring_rows, ring_manifest = _ring_events(rng, plans=ring_plans, account_lookup=lookup, config=config, start=start)
    return all_accounts, hard_rows, ring_rows, ring_manifest


def generate_world(*, rng: np.random.Generator, world_id, start: datetime, config: WorldConfig, legitimate_accounts: list[Account], hard_negative_accounts: list[Account], ring_accounts: list[Account], hard_negative_clusters: list[HardNegativeCluster], ring_plans: list[RingPlan]) -> GeneratedWorld:
    """Small/in-memory API for tests and development worlds."""
    all_accounts, hard_rows, ring_rows, ring_manifest = _component_data(
        rng, world_id, start, config, legitimate_accounts, hard_negative_accounts, ring_accounts, hard_negative_clusters, ring_plans
    )
    raw: list[dict] = []
    event_index = 0
    for account in all_accounts:
        raw.append({
            "event_id": _event_id(world_id.value, event_index), "event_type": "account_created", "world_id": world_id.value,
            "timestamp": account.created_at, "account_id": account.account_id,
            "device_id": account.device_id, "ip_prefix": account.ip_prefix,
        })
        event_index += 1
        if account.churn_at is not None:
            raw.append({
                "event_id": _event_id(world_id.value, event_index), "event_type": "account_updated", "world_id": world_id.value,
                "timestamp": account.churn_at, "account_id": account.account_id,
                "old_device_id": account.device_id, "old_ip_prefix": account.ip_prefix,
                "new_device_id": account.post_churn_device_id, "new_ip_prefix": account.post_churn_ip_prefix,
                "update_reason": "device_ip_churn",
            })
            event_index += 1

    account_ids = np.asarray([a.account_id for a in all_accounts], dtype=object)
    device_ids = np.asarray([a.device_id for a in all_accounts], dtype=object)
    ip_prefixes = np.asarray([a.ip_prefix for a in all_accounts], dtype=object)
    weights = np.asarray([a.activity_weight for a in all_accounts], dtype=float)
    created_days = np.asarray([(a.created_at - start).total_seconds() / 86400 for a in all_accounts])
    day_counts = _build_day_counts(
        min(config.organic_transactions, 10000), config.duration_days, rng,
        eligible_days=np.asarray([np.any(created_days <= d) for d in range(config.duration_days)], dtype=bool),
    )
    update_by_day = [[] for _ in range(config.duration_days)]
    for idx, account in enumerate(all_accounts):
        if account.churn_at is not None:
            d = int((account.churn_at - start).total_seconds() // 86400)
            if 0 <= d < config.duration_days:
                update_by_day[d].append((idx, account))

    for day, count in enumerate(day_counts):
        for idx, account in update_by_day[day]:
            device_ids[idx] = account.post_churn_device_id or account.device_id
            ip_prefixes[idx] = account.post_churn_ip_prefix or account.ip_prefix
        active = np.flatnonzero(created_days <= day)
        organic = _bulk_organic_day_dicts(
            rng, account_ids=account_ids, device_ids=device_ids, ip_prefixes=ip_prefixes,
            activity_weights=weights, active_indices=active, n_transactions=int(count),
            p2p_fraction=config.organic_p2p_fraction, start=start, day=day,
            event_start=event_index, world_id=world_id.value,
        )
        raw.extend(organic)
        event_index += len(organic)

    for row in hard_rows:
        row = dict(row)
        row.update({"event_id": _event_id(world_id.value, event_index), "event_type": "transaction", "world_id": world_id.value})
        event_index += 1
        raw.append(row)
    for row in ring_rows:
        row = dict(row)
        row.update({
            "event_id": _event_id(world_id.value, event_index),
            "event_type": "transaction",
            "world_id": world_id.value,
            "_is_fraud": bool(row.pop("is_fraud", False)),
            "_ring_id": row.pop("ring_id", None),
        })
        event_index += 1
        raw.append(row)

    raw.sort(key=lambda x: (x["timestamp"], x["event_id"]))
    records = [_raw_to_record(row, world_id.value) for row in raw]
    return GeneratedWorld(records=records, manifest=_manifest(config, all_accounts, hard_negative_clusters, ring_manifest, len(records)))


def write_world(*, rng: np.random.Generator, world_id, start: datetime, config: WorldConfig, legitimate_accounts: list[Account], hard_negative_accounts: list[Account], ring_accounts: list[Account], hard_negative_clusters: list[HardNegativeCluster], ring_plans: list[RingPlan], output_dir: Path) -> dict:
    """Memory-bounded production generator. Organic events are emitted one day at a time."""
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.jsonl"
    gt_path = output_dir / "ground_truth.jsonl"

    all_accounts = legitimate_accounts + hard_negative_accounts + ring_accounts
    lookup = {a.account_id: a for a in all_accounts}
    account_ids = np.asarray([a.account_id for a in all_accounts], dtype=object)
    base_devices = np.asarray([a.device_id for a in all_accounts], dtype=object)
    base_ips = np.asarray([a.ip_prefix for a in all_accounts], dtype=object)
    weights = np.asarray([a.activity_weight for a in all_accounts], dtype=float)
    created_days = np.asarray([(a.created_at - start).total_seconds() / 86400 for a in all_accounts], dtype=float)

    hard_rows = _hard_negative_events(rng, clusters=hard_negative_clusters, lookup=lookup, config=config, start=start)
    ring_rows, ring_manifest = _ring_events(rng, plans=ring_plans, account_lookup=lookup, config=config, start=start)

    hard_by_day = [[] for _ in range(config.duration_days)]
    for row in hard_rows:
        d = int((row["timestamp"] - start).total_seconds() // 86400)
        hard_by_day[max(0, min(config.duration_days - 1, d))].append(row)
    ring_by_day = [[] for _ in range(config.duration_days)]
    for row in ring_rows:
        d = int((row["timestamp"] - start).total_seconds() // 86400)
        ring_by_day[max(0, min(config.duration_days - 1, d))].append(row)

    creation_by_day = [[] for _ in range(config.duration_days)]
    update_by_day = [[] for _ in range(config.duration_days)]
    for account in all_accounts:
        cd = max(0, min(config.duration_days - 1, int((account.created_at - start).total_seconds() // 86400)))
        creation_by_day[cd].append(account)
        if account.churn_at is not None:
            ud = max(0, min(config.duration_days - 1, int((account.churn_at - start).total_seconds() // 86400)))
            update_by_day[ud].append(account)

    eligible = np.asarray([np.any(created_days <= d) for d in range(config.duration_days)], dtype=bool)
    day_counts = _build_day_counts(config.organic_transactions, config.duration_days, rng, eligible_days=eligible)
    current_devices = base_devices.copy()
    current_ips = base_ips.copy()
    index_by_account = {a.account_id: i for i, a in enumerate(all_accounts)}

    event_index = 0
    fraud_count = 0
    event_count = 0
    transaction_count = 0
    p2p_count = 0
    update_count = 0

    with events_path.open("wb", buffering=1024 * 1024) as events_file, gt_path.open("wb", buffering=1024 * 1024) as gt_file:
        for day in range(config.duration_days):
            rows: list[dict] = []
            for account in creation_by_day[day]:
                rows.append({
                    "event_id": _event_id(world_id.value, event_index), "event_type": "account_created", "world_id": world_id.value,
                    "timestamp": _iso(account.created_at), "account_id": account.account_id,
                    "device_id": account.device_id, "ip_prefix": account.ip_prefix, "_sort": account.created_at,
                })
                event_index += 1
                event_count += 1

            # Churn update is deliberately emitted before any same-day transaction events.
            for account in update_by_day[day]:
                idx = index_by_account[account.account_id]
                current_devices[idx] = account.post_churn_device_id or account.device_id
                current_ips[idx] = account.post_churn_ip_prefix or account.ip_prefix
                rows.append({
                    "event_id": _event_id(world_id.value, event_index), "event_type": "account_updated", "world_id": world_id.value,
                    "timestamp": _iso(account.churn_at), "account_id": account.account_id,
                    "old_device_id": account.device_id, "old_ip_prefix": account.ip_prefix,
                    "new_device_id": account.post_churn_device_id, "new_ip_prefix": account.post_churn_ip_prefix,
                    "update_reason": "device_ip_churn", "_sort": account.churn_at,
                })
                event_index += 1
                event_count += 1
                update_count += 1

            active = np.flatnonzero(created_days <= day)
            for row in hard_by_day[day]:
                rows.append({
                    "event_id": _event_id(world_id.value, event_index), "event_type": "transaction", "world_id": world_id.value,
                    "timestamp": _iso(row["timestamp"]), "account_id": row["account_id"],
                    "merchant_id": None, "counterparty_account_id": row["counterparty_account_id"],
                    "amount": row["amount"], "channel": row["channel"], "device_id": row["device_id"], "ip_prefix": row["ip_prefix"],
                    "_sort": row["timestamp"], "_is_fraud": False, "_ring_id": None,
                })
                event_index += 1
                event_count += 1
                transaction_count += 1
                p2p_count += 1

            organic = _bulk_organic_day_dicts(
                rng, account_ids=account_ids, device_ids=current_devices, ip_prefixes=current_ips,
                activity_weights=weights, active_indices=active, n_transactions=int(day_counts[day]),
                p2p_fraction=config.organic_p2p_fraction, start=start, day=day,
                event_start=event_index, world_id=world_id.value,
            )
            rows.extend(organic)
            event_index += len(organic)
            event_count += len(organic)
            transaction_count += len(organic)
            p2p_count += sum(1 for r in organic if r["counterparty_account_id"] is not None)

            for row in ring_by_day[day]:
                rows.append({
                    "event_id": _event_id(world_id.value, event_index), "event_type": "transaction", "world_id": world_id.value,
                    "timestamp": _iso(row["timestamp"]), "account_id": row["account_id"],
                    "merchant_id": row["merchant_id"], "counterparty_account_id": row["counterparty_account_id"],
                    "amount": row["amount"], "channel": row["channel"], "device_id": row["device_id"], "ip_prefix": row["ip_prefix"],
                    "_sort": row["timestamp"], "_is_fraud": row["is_fraud"], "_ring_id": row["ring_id"],
                })
                event_index += 1
                event_count += 1
                transaction_count += 1
                fraud_count += int(row["is_fraud"])
                p2p_count += int(row["counterparty_account_id"] is not None)

            rows.sort(key=lambda r: (r["_sort"], r["event_id"]))
            for row in rows:
                row.pop("_sort")
                if isinstance(row["timestamp"], datetime):
                    row["timestamp"] = _iso(row["timestamp"])
                is_fraud = bool(row.pop("_is_fraud", False))
                ring_id = row.pop("_ring_id", None)
                events_file.write(orjson.dumps(row) + b"\n")
                if row["event_type"] == "transaction":
                    gt_file.write(orjson.dumps({
                        "event_id": row["event_id"], "world_id": world_id.value,
                        "is_fraud": is_fraud, "ring_id": ring_id,
                    }) + b"\n")

    manifest = _manifest(config, all_accounts, hard_negative_clusters, ring_manifest, event_count)
    manifest.update({
        "transaction_count": transaction_count,
        "p2p_transaction_count": p2p_count,
        "fraud_transaction_count": fraud_count,
        "account_update_count": update_count,
        "output_format": "chronological_jsonl",
    })
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
