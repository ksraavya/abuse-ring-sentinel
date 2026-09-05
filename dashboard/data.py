from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Paths:
    world_c_records: Path = ROOT / "artifacts/verifier/world_c/verification_records.jsonl"
    world_c_manifest: Path = ROOT / "data/generated/world_c/manifest.json"
    world_b_eval: Path = ROOT / "artifacts/evaluation/world_b"
    world_d_eval: Path = ROOT / "artifacts/evaluation/world_d"
    verifier_freeze: Path = ROOT / "artifacts/verifier/world_c/tuning/freeze_config.json"
    snapshot: Path = ROOT / "dashboard/snapshot.json"


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def iter_jsonl(path: Path):
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_jsonl_sample(path: Path, limit: int = 1000) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl(path) or ():
        if isinstance(row, dict):
            rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def load_manifest(path: Path) -> dict[str, Any]:
    return read_json(path) or {}


def load_freeze(path: Path) -> dict[str, Any]:
    return read_json(path) or {}


def find_metric(obj: Any, keys: tuple[str, ...]) -> Any:
    """Find the first matching key anywhere in a nested JSON object."""
    if isinstance(obj, dict):
        for key in keys:
            if key in obj:
                return obj[key]
        for value in obj.values():
            found = find_metric(value, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_metric(value, keys)
            if found is not None:
                return found
    return None


def discover_eval_json(directory: Path) -> Path | None:
    if not directory.is_dir():
        return None
    preferred = (
        "metrics.json", "world_d_evaluation.json", "evaluation_summary.json",
        "summary.json", "results.json", "world_d_summary.json",
    )
    for name in preferred:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    files = sorted(directory.rglob("*.json"))
    return files[0] if files else None


def load_evaluation(directory: Path) -> dict[str, Any]:
    path = discover_eval_json(directory)
    return read_json(path) if path else {}


def _snapshot() -> dict[str, Any]:
    return read_json(Paths().snapshot) or {}


def extract_metrics(obj: dict[str, Any]) -> dict[str, Any]:
    """Normalize both legacy result files and Commit-13 metrics.json."""
    aliases = {
        "precision": ("detector_precision", "precision"),
        "recall": ("detector_recall", "recall"),
        "fpr": ("detector_fpr", "fpr", "false_positive_rate"),
        "pr_auc": ("pr_auc", "pr_auc_score"),
        "roc_auc": ("roc_auc", "roc_auc_score"),
        "tp": ("detector_tp", "tp"), "fp": ("detector_fp", "fp"),
        "fn": ("detector_fn", "fn"), "tn": ("detector_tn", "tn"),
        "cost": ("economic_cost_detector", "economic_cost", "cost"),
        "detector_cost": ("economic_cost_detector",),
        "pre_abuse": ("pre_abuse_recall", "pre_abuse_ring_recall", "pre_abuse_rings_pct"),
        "pre_abuse_rings": ("pre_abuse_rings",),
        "ring_recall": ("eventual_ring_recall", "ring_recall", "ring_detection_recall"),
        "eventual_rings": ("eventual_rings",),
        "exposure": ("observed_fraud_exposure_blocked_pct", "exposure_prevented_pct", "exposure_prevented"),
        "fraud_exposure": ("fraud_exposure", "fraud_amount", "total_fraud_exposure"),
        "blocked_exposure": ("blocked_fraud_exposure", "fraud_amount_blocked"),
        "transactions": ("transactions", "transaction_count"),
        "fraud_transactions": ("fraud_transactions",),
        "alerts": ("alerts", "alert_count"),
        "blocks": ("blocks", "block_count"),
        "reviews": ("reviews", "review_count"),
        "allows": ("allows", "allow_count"),
        "block_precision": ("block_precision",),
        "block_recall": ("block_recall",),
        "block_fpr": ("block_fpr", "false_positive_block_rate"),
        "block_fnr": ("block_fnr",),
        "block_tp": ("block_tp",), "block_fp": ("block_fp",), "block_fn": ("block_fn",),
        "median_lead_days": ("median_lead_time_days", "median_detection_lead_days", "median_lead_days"),
        "median_block_lead_days": ("median_block_lead_time_days", "median_block_lead_days"),
        "alerted_exposure": ("alerted_fraud_exposure",),
        "blocked_exposure_pct": ("blocked_fraud_exposure_pct", "observed_fraud_exposure_blocked_pct"),
        "cost_reduction_pct": ("economic_cost_reduction_pct", "cost_reduction_pct"),
        "model_sha": ("temporal_model_sha256", "model_sha256"),
        "verifier_sha": ("verifier_config_sha256",),
        "tuning_after_freeze": ("world_d_tuning_after_freeze", "tuning_after_freeze"),
    }
    return {name: find_metric(obj, keys) for name, keys in aliases.items()}


def _merge_missing(primary: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    out = dict(primary)
    for k, v in fallback.items():
        if out.get(k) is None:
            out[k] = v
    return out


def load_world_b() -> dict[str, Any]:
    live = extract_metrics(load_evaluation(Paths().world_b_eval))
    return _merge_missing(live, _snapshot().get("world_b", {}))


def load_world_d() -> dict[str, Any]:
    live = extract_metrics(load_evaluation(Paths().world_d_eval))
    return _merge_missing(live, _snapshot().get("world_d", {}))


def load_snapshot() -> dict[str, Any]:
    return _snapshot()


def money(value: Any) -> str:
    try:
        x = float(value)
        if not math.isfinite(x):
            return "—"
        return f"₹{x:,.0f}"
    except (TypeError, ValueError):
        return "—"


def pct(value: Any, digits: int = 1) -> str:
    try:
        x = float(value)
        if not math.isfinite(x):
            return "—"
        if x <= 1.0:
            x *= 100
        return f"{x:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def num(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "—"


def demo_records(limit: int = 120) -> list[dict[str, Any]]:
    """Return a deliberately varied replay: reviews first, then strong blocks."""
    rows = load_jsonl_sample(Paths().world_c_records, 5000)
    if not rows:
        return []

    def score(r: dict[str, Any]) -> tuple:
        action = str(r.get("policy", {}).get("action", "review")).lower()
        # Prefer a visually useful mix rather than a block-only stream.
        bucket = {"review": 0, "block": 1, "allow": 2}.get(action, 3)
        evidence = r.get("evidence", []) or []
        strong = sum(str(x.get("strength", "")).lower() == "strong" for x in evidence if isinstance(x, dict))
        return (bucket, -strong, -len(evidence), -float(r.get("detector_probability", 0.0)))

    reviews = [r for r in rows if str(r.get("policy", {}).get("action", "review")).lower() == "review"]
    blocks = [r for r in rows if str(r.get("policy", {}).get("action", "review")).lower() == "block"]
    allows = [r for r in rows if str(r.get("policy", {}).get("action", "review")).lower() == "allow"]
    reviews.sort(key=score)
    blocks.sort(key=lambda r: (-len(r.get("evidence", []) or []), -float(r.get("verification_confidence", 0))))
    allows.sort(key=score)

    # Curate a 60/35/5 mix. If one class is absent, fill from others.
    target = min(limit, len(rows))
    n_review = min(len(reviews), max(1, int(target * 0.60)))
    n_block = min(len(blocks), max(1, int(target * 0.35)))
    n_allow = min(len(allows), target - n_review - n_block)
    chosen = reviews[:n_review] + blocks[:n_block] + allows[:n_allow]
    if len(chosen) < target:
        used = {id(x) for x in chosen}
        for r in rows:
            if id(r) not in used:
                chosen.append(r)
            if len(chosen) >= target:
                break

    # Interleave classes so the demo doesn't become a wall of reviews.
    buckets = [reviews[:n_review], blocks[:n_block], allows[:n_allow]]
    result: list[dict[str, Any]] = []
    while any(buckets):
        for b in buckets:
            if b:
                result.append(b.pop(0))
                if len(result) >= target:
                    return result
    return result


def rings_from_manifest(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(r["ring_id"]): r for r in manifest.get("rings", []) if isinstance(r, dict) and r.get("ring_id")}


def ring_for_account(account_id: str, manifest: dict[str, Any]) -> dict[str, Any] | None:
    account_id = str(account_id)
    for ring in rings_from_manifest(manifest).values():
        if account_id in {str(x) for x in ring.get("account_ids", [])}:
            return ring
    return None


def account_ring(account_id: str, manifest: dict[str, Any]) -> dict[str, Any] | None:
    """Backward-compatible alias used by the dashboard UI."""
    return ring_for_account(account_id, manifest)


def manifest_graph(ring: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ids = [str(x) for x in ring.get("account_ids", [])]
    nodes = [{"id": x, "label": x, "role": "peer"} for x in ids]
    edges: list[dict[str, Any]] = []
    if not ids:
        return nodes, edges
    nodes[0]["role"] = "alert"
    topology = str(ring.get("topology", "distributed"))
    if len(ids) < 2:
        return nodes, edges
    if topology == "star":
        edges = [{"source": ids[0], "target": x, "kind": "behavioral"} for x in ids[1:]]
    elif topology == "chain":
        edges = [{"source": a, "target": b, "kind": "behavioral"} for a, b in zip(ids, ids[1:])]
    elif topology == "cluster":
        groups = [ids[i::3] for i in range(3)]
        for group in groups:
            for i, a in enumerate(group):
                for b in group[i + 1 : i + 4]:
                    edges.append({"source": a, "target": b, "kind": "behavioral"})
        edges.extend({"source": a, "target": b, "kind": "behavioral"} for a, b in zip(ids, ids[1:]))
    else:
        for i in range(min(len(ids), 6)):
            a, b = ids[i], ids[(i + 1) % len(ids)]
            if a != b:
                edges.append({"source": a, "target": b, "kind": "behavioral"})
        for x in ids[2 : min(len(ids), 7)]:
            edges.append({"source": ids[0], "target": x, "kind": "behavioral"})
    return nodes, edges


def neo4j_neighbors(account_id: str, limit: int = 30) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not (os.getenv("NEO4J_URI") and os.getenv("NEO4J_PASSWORD")):
        return [], []
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.environ["NEO4J_PASSWORD"]))
        with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
            rows = list(session.run("""
                MATCH (a:Account {id:$id})-[r:TRANSACTED_WITH]-(b:Account)
                RETURN b.id AS id, r.count AS count, r.total_amount AS amount,
                       r.first_seen AS first_seen, r.last_seen AS last_seen
                ORDER BY r.count DESC LIMIT $limit
            """, id=account_id, limit=limit))
        driver.close()
        nodes = [{"id": account_id, "label": account_id, "role": "alert"}]
        edges = []
        for row in rows:
            bid = str(row["id"])
            nodes.append({"id": bid, "label": bid, "role": "peer"})
            edges.append({"source": account_id, "target": bid, "kind": "neo4j", "count": int(row["count"] or 0), "amount": float(row["amount"] or 0)})
        return nodes, edges
    except Exception:
        return [], []
