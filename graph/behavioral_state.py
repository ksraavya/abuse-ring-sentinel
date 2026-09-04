from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BehavioralEdge:
    """Metadata for one directed account-to-account behavioral edge."""

    count: int = 0
    total_amount: float = 0.0
    first_seen: str | None = None
    last_seen: str | None = None

    def update(self, timestamp: str, amount: float) -> None:
        """Apply one observed transaction to this edge."""
        if not timestamp:
            raise ValueError("timestamp must be non-empty")
        if amount <= 0:
            raise ValueError("amount must be greater than zero")

        self.count += 1
        self.total_amount += float(amount)

        if self.first_seen is None or timestamp < self.first_seen:
            self.first_seen = timestamp
        if self.last_seen is None or timestamp > self.last_seen:
            self.last_seen = timestamp

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "total_amount": self.total_amount,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }


@dataclass
class BehavioralState:
    """In-memory event-time state for the directed behavioral graph.

    The state stores only observable account-to-account transaction edges.
    Feature engineering belongs in the feature layer; this object intentionally
    exposes only graph primitives and the state mutation operation.

    Edges are directed as ``(sender, receiver)``. ``get_neighbors`` returns
    the union of incoming and outgoing neighbors so feature code can choose
    whether to reason about direction explicitly or aggregate to an
    undirected neighborhood.
    """

    edges: dict[tuple[str, str], BehavioralEdge] = field(default_factory=dict)
    out_neighbors: dict[str, set[str]] = field(default_factory=dict)
    in_neighbors: dict[str, set[str]] = field(default_factory=dict)

    def get_edge(self, sender: str, receiver: str) -> BehavioralEdge | None:
        """Return directed edge metadata, or ``None`` if the edge is unseen."""
        return self.edges.get((sender, receiver))

    def get_neighbors(self, account: str) -> set[str]:
        """Return all accounts connected to ``account`` in either direction."""
        return set(self.out_neighbors.get(account, set())) | set(
            self.in_neighbors.get(account, set())
        )

    def update(self, sender: str, receiver: str, timestamp: str, amount: float) -> None:
        """Record one completed account-to-account transaction.

        This method is deliberately separate from feature reads. The replay
        layer must call it only after the current transaction has been scored,
        preserving the event-time causal boundary.
        """
        if not sender or not receiver:
            raise ValueError("sender and receiver must be non-empty")
        if sender == receiver:
            raise ValueError("sender and receiver must be different accounts")

        key = (sender, receiver)
        edge = self.edges.get(key)
        if edge is None:
            edge = BehavioralEdge()
            self.edges[key] = edge

        edge.update(timestamp=timestamp, amount=amount)
        self.out_neighbors.setdefault(sender, set()).add(receiver)
        self.in_neighbors.setdefault(receiver, set()).add(sender)
