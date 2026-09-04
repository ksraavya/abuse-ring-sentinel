from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neo4j import GraphDatabase

from graph.neo4j_infrastructure import Neo4jConfig


@dataclass(frozen=True)
class BehavioralGraphConfig:
    """Neo4j configuration for behavioral-edge persistence."""

    neo4j: Neo4jConfig


class BehavioralGraph:
    """Neo4j persistence for directed account-to-account behavioral edges.

    The in-memory BehavioralState is authoritative for event-time feature
    extraction during offline replay. This class is the persistence layer.
    Writes are batched so Neo4j is not contacted once per transaction.
    """

    def __init__(self, config: BehavioralGraphConfig | None = None) -> None:
        if config is None:
            config = BehavioralGraphConfig(neo4j=Neo4jConfig())
        self.config = config
        neo4j_config = config.neo4j
        if not neo4j_config.uri or not neo4j_config.password:
            raise ValueError("Set NEO4J_URI and NEO4J_PASSWORD before connecting to Neo4j.")
        self.driver = GraphDatabase.driver(
            neo4j_config.uri,
            auth=(neo4j_config.username, neo4j_config.password),
        )

    def close(self) -> None:
        self.driver.close()

    def verify(self) -> None:
        self.driver.verify_connectivity()

    def initialize(self) -> None:
        """Create account constraints needed by behavioral relationships."""
        queries = [
            "CREATE CONSTRAINT account_id IF NOT EXISTS FOR (a:Account) REQUIRE a.id IS UNIQUE",
        ]
        with self.driver.session(database=self.config.neo4j.database) as session:
            for query in queries:
                session.run(query).consume()

    def reset(self) -> None:
        with self.driver.session(database=self.config.neo4j.database) as session:
            session.run("MATCH ()-[r:TRANSACTED_WITH]->() DELETE r").consume()
            
    def upsert_edge(
        self,
        sender: str,
        receiver: str,
        timestamp: str,
        amount: float,
    ) -> None:
        """Persist one directed behavioral edge update.

        Kept for runtime/smoke-test use. Offline replay should prefer the
        batched method below.
        """
        self.upsert_edges_batch(
            [
                {
                    "sender": sender,
                    "receiver": receiver,
                    "timestamp": timestamp,
                    "amount": float(amount),
                }
            ]
        )

    def upsert_edges_batch(self, edges: list[dict[str, Any]]) -> None:
        """Persist many directed edge updates in one Cypher query.

        Repeated ``(sender, receiver)`` pairs within a batch are handled by
        MERGE + incremental updates. The in-memory event-time state remains
        independent of when this persistence batch is flushed.
        """
        if not edges:
            return

        query = """
        UNWIND $rows AS row
        MERGE (a:Account {id: row.sender})
        MERGE (b:Account {id: row.receiver})
        MERGE (a)-[r:TRANSACTED_WITH]->(b)
        ON CREATE SET
            r.count = 1,
            r.total_amount = row.amount,
            r.first_seen = row.timestamp,
            r.last_seen = row.timestamp
        ON MATCH SET
            r.count = r.count + 1,
            r.total_amount = r.total_amount + row.amount,
            r.first_seen = CASE
                WHEN row.timestamp < r.first_seen THEN row.timestamp
                ELSE r.first_seen
            END,
            r.last_seen = CASE
                WHEN row.timestamp > r.last_seen THEN row.timestamp
                ELSE r.last_seen
            END
        """
        with self.driver.session(database=self.config.neo4j.database) as session:
            session.run(query, rows=edges).consume()

    def count_edges(self) -> int:
        query = "MATCH ()-[r:TRANSACTED_WITH]->() RETURN count(r) AS edges"
        with self.driver.session(database=self.config.neo4j.database) as session:
            record = session.run(query).single()
        return int(record["edges"])
