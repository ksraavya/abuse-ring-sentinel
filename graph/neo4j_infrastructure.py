from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import os
from dataclasses import dataclass, field

from neo4j import GraphDatabase

@dataclass(frozen=True)
class Neo4jConfig:
    uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", ""))
    username: str = field(default_factory=lambda: os.getenv("NEO4J_USERNAME", "neo4j"))
    password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", ""))
    database: str = field(default_factory=lambda: os.getenv("NEO4J_DATABASE", "neo4j"))


class InfrastructureGraph:
    """Neo4j persistence for Baseline B's infrastructure graph.

    The graph contains only current account/device/IP relationships.
    Transactional behavior is intentionally not represented here.

    During offline training, InfrastructureState is the authoritative
    event-time feature state. Neo4j is a persisted graph mirror, so account
    lifecycle writes can safely be batched without changing feature values.
    """

    def __init__(self, config: Neo4jConfig | None = None) -> None:
        self.config = config or Neo4jConfig()
        if not self.config.uri or not self.config.password:
            raise ValueError("Set NEO4J_URI and NEO4J_PASSWORD in .env")
        self.driver = GraphDatabase.driver(
            self.config.uri,
            auth=(self.config.username, self.config.password),
        )

    def close(self) -> None:
        self.driver.close()

    def verify(self) -> None:
        self.driver.verify_connectivity()

    def initialize(self) -> None:
        """Create constraints used by the graph."""
        queries = [
            "CREATE CONSTRAINT account_id IF NOT EXISTS FOR (a:Account) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT device_id IF NOT EXISTS FOR (d:Device) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT ip_prefix_id IF NOT EXISTS FOR (i:IPPrefix) REQUIRE i.id IS UNIQUE",
        ]
        with self.driver.session(database=self.config.database) as session:
            for query in queries:
                session.run(query).consume()

    def reset(self) -> None:
        """Delete graph data while preserving the schema constraints."""
        with self.driver.session(database=self.config.database) as session:
            session.run("MATCH (n) DETACH DELETE n").consume()

    def upsert_account(
        self,
        account_id: str,
        device_id: str,
        ip_prefix: str,
    ) -> None:
        """Upsert one account.

        This single-row method remains available for runtime/smoke-test use.
        Offline training should use upsert_accounts_batch() instead.
        """
        query = """
        MERGE (a:Account {id: $account_id})
        WITH a
        OPTIONAL MATCH (a)-[rd:USES_DEVICE]->(:Device)
        DELETE rd
        WITH a
        OPTIONAL MATCH (a)-[ri:USES_IP]->(:IPPrefix)
        DELETE ri
        WITH a
        MERGE (d2:Device {id: $device_id})
        MERGE (ip2:IPPrefix {id: $ip_prefix})
        MERGE (a)-[:USES_DEVICE]->(d2)
        MERGE (a)-[:USES_IP]->(ip2)
        """
        with self.driver.session(database=self.config.database) as session:
            session.run(
                query,
                account_id=account_id,
                device_id=device_id,
                ip_prefix=ip_prefix,
            ).consume()

    def upsert_accounts_batch(self, accounts: list[dict[str, str]]) -> None:
        """Persist a batch of current account infrastructure relationships.

        Each row must contain account_id, device_id and ip_prefix. The caller
        should provide at most one row per account in a batch, representing
        that account's latest state at the time the batch is flushed.
        """
        if not accounts:
            return

        query = """
        UNWIND $rows AS row
        MERGE (a:Account {id: row.account_id})
        WITH a, row
        OPTIONAL MATCH (a)-[rd:USES_DEVICE]->(:Device)
        DELETE rd
        WITH a, row
        OPTIONAL MATCH (a)-[ri:USES_IP]->(:IPPrefix)
        DELETE ri
        WITH a, row
        MERGE (d:Device {id: row.device_id})
        MERGE (ip:IPPrefix {id: row.ip_prefix})
        MERGE (a)-[:USES_DEVICE]->(d)
        MERGE (a)-[:USES_IP]->(ip)
        """
        with self.driver.session(database=self.config.database) as session:
            session.run(query, rows=accounts).consume()

    def count(self) -> tuple[int, int, int]:
        query = """
        RETURN
          count { (a:Account) } AS accounts,
          count { (d:Device) } AS devices,
          count { (i:IPPrefix) } AS ips
        """
        with self.driver.session(database=self.config.database) as session:
            record = session.run(query).single()
        return (
            int(record["accounts"]),
            int(record["devices"]),
            int(record["ips"]),
        )
