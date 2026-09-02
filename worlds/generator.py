from __future__ import annotations
from datetime import datetime, timedelta, timezone
from random import Random

from events.schema import (
    AccountCreatedEvent, EventRecord, TransactionEvent,
    TransactionGroundTruth, TransactionChannel, WorldId as EventWorldId,
)
from .schema import WorldConfig

class WorldGenerator:
    """Deterministically generates a complete synthetic world.

    Commit 3 establishes the deterministic generation contract.
    Kafka streaming, Neo4j state, and full ring/behavior generation
    are intentionally separate implementation stages.
    """
    START_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __init__(self, config: WorldConfig):
        self.config = config
        self.rng = Random(config.seed)

    def generate(self) -> list[EventRecord]:
        events: list[EventRecord] = []
        world_id = EventWorldId(self.config.world_id.value)
        end = self.START_TIME + timedelta(days=self.config.duration_days)
        accounts: dict[str, AccountCreatedEvent] = {}

        for i in range(self.config.legitimate_accounts):
            ts = self.START_TIME + timedelta(
                minutes=self.rng.randrange(self.config.duration_days * 24 * 60)
            )
            account_id = self._id("acct", i)
            account = AccountCreatedEvent(
                event_id=self._id("event", len(events)),
                world_id=world_id,
                timestamp=ts,
                account_id=account_id,
                device_id=self._id("device", self.rng.randrange(max(1, self.config.legitimate_accounts // 3))),
                ip_prefix=f"10.{self.rng.randrange(1,223)}.{self.rng.randrange(256)}.0/24",
            )
            accounts[account_id] = account
            events.append(EventRecord(event=account))

        merchants = [self._id("merchant", i) for i in range(100)]
        channels = list(TransactionChannel)

        for account in accounts.values():
            for _ in range(self.rng.randint(2, 6)):
                ts = account.timestamp + timedelta(
                    hours=self.rng.randrange(1, self.config.duration_days * 24)
                )
                ts = min(ts, end - timedelta(seconds=1))
                tx = TransactionEvent(
                    event_id=self._id("event", len(events)),
                    world_id=world_id,
                    timestamp=ts,
                    account_id=account.account_id,
                    merchant_id=self.rng.choice(merchants),
                    amount=max(1.0, self.rng.lognormvariate(6.5, 1.2)),
                    channel=self.rng.choice(channels),
                    device_id=account.device_id,
                    ip_prefix=account.ip_prefix,
                )
                events.append(EventRecord(
                    event=tx,
                    ground_truth=TransactionGroundTruth(
                        event_id=tx.event_id,
                        world_id=world_id,
                        is_fraud=False,
                    ),
                ))

        events.sort(key=lambda r: (r.event.timestamp, r.event.event_id))
        return events

    @staticmethod
    def _id(prefix: str, value: int) -> str:
        return f"{prefix}_{value:08d}"
