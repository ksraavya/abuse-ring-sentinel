from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class WorldId(str, Enum):
    WORLD_A = "world_a"
    WORLD_B = "world_b"


class EventType(str, Enum):
    ACCOUNT_CREATED = "account_created"
    TRANSACTION = "transaction"


class TransactionChannel(str, Enum):
    UPI = "upi"
    CARD = "card"
    WALLET = "wallet"
    NETBANKING = "netbanking"


class AccountCreatedEvent(BaseModel):
    """
    Observable account-creation event.

    Used to establish account and infrastructure state before
    subsequent transactions arrive.
    """

    model_config = ConfigDict(extra="forbid")

    event_id: str
    event_type: EventType = EventType.ACCOUNT_CREATED

    world_id: WorldId
    timestamp: datetime

    account_id: str
    device_id: str
    ip_prefix: str


class TransactionEvent(BaseModel):
    """
    Canonical observable transaction event transported through Kafka.

    Ground-truth fields are intentionally excluded.
    """

    model_config = ConfigDict(extra="forbid")

    event_id: str
    event_type: EventType = EventType.TRANSACTION

    world_id: WorldId
    timestamp: datetime

    account_id: str
    merchant_id: str | None = None
    counterparty_account_id: str | None = None

    amount: float = Field(gt=0)
    channel: TransactionChannel

    device_id: str
    ip_prefix: str


class TransactionGroundTruth(BaseModel):
    """
    Evaluation-only metadata.

    This object must never be published to detector consumers.
    """

    model_config = ConfigDict(extra="forbid")

    event_id: str
    world_id: WorldId

    is_fraud: bool
    ring_id: str | None = None


class EventRecord(BaseModel):
    """
    Complete event record used internally by the world generator
    and evaluation pipeline.

    The observable event and evaluation-only ground truth are
    structurally separated.
    """

    model_config = ConfigDict(extra="forbid")

    event: AccountCreatedEvent | TransactionEvent
    ground_truth: TransactionGroundTruth | None = None