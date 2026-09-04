from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class WorldId(str, Enum):
    WORLD_A = "world_a"
    WORLD_B = "world_b"
    WORLD_C = "world_c"
    WORLD_D = "world_d"


class EventType(str, Enum):
    ACCOUNT_CREATED = "account_created"
    ACCOUNT_UPDATED = "account_updated"
    TRANSACTION = "transaction"


class TransactionChannel(str, Enum):
    UPI = "upi"
    CARD = "card"
    WALLET = "wallet"
    NETBANKING = "netbanking"


class AccountCreatedEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    event_type: EventType = EventType.ACCOUNT_CREATED
    world_id: WorldId
    timestamp: datetime
    account_id: str
    device_id: str
    ip_prefix: str


class AccountUpdatedEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    event_type: EventType = EventType.ACCOUNT_UPDATED
    world_id: WorldId
    timestamp: datetime
    account_id: str
    old_device_id: str
    old_ip_prefix: str
    new_device_id: str
    new_ip_prefix: str
    update_reason: str


class TransactionEvent(BaseModel):
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
    model_config = ConfigDict(extra="forbid")

    event_id: str
    world_id: WorldId
    is_fraud: bool
    ring_id: str | None = None


class EventRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event: AccountCreatedEvent | AccountUpdatedEvent | TransactionEvent
    ground_truth: TransactionGroundTruth | None = None
