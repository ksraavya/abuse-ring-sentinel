from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WorldId(str, Enum):
    WORLD_A = "world_a"
    WORLD_B = "world_b"
    WORLD_C = "world_c"
    WORLD_D = "world_d"
    

class RingKind(str, Enum):
    FAST = "fast"
    SLOW_BURN = "slow_burn"
    OBFUSCATED = "obfuscated"


class CoordinationStrength(str, Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


class WorldConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    world_id: WorldId
    seed: int = Field(ge=0)
    duration_days: int = Field(gt=0)
    legitimate_accounts: int = Field(gt=0)

    hard_negative_family_clusters: int = Field(ge=0)
    hard_negative_hostel_clusters: int = Field(ge=0)
    hard_negative_corporate_clusters: int = Field(ge=0)

    fast_forming_rings: int = Field(ge=0)
    slow_burn_rings: int = Field(ge=0)
    obfuscated_rings: int = Field(ge=0)

    topology_distribution: dict[str, float]

    organic_transactions: int = Field(default=2_900_000, gt=0)
    organic_p2p_fraction: float = Field(default=0.20, ge=0, le=0.8)
    activity_sigma: float = Field(default=1.0, gt=0)
    activity_min_weight: float = Field(default=0.10, gt=0)

    ring_size_min: int = Field(default=8, ge=4)
    ring_size_max: int = Field(default=20, ge=4)
    ring_cover_events_per_member: int = Field(default=4, ge=1)
    precursor_interactions_per_ring: int = Field(default=18, ge=2)
    precursor_merchant_events_per_ring: int = Field(default=18, ge=2)
    precursor_acceleration_events_per_member: int = Field(default=3, ge=1)
    fraud_events_per_participant_min: int = Field(default=8, ge=1)
    fraud_events_per_participant_max: int = Field(default=20, ge=1)

    hard_negative_p2p_rate_per_account_day: float = Field(default=0.18, ge=0)
    hard_negative_activity_multiplier: float = Field(default=1.35, gt=0)

    fraud_participation_probability: float = Field(default=0.65, ge=0, le=1)
    precursor_probability: float = Field(default=1.0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_config(self) -> "WorldConfig":
        allowed = {"distributed", "star", "chain", "cluster"}
        unknown = set(self.topology_distribution) - allowed
        if unknown:
            raise ValueError(f"unknown topology types: {sorted(unknown)}")
        if not self.topology_distribution:
            raise ValueError("topology_distribution cannot be empty")
        if any(v < 0 for v in self.topology_distribution.values()):
            raise ValueError("topology weights must be non-negative")
        if sum(self.topology_distribution.values()) <= 0:
            raise ValueError("topology weights must sum to a positive value")
        if self.ring_size_min > self.ring_size_max:
            raise ValueError("ring_size_min cannot exceed ring_size_max")
        if self.fraud_events_per_participant_min > self.fraud_events_per_participant_max:
            raise ValueError("fraud_events_per_participant_min cannot exceed fraud_events_per_participant_max")
        return self

    @property
    def total_rings(self) -> int:
        return self.fast_forming_rings + self.slow_burn_rings + self.obfuscated_rings


class RingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ring_id: str
    kind: RingKind
    topology: str
    account_ids: list[str]
    creation_start_day: float
    activation_day: float
    coordination_start_day: float
    coordination_end_day: float
    strength: CoordinationStrength
    churn_probability: float
    participant_probability: float
