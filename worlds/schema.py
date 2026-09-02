from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, model_validator

class WorldId(str, Enum):
    WORLD_A = "world_a"
    WORLD_B = "world_b"

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

    @model_validator(mode="after")
    def validate_topology_distribution(self) -> "WorldConfig":
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
        return self

    @property
    def total_rings(self) -> int:
        return self.fast_forming_rings + self.slow_burn_rings + self.obfuscated_rings
