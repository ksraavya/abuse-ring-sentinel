from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class WorldId(str, Enum):
    WORLD_A = "world_a"
    WORLD_B = "world_b"


class WorldConfig(BaseModel):
    """
    Configuration describing one independently generated synthetic world.
    """

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

    notes: str | None = None