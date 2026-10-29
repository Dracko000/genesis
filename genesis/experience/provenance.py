"""Provenance metadata for experience records.

Provenance is deliberately simple: it records where a record came from and
whether it is an *observed* fact or an *inferred* explanation. The learning
pipeline must never treat an inference as a verified observation.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Provenance(BaseModel):
    """Where an experience record came from.

    ``inferred`` is the machine-checkable flag that separates an observation
    from a hypothesis. ``source`` is free-form so that synthetic environments,
    humans, and future sensors can all be represented without a schema change.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(default="unknown", min_length=1, max_length=200)
    created_by: str | None = Field(default=None, max_length=200)
    notes: str | None = Field(default=None, max_length=2000)
    inferred: bool = False
    generator: str | None = Field(
        default=None,
        max_length=200,
        description="Name of the generator/version, e.g. 'synthetic_world.v1'.",
    )
    generator_seed: int | None = Field(
        default=None, description="Seed used by the generator, when applicable."
    )

    @classmethod
    def synthetic(cls, generator: str, seed: int) -> "Provenance":
        """Provenance for deterministically generated observations."""

        return cls(
            source="synthetic",
            created_by="genesis.environments",
            inferred=False,
            generator=generator,
            generator_seed=seed,
        )
