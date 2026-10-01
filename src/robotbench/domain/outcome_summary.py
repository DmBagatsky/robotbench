"""Aggregated summary of episode outcome annotations."""

from __future__ import annotations

from pydantic import BaseModel, Field


class OutcomeSummary(BaseModel):
    """Aggregated outcome statistics for a collection of episodes."""

    schema_version: str = "0.1.0"
    name: str

    total_episodes: int = Field(ge=0)
    annotated_episodes: int = Field(ge=0)
    missing_annotations: list[int] = Field(default_factory=list)

    coverage_rate: float = Field(ge=0.0, le=1.0)
    success_rate: float = Field(ge=0.0, le=1.0)

    outcome_counts: dict[str, int]
    failure_mode_counts: dict[str, int]
    phase_counts: dict[str, int]

    low_confidence_episodes: list[int] = Field(default_factory=list)
