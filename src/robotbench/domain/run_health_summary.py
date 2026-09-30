"""Domain model for an aggregated RobotBench health summary."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RunHealthSummary(BaseModel):
    """Aggregated health statistics for a collection of robot runs."""

    schema_version: str = "0.1.0"
    name: str

    total_episodes: int = Field(ge=0)
    healthy_episodes: int = Field(ge=0)
    warning_episodes: int = Field(ge=0)
    failed_episodes: int = Field(ge=0)

    health_rate: float = Field(ge=0.0, le=1.0)

    check_totals: dict[str, int]
    issues_by_code: dict[str, dict[str, int]]
    problem_episodes: list[int]
