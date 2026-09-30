"""Domain models for comparing RobotBench run summaries."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ComparisonVerdict = Literal[
    "improved",
    "regressed",
    "mixed",
    "unchanged",
]


class MetricComparison(BaseModel):
    """Baseline, candidate and their difference."""

    baseline: float
    candidate: float
    delta: float


class RunComparison(BaseModel):
    """Comparison between two aggregated health summaries."""

    schema_version: str = "0.1.0"

    baseline: str
    candidate: str
    verdict: ComparisonVerdict

    health_rate: MetricComparison

    new_issues: list[str] = Field(default_factory=list)
    resolved_issues: list[str] = Field(default_factory=list)
    regressed_checks: list[str] = Field(default_factory=list)
    improved_checks: list[str] = Field(default_factory=list)

    issue_score_deltas: dict[str, float] = Field(default_factory=dict)
