"""Domain models for annotating robot episode outcomes."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, model_validator


class EpisodeOutcome(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    ABORTED = "aborted"


class FailureMode(StrEnum):
    GRASP_FAILED = "grasp_failed"
    OBJECT_DROPPED = "object_dropped"
    MISSED_TARGET = "missed_target"
    COLLISION = "collision"
    TIMEOUT = "timeout"
    HARDWARE_FAULT = "hardware_fault"
    SAFETY_STOP = "safety_stop"
    UNKNOWN = "unknown"


class TaskPhase(StrEnum):
    INITIALIZATION = "initialization"
    APPROACH = "approach"
    GRASP = "grasp"
    TRANSPORT = "transport"
    PLACE = "place"
    RECOVERY = "recovery"
    UNKNOWN = "unknown"


class EpisodeAnnotation(BaseModel):
    """Human annotation describing the outcome of one episode."""

    schema_version: str = "0.1.0"

    run_id: str
    episode_index: int = Field(ge=0)

    outcome: EpisodeOutcome
    failure_mode: FailureMode | None = None
    phase: TaskPhase | None = None

    timestamp_seconds: float | None = Field(
        default=None,
        ge=0.0,
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
    )
    notes: str | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if self.outcome == EpisodeOutcome.SUCCESS and self.failure_mode is not None:
            raise ValueError("Successful episodes cannot have a failure_mode")

        if self.outcome != EpisodeOutcome.SUCCESS and self.failure_mode is None:
            raise ValueError("failure_mode is required for non-successful episodes")

        return self
