"""Internal RobotBench schema for a single robot run."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class SchemaModel(BaseModel):
    """Base class for all RobotBench schema models."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )


class RunSource(SchemaModel):
    """Location and format of the original run data."""

    type: Literal["lerobot"] = "lerobot"
    repo_id: str = Field(min_length=1)
    revision: str = Field(default="main", min_length=1)
    episode_index: int = Field(ge=0)
    dataset_subdir: str | None = None


class RobotInfo(SchemaModel):
    """Robot used to record the run."""

    type: str = Field(min_length=1)
    name: str | None = None


class TaskInfo(SchemaModel):
    """Task performed during the run."""

    name: str = Field(min_length=1)
    task_index: int | None = Field(default=None, ge=0)


class RunTiming(SchemaModel):
    """Timing information for the run."""

    fps: float = Field(gt=0)
    num_frames: int = Field(gt=0)

    @computed_field
    @property
    def duration_seconds(self) -> float:
        """Approximate duration calculated from frames and FPS."""
        return self.num_frames / self.fps


class SignalSpec(SchemaModel):
    """Schema of a numeric time-series signal."""

    source_key: str = Field(min_length=1)
    dtype: str = Field(min_length=1)
    shape: list[int] = Field(min_length=1)
    labels: list[str] | None = None
    units: list[str | None] | None = None

    @model_validator(mode="after")
    def validate_dimensions(self) -> "SignalSpec":
        if any(dimension <= 0 for dimension in self.shape):
            raise ValueError("Every signal dimension must be greater than zero")

        if self.labels is not None and len(self.shape) == 1:
            if len(self.labels) != self.shape[0]:
                raise ValueError("Number of labels must match the signal dimension")

        if self.units is not None and self.labels is not None:
            if len(self.units) != len(self.labels):
                raise ValueError("Number of units must match the number of labels")

        return self


class RunSignals(SchemaModel):
    """Actions and robot states recorded during the run."""

    actions: SignalSpec
    states: SignalSpec


class VideoStreamSpec(SchemaModel):
    """Description of one video stream."""

    source_key: str = Field(min_length=1)
    camera: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)
    codec: str | None = None


class RobotRun(SchemaModel):
    """RobotBench representation of one recorded robot episode."""

    schema_version: Literal["0.1.0"] = "0.1.0"
    run_id: str = Field(
        min_length=1,
        pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$",
    )
    source: RunSource
    robot: RobotInfo
    task: TaskInfo
    timing: RunTiming
    signals: RunSignals
    video_streams: list[VideoStreamSpec] = Field(default_factory=list)
