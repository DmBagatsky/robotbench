"""Health-report schema for one robot episode."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, computed_field

from robotbench.domain.robot_run import SchemaModel

CheckStatus = Literal["pass", "warning", "fail", "skipped"]
OverallStatus = Literal["healthy", "warning", "unhealthy"]


class HealthCheck(SchemaModel):
    """Result of one diagnostic check."""

    code: str = Field(
        min_length=1,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    status: CheckStatus
    message: str = Field(min_length=1)
    metrics: dict[str, int | float | str | bool | None] = Field(default_factory=dict)


class EpisodeHealthReport(SchemaModel):
    """Diagnostic report for one RobotRun."""

    schema_version: Literal["0.1.0"] = "0.1.0"
    run_id: str = Field(min_length=1)
    episode_index: int = Field(ge=0)
    checks: list[HealthCheck] = Field(default_factory=list)

    @computed_field
    @property
    def overall_status(self) -> OverallStatus:
        statuses = {check.status for check in self.checks}

        if "fail" in statuses:
            return "unhealthy"

        if "warning" in statuses:
            return "warning"

        return "healthy"

    @computed_field
    @property
    def passed_checks(self) -> int:
        return sum(check.status == "pass" for check in self.checks)

    @computed_field
    @property
    def warning_checks(self) -> int:
        return sum(check.status == "warning" for check in self.checks)

    @computed_field
    @property
    def failed_checks(self) -> int:
        return sum(check.status == "fail" for check in self.checks)
