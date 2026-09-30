from robotbench.analysis.compare_runs import compare_run_summaries
from robotbench.domain import RunHealthSummary


def _summary(
    *,
    name: str,
    total: int,
    healthy: int,
    warnings: int,
    failed: int,
    issues: dict[str, dict[str, int]],
) -> RunHealthSummary:
    return RunHealthSummary(
        name=name,
        total_episodes=total,
        healthy_episodes=healthy,
        warning_episodes=warnings,
        failed_episodes=failed,
        health_rate=healthy / total,
        check_totals={
            "pass": 0,
            "warning": 0,
            "fail": 0,
            "skipped": 0,
        },
        issues_by_code=issues,
        problem_episodes=[],
    )


def test_detects_improvement() -> None:
    baseline = _summary(
        name="act-v1",
        total=10,
        healthy=7,
        warnings=2,
        failed=1,
        issues={
            "timing": {
                "warning": 2,
                "fail": 0,
            },
            "action_jumps": {
                "warning": 1,
                "fail": 0,
            },
        },
    )

    candidate = _summary(
        name="act-v2",
        total=10,
        healthy=9,
        warnings=1,
        failed=0,
        issues={
            "action_jumps": {
                "warning": 1,
                "fail": 0,
            },
        },
    )

    comparison = compare_run_summaries(
        baseline,
        candidate,
    )

    assert comparison.verdict == "improved"
    assert comparison.health_rate.delta == 0.2
    assert comparison.resolved_issues == ["timing"]
    assert comparison.improved_checks == ["timing"]
    assert comparison.new_issues == []
    assert comparison.regressed_checks == []


def test_detects_regression() -> None:
    baseline = _summary(
        name="act-v1",
        total=10,
        healthy=10,
        warnings=0,
        failed=0,
        issues={},
    )

    candidate = _summary(
        name="act-v2",
        total=10,
        healthy=8,
        warnings=1,
        failed=1,
        issues={
            "finite_values": {
                "warning": 0,
                "fail": 1,
            },
        },
    )

    comparison = compare_run_summaries(
        baseline,
        candidate,
    )

    assert comparison.verdict == "regressed"
    assert comparison.health_rate.delta == -0.2
    assert comparison.new_issues == ["finite_values"]
    assert comparison.regressed_checks == ["finite_values"]
    assert comparison.resolved_issues == []
    assert comparison.improved_checks == []


def test_detects_mixed_result() -> None:
    baseline = _summary(
        name="act-v1",
        total=10,
        healthy=7,
        warnings=3,
        failed=0,
        issues={
            "timing": {
                "warning": 3,
                "fail": 0,
            },
        },
    )

    candidate = _summary(
        name="act-v2",
        total=10,
        healthy=8,
        warnings=2,
        failed=0,
        issues={
            "action_jumps": {
                "warning": 2,
                "fail": 0,
            },
        },
    )

    comparison = compare_run_summaries(
        baseline,
        candidate,
    )

    assert comparison.verdict == "mixed"
    assert comparison.resolved_issues == ["timing"]
    assert comparison.new_issues == ["action_jumps"]
    assert comparison.improved_checks == ["timing"]
    assert comparison.regressed_checks == ["action_jumps"]


def test_identical_runs_are_unchanged() -> None:
    baseline = _summary(
        name="baseline",
        total=5,
        healthy=5,
        warnings=0,
        failed=0,
        issues={},
    )

    candidate = baseline.model_copy(update={"name": "candidate"})

    comparison = compare_run_summaries(
        baseline,
        candidate,
    )

    assert comparison.verdict == "unchanged"
    assert comparison.health_rate.delta == 0.0
    assert comparison.new_issues == []
    assert comparison.resolved_issues == []
