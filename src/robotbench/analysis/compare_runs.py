"""Compare two aggregated RobotBench health summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

from robotbench.domain import (
    MetricComparison,
    RunComparison,
    RunHealthSummary,
)

EPSILON = 1e-9


def load_summary(path: Path) -> RunHealthSummary:
    """Load and validate a RobotBench health summary."""
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Health summary does not exist: {path}") from error

    try:
        return RunHealthSummary.model_validate_json(content)
    except ValueError as error:
        raise ValueError(f"Invalid health summary: {path}") from error


def _issue_score(
    summary: RunHealthSummary,
    code: str,
) -> float:
    """Return a normalized severity score for one check.

    A warning has weight 1 and a failure has weight 2.
    The value is normalized by the number of episodes.
    """
    issue = summary.issues_by_code.get(code, {})

    warning_count = int(issue.get("warning", 0))
    fail_count = int(issue.get("fail", 0))

    weighted_count = warning_count + fail_count * 2
    denominator = max(summary.total_episodes, 1)

    return weighted_count / denominator


def _determine_verdict(
    health_rate_delta: float,
    *,
    improved_checks: list[str],
    regressed_checks: list[str],
) -> str:
    health_improved = health_rate_delta > EPSILON
    health_regressed = health_rate_delta < -EPSILON

    checks_improved = bool(improved_checks)
    checks_regressed = bool(regressed_checks)

    if health_improved:
        return "mixed" if checks_regressed else "improved"

    if health_regressed:
        return "mixed" if checks_improved else "regressed"

    if checks_improved and checks_regressed:
        return "mixed"

    if checks_improved:
        return "improved"

    if checks_regressed:
        return "regressed"

    return "unchanged"


def compare_run_summaries(
    baseline: RunHealthSummary,
    candidate: RunHealthSummary,
) -> RunComparison:
    """Compare candidate health against a baseline."""
    all_issue_codes = sorted(
        set(baseline.issues_by_code) | set(candidate.issues_by_code)
    )

    new_issues: list[str] = []
    resolved_issues: list[str] = []
    regressed_checks: list[str] = []
    improved_checks: list[str] = []
    issue_score_deltas: dict[str, float] = {}

    for code in all_issue_codes:
        baseline_score = _issue_score(baseline, code)
        candidate_score = _issue_score(candidate, code)
        delta = candidate_score - baseline_score

        issue_score_deltas[code] = round(delta, 6)

        if baseline_score <= EPSILON and candidate_score > EPSILON:
            new_issues.append(code)

        if baseline_score > EPSILON and candidate_score <= EPSILON:
            resolved_issues.append(code)

        if delta > EPSILON:
            regressed_checks.append(code)
        elif delta < -EPSILON:
            improved_checks.append(code)

    health_rate_delta = candidate.health_rate - baseline.health_rate

    verdict = _determine_verdict(
        health_rate_delta,
        improved_checks=improved_checks,
        regressed_checks=regressed_checks,
    )

    return RunComparison(
        baseline=baseline.name,
        candidate=candidate.name,
        verdict=verdict,
        health_rate=MetricComparison(
            baseline=baseline.health_rate,
            candidate=candidate.health_rate,
            delta=round(health_rate_delta, 6),
        ),
        new_issues=new_issues,
        resolved_issues=resolved_issues,
        regressed_checks=regressed_checks,
        improved_checks=improved_checks,
        issue_score_deltas=issue_score_deltas,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Path to the baseline health summary",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="Path to the candidate health summary",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the comparison JSON",
    )
    args = parser.parse_args()

    baseline = load_summary(args.baseline)
    candidate = load_summary(args.candidate)

    comparison = compare_run_summaries(
        baseline,
        candidate,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        comparison.model_dump_json(indent=2),
        encoding="utf-8",
    )

    delta_percent = comparison.health_rate.delta * 100

    print(
        f"{comparison.baseline} -> "
        f"{comparison.candidate}: "
        f"{comparison.verdict.upper()}"
    )
    print(
        f"Health rate: "
        f"{comparison.health_rate.baseline:.1%} -> "
        f"{comparison.health_rate.candidate:.1%} "
        f"({delta_percent:+.1f}%)"
    )

    if comparison.new_issues:
        print(f"New issues: {', '.join(comparison.new_issues)}")

    if comparison.resolved_issues:
        print(f"Resolved issues: " f"{', '.join(comparison.resolved_issues)}")

    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
