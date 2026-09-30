"""Aggregate RobotBench episode health reports."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from robotbench.domain import RunHealthSummary

VALID_OVERALL_STATUSES = {"healthy", "warning", "failed"}
VALID_CHECK_STATUSES = {"pass", "warning", "fail", "skipped"}


def _load_health_report(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}") from error

    required_fields = {
        "run_id",
        "episode_index",
        "overall_status",
        "checks",
    }
    missing_fields = required_fields - report.keys()

    if missing_fields:
        raise ValueError(f"{path} is missing required fields: {sorted(missing_fields)}")

    return report


def summarize_health_reports(
    input_dir: Path,
    *,
    name: str | None = None,
) -> RunHealthSummary:
    """Aggregate all *.health.json reports from a directory."""
    report_paths = sorted(input_dir.glob("*.health.json"))

    if not report_paths:
        raise FileNotFoundError(f"No *.health.json reports found in {input_dir}")

    overall_counts: Counter[str] = Counter()
    check_totals: Counter[str] = Counter()
    issues: dict[str, Counter[str]] = defaultdict(Counter)
    problem_episodes: list[int] = []

    for report_path in report_paths:
        report = _load_health_report(report_path)

        episode_index = int(report["episode_index"])
        overall_status = str(report["overall_status"]).lower()

        if overall_status not in VALID_OVERALL_STATUSES:
            raise ValueError(
                f"Unknown overall status {overall_status!r} " f"in {report_path}"
            )

        overall_counts[overall_status] += 1

        if overall_status != "healthy":
            problem_episodes.append(episode_index)

        for check in report["checks"]:
            code = str(check["code"])
            status = str(check["status"]).lower()

            if status not in VALID_CHECK_STATUSES:
                raise ValueError(
                    f"Unknown check status {status!r} " f"for {code!r} in {report_path}"
                )

            check_totals[status] += 1

            if status in {"warning", "fail"}:
                issues[code][status] += 1

    total_episodes = len(report_paths)
    healthy_episodes = overall_counts["healthy"]

    normalized_check_totals = {
        status: check_totals[status]
        for status in ("pass", "warning", "fail", "skipped")
    }

    normalized_issues = {
        code: {
            "warning": counts["warning"],
            "fail": counts["fail"],
        }
        for code, counts in sorted(issues.items())
    }

    return RunHealthSummary(
        name=name or input_dir.name,
        total_episodes=total_episodes,
        healthy_episodes=healthy_episodes,
        warning_episodes=overall_counts["warning"],
        failed_episodes=overall_counts["failed"],
        health_rate=round(healthy_episodes / total_episodes, 6),
        check_totals=normalized_check_totals,
        issues_by_code=normalized_issues,
        problem_episodes=sorted(problem_episodes),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing *.health.json reports",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the aggregated JSON summary",
    )
    parser.add_argument(
        "--name",
        help="Human-readable name of this run collection",
    )
    args = parser.parse_args()

    summary = summarize_health_reports(
        args.input,
        name=args.name,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        summary.model_dump_json(indent=2),
        encoding="utf-8",
    )

    print(f"Summary: {summary.name}")
    print(f"Episodes: {summary.total_episodes}")
    print(
        f"Healthy: {summary.healthy_episodes}, "
        f"warnings: {summary.warning_episodes}, "
        f"failed: {summary.failed_episodes}"
    )
    print(f"Health rate: {summary.health_rate:.1%}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
