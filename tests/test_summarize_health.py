import json
from pathlib import Path

import pytest

from robotbench.analysis.summarize_health import summarize_health_reports


def _write_report(
    directory: Path,
    *,
    episode_index: int,
    overall_status: str,
    checks: list[dict],
) -> None:
    report = {
        "schema_version": "0.1.0",
        "run_id": f"test-episode-{episode_index:06d}",
        "episode_index": episode_index,
        "overall_status": overall_status,
        "checks": checks,
    }

    path = directory / f"episode_{episode_index:06d}.health.json"
    path.write_text(json.dumps(report), encoding="utf-8")


def test_summarizes_health_reports(tmp_path: Path) -> None:
    _write_report(
        tmp_path,
        episode_index=0,
        overall_status="healthy",
        checks=[
            {"code": "finite_values", "status": "pass"},
            {"code": "timing", "status": "pass"},
        ],
    )

    _write_report(
        tmp_path,
        episode_index=1,
        overall_status="warning",
        checks=[
            {"code": "finite_values", "status": "pass"},
            {"code": "timing", "status": "warning"},
        ],
    )

    _write_report(
        tmp_path,
        episode_index=2,
        overall_status="failed",
        checks=[
            {"code": "finite_values", "status": "fail"},
            {"code": "timing", "status": "skipped"},
        ],
    )

    summary = summarize_health_reports(
        tmp_path,
        name="test-run",
    )

    assert summary.name == "test-run"
    assert summary.total_episodes == 3
    assert summary.healthy_episodes == 1
    assert summary.warning_episodes == 1
    assert summary.failed_episodes == 1
    assert summary.health_rate == pytest.approx(1 / 3, abs=1e-6)

    assert summary.check_totals == {
        "pass": 3,
        "warning": 1,
        "fail": 1,
        "skipped": 1,
    }

    assert summary.issues_by_code == {
        "finite_values": {
            "warning": 0,
            "fail": 1,
        },
        "timing": {
            "warning": 1,
            "fail": 0,
        },
    }

    assert summary.problem_episodes == [1, 2]


def test_fails_when_directory_has_no_reports(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        summarize_health_reports(tmp_path)
