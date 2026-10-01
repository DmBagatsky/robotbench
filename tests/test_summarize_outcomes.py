import json
from pathlib import Path

import pytest

from robotbench.analysis.summarize_outcomes import (
    summarize_outcomes,
)


def _write_manifest(
    directory: Path,
    episode_index: int,
) -> None:
    manifest = {
        "run_id": f"test-episode-{episode_index:06d}",
        "source": {
            "episode_index": episode_index,
        },
    }

    path = directory / f"episode_{episode_index:06d}.json"
    path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def _write_annotation(
    directory: Path,
    *,
    episode_index: int,
    outcome: str,
    failure_mode: str | None = None,
    phase: str | None = None,
    confidence: float = 1.0,
) -> None:
    annotation = {
        "schema_version": "0.1.0",
        "run_id": f"test-episode-{episode_index:06d}",
        "episode_index": episode_index,
        "outcome": outcome,
        "failure_mode": failure_mode,
        "phase": phase,
        "timestamp_seconds": None,
        "confidence": confidence,
        "notes": None,
    }

    path = directory / f"episode_{episode_index:06d}.annotation.json"
    path.write_text(
        json.dumps(annotation),
        encoding="utf-8",
    )


def test_summarizes_outcomes(
    tmp_path: Path,
) -> None:
    runs_dir = tmp_path / "runs"
    annotations_dir = tmp_path / "annotations"
    runs_dir.mkdir()
    annotations_dir.mkdir()

    for episode_index in range(3):
        _write_manifest(
            runs_dir,
            episode_index,
        )

    _write_annotation(
        annotations_dir,
        episode_index=0,
        outcome="success",
    )
    _write_annotation(
        annotations_dir,
        episode_index=1,
        outcome="failure",
        failure_mode="object_dropped",
        phase="transport",
        confidence=0.9,
    )
    _write_annotation(
        annotations_dir,
        episode_index=2,
        outcome="partial",
        failure_mode="missed_target",
        phase="place",
        confidence=0.5,
    )

    summary = summarize_outcomes(
        runs_dir=runs_dir,
        annotations_dir=annotations_dir,
        name="test-evaluation",
    )

    assert summary.name == "test-evaluation"
    assert summary.total_episodes == 3
    assert summary.annotated_episodes == 3
    assert summary.coverage_rate == 1.0
    assert summary.success_rate == pytest.approx(
        1 / 3,
        abs=1e-6,
    )

    assert summary.outcome_counts == {
        "success": 1,
        "failure": 1,
        "partial": 1,
        "aborted": 0,
    }
    assert summary.failure_mode_counts == {
        "missed_target": 1,
        "object_dropped": 1,
    }
    assert summary.phase_counts == {
        "place": 1,
        "transport": 1,
    }
    assert summary.low_confidence_episodes == [2]


def test_reports_missing_annotations(
    tmp_path: Path,
) -> None:
    runs_dir = tmp_path / "runs"
    annotations_dir = tmp_path / "annotations"
    runs_dir.mkdir()
    annotations_dir.mkdir()

    _write_manifest(runs_dir, 0)
    _write_manifest(runs_dir, 1)

    _write_annotation(
        annotations_dir,
        episode_index=0,
        outcome="success",
    )

    summary = summarize_outcomes(
        runs_dir=runs_dir,
        annotations_dir=annotations_dir,
    )

    assert summary.total_episodes == 2
    assert summary.annotated_episodes == 1
    assert summary.coverage_rate == 0.5
    assert summary.missing_annotations == [1]


def test_rejects_unknown_episode_annotation(
    tmp_path: Path,
) -> None:
    runs_dir = tmp_path / "runs"
    annotations_dir = tmp_path / "annotations"
    runs_dir.mkdir()
    annotations_dir.mkdir()

    _write_manifest(runs_dir, 0)

    _write_annotation(
        annotations_dir,
        episode_index=99,
        outcome="success",
    )

    with pytest.raises(
        ValueError,
        match="unknown episodes",
    ):
        summarize_outcomes(
            runs_dir=runs_dir,
            annotations_dir=annotations_dir,
        )
