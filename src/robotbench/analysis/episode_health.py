"""Analyze LeRobot episodes and produce RobotBench health reports."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from lerobot.datasets import LeRobotDataset

from robotbench.domain import EpisodeHealthReport, HealthCheck, RobotRun
from robotbench.importers.lerobot_adapter import adapt_lerobot_episode
from robotbench.importers.lerobot_loader import (
    DEFAULT_DATASET_SUBDIR,
    DEFAULT_REPO_ID,
    DEFAULT_REVISION,
    load_lerobot_dataset,
    parse_episode_indices,
)


def _episode_columns(
    dataset: LeRobotDataset,
    episode_index: int,
) -> dict[str, np.ndarray]:
    """Return tabular columns belonging to one episode without decoding video."""
    table = dataset.hf_dataset
    available = set(table.column_names)
    required = {
        "episode_index",
        "frame_index",
        "timestamp",
        "action",
        "observation.state",
    }
    missing = required - available
    if missing:
        raise KeyError(f"Dataset is missing required columns: {sorted(missing)}")

    all_episode_indices = np.asarray(table["episode_index"], dtype=np.int64)
    mask = all_episode_indices == episode_index
    if not np.any(mask):
        raise IndexError(f"Episode {episode_index} does not contain any frames")

    return {
        "frame_index": np.asarray(table["frame_index"], dtype=np.int64)[mask],
        "timestamp": np.asarray(table["timestamp"], dtype=np.float64)[mask],
        "action": np.asarray(table["action"], dtype=np.float64)[mask],
        "state": np.asarray(table["observation.state"], dtype=np.float64)[mask],
    }


def _frame_count_check(actual_frames: int, expected_frames: int) -> HealthCheck:
    if actual_frames == expected_frames:
        return HealthCheck(
            code="frame_count",
            status="pass",
            message="Frame count matches episode metadata",
            metrics={"actual": actual_frames, "expected": expected_frames},
        )

    return HealthCheck(
        code="frame_count",
        status="fail",
        message="Frame count does not match episode metadata",
        metrics={"actual": actual_frames, "expected": expected_frames},
    )


def _signal_shape_check(
    actions: np.ndarray,
    states: np.ndarray,
    run: RobotRun,
) -> HealthCheck:
    expected_action_shape = tuple(run.signals.actions.shape)
    expected_state_shape = tuple(run.signals.states.shape)
    actual_action_shape = tuple(actions.shape[1:])
    actual_state_shape = tuple(states.shape[1:])

    valid = (
        actual_action_shape == expected_action_shape
        and actual_state_shape == expected_state_shape
        and len(actions) == len(states)
    )

    return HealthCheck(
        code="signal_shapes",
        status="pass" if valid else "fail",
        message=(
            "Actions and states match the declared schema"
            if valid
            else "Actions or states do not match the declared schema"
        ),
        metrics={
            "action_shape": str(actual_action_shape),
            "expected_action_shape": str(expected_action_shape),
            "state_shape": str(actual_state_shape),
            "expected_state_shape": str(expected_state_shape),
        },
    )


def _finite_values_check(
    actions: np.ndarray,
    states: np.ndarray,
    timestamps: np.ndarray,
) -> HealthCheck:
    action_non_finite = int(np.size(actions) - np.count_nonzero(np.isfinite(actions)))
    state_non_finite = int(np.size(states) - np.count_nonzero(np.isfinite(states)))
    timestamp_non_finite = int(
        np.size(timestamps) - np.count_nonzero(np.isfinite(timestamps))
    )
    total = action_non_finite + state_non_finite + timestamp_non_finite

    return HealthCheck(
        code="finite_values",
        status="pass" if total == 0 else "fail",
        message=(
            "No NaN or infinite values detected"
            if total == 0
            else "NaN or infinite values detected"
        ),
        metrics={
            "action_non_finite": action_non_finite,
            "state_non_finite": state_non_finite,
            "timestamp_non_finite": timestamp_non_finite,
        },
    )


def _frame_sequence_check(frame_indices: np.ndarray) -> HealthCheck:
    expected = np.arange(len(frame_indices), dtype=np.int64)
    mismatch_count = int(np.count_nonzero(frame_indices != expected))
    duplicate_count = int(len(frame_indices) - len(np.unique(frame_indices)))

    return HealthCheck(
        code="frame_sequence",
        status="pass" if mismatch_count == 0 else "fail",
        message=(
            "Frame indices are continuous and unique"
            if mismatch_count == 0
            else "Missing, duplicated, or reordered frame indices detected"
        ),
        metrics={
            "mismatch_count": mismatch_count,
            "duplicate_count": duplicate_count,
        },
    )


def _timing_check(timestamps: np.ndarray, expected_fps: float) -> HealthCheck:
    if len(timestamps) < 2 or not np.all(np.isfinite(timestamps)):
        return HealthCheck(
            code="timing",
            status="skipped",
            message="Timing check requires at least two finite timestamps",
        )

    differences = np.diff(timestamps)
    non_increasing = int(np.count_nonzero(differences <= 0))
    positive_differences = differences[differences > 0]

    if len(positive_differences) == 0:
        return HealthCheck(
            code="timing",
            status="fail",
            message="Timestamps do not increase",
            metrics={"non_increasing_count": non_increasing},
        )

    expected_interval = 1.0 / expected_fps
    median_interval = float(np.median(positive_differences))
    measured_fps = 1.0 / median_interval
    relative_fps_error = abs(measured_fps - expected_fps) / expected_fps
    large_gap_count = int(
        np.count_nonzero(positive_differences > expected_interval * 1.5)
    )
    p95_jitter_ms = float(
        np.percentile(np.abs(positive_differences - expected_interval), 95) * 1000
    )

    if non_increasing > 0:
        status = "fail"
        message = "Non-increasing timestamps detected"
    elif large_gap_count > 0 or relative_fps_error > 0.05:
        status = "warning"
        message = "Timing gaps or control-rate drift detected"
    else:
        status = "pass"
        message = "Timestamps and control rate look consistent"

    return HealthCheck(
        code="timing",
        status=status,
        message=message,
        metrics={
            "expected_fps": round(expected_fps, 4),
            "measured_fps": round(measured_fps, 4),
            "relative_fps_error": round(relative_fps_error, 6),
            "large_gap_count": large_gap_count,
            "non_increasing_count": non_increasing,
            "p95_jitter_ms": round(p95_jitter_ms, 4),
        },
    )


def _low_variation_check(
    actions: np.ndarray,
    states: np.ndarray,
    labels: Sequence[str] | None,
    epsilon: float = 1e-6,
) -> HealthCheck:
    if actions.ndim != 2 or states.ndim != 2 or actions.shape[1] != states.shape[1]:
        return HealthCheck(
            code="joint_variation",
            status="skipped",
            message="Joint variation requires matching two-dimensional signals",
        )

    action_ranges = np.ptp(actions, axis=0)
    state_ranges = np.ptp(states, axis=0)
    inactive_indices = np.flatnonzero(
        (action_ranges <= epsilon) & (state_ranges <= epsilon)
    )
    names = list(labels or [f"joint_{index}" for index in range(actions.shape[1])])
    inactive_names = [names[index] for index in inactive_indices]

    return HealthCheck(
        code="joint_variation",
        status="warning" if inactive_names else "pass",
        message=(
            "One or more joints show no measurable movement"
            if inactive_names
            else "All joints show measurable variation"
        ),
        metrics={
            "inactive_joint_count": len(inactive_names),
            "inactive_joints": ", ".join(inactive_names),
        },
    )


def _action_jump_check(actions: np.ndarray) -> HealthCheck:
    if actions.ndim != 2 or len(actions) < 3 or not np.all(np.isfinite(actions)):
        return HealthCheck(
            code="action_jumps",
            status="skipped",
            message="Action-jump check requires finite two-dimensional actions",
        )

    deltas = np.abs(np.diff(actions, axis=0))
    joint_ranges = np.ptp(actions, axis=0)
    active_joints = joint_ranges > 1e-6
    if not np.any(active_joints):
        return HealthCheck(
            code="action_jumps",
            status="skipped",
            message="Action discontinuity check requires at least one moving joint",
        )

    # This is deliberately a conservative, unitless data-integrity check.
    # Physical velocity checks require calibration limits for the actual robot.
    relative_deltas = deltas[:, active_joints] / joint_ranges[active_joints]
    discontinuity_threshold = 0.5
    discontinuities = relative_deltas > discontinuity_threshold
    discontinuity_count = int(np.count_nonzero(discontinuities))
    transition_count = int(discontinuities.size)
    discontinuity_ratio = (
        discontinuity_count / transition_count if transition_count else 0.0
    )
    max_relative_jump = float(np.max(relative_deltas))
    p99_relative_jump = float(np.percentile(relative_deltas, 99))

    status = "warning" if discontinuity_count > 0 else "pass"
    return HealthCheck(
        code="action_jumps",
        status=status,
        message=(
            "Extreme single-frame action discontinuities detected"
            if status == "warning"
            else "No extreme single-frame action discontinuities detected"
        ),
        metrics={
            "threshold_fraction_of_episode_range": discontinuity_threshold,
            "discontinuity_count": discontinuity_count,
            "discontinuity_ratio": round(discontinuity_ratio, 6),
            "p99_relative_jump": round(p99_relative_jump, 6),
            "max_relative_jump": round(max_relative_jump, 6),
        },
    )


def analyze_episode(
    dataset: LeRobotDataset,
    run: RobotRun,
    episode_index: int,
) -> EpisodeHealthReport:
    """Run the initial RobotBench data-health checks for one episode."""
    columns = _episode_columns(dataset, episode_index)
    frame_indices = columns["frame_index"]
    timestamps = columns["timestamp"]
    actions = columns["action"]
    states = columns["state"]

    checks = [
        _frame_count_check(len(frame_indices), run.timing.num_frames),
        _signal_shape_check(actions, states, run),
        _finite_values_check(actions, states, timestamps),
        _frame_sequence_check(frame_indices),
        _timing_check(timestamps, run.timing.fps),
        _low_variation_check(actions, states, run.signals.actions.labels),
        _action_jump_check(actions),
    ]

    return EpisodeHealthReport(
        run_id=run.run_id,
        episode_index=episode_index,
        checks=checks,
    )


def analyze_lerobot_episodes(
    *,
    repo_id: str,
    revision: str,
    episode_indices: Sequence[int],
    output_dir: Path,
    root: Path | None = None,
    dataset_subdir: str | None = None,
) -> list[Path]:
    """Analyze selected LeRobot episodes and write JSON health reports."""
    dataset = load_lerobot_dataset(
        repo_id=repo_id,
        root=root,
        episodes=episode_indices,
        revision=revision,
        dataset_subdir=dataset_subdir,
    )

    effective_subdir = dataset_subdir
    if effective_subdir is None and root is None and repo_id == DEFAULT_REPO_ID:
        effective_subdir = DEFAULT_DATASET_SUBDIR

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for episode_index in episode_indices:
        run = adapt_lerobot_episode(
            dataset,
            int(episode_index),
            repo_id=repo_id,
            revision=revision,
            dataset_subdir=effective_subdir,
        )
        report = analyze_episode(dataset, run, int(episode_index))
        output_path = output_dir / f"episode_{int(episode_index):06d}.health.json"
        output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        paths.append(output_path)
        print(
            f"Episode {episode_index}: {report.overall_status.upper()} "
            f"({report.passed_checks} passed, "
            f"{report.warning_checks} warnings, "
            f"{report.failed_checks} failed)"
        )

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--subdir")
    parser.add_argument(
        "--episodes",
        type=parse_episode_indices,
        required=True,
        help="Comma-separated episode indices, for example: 0,1,2,3,4",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/health"),
        help="Directory for generated EpisodeHealthReport files",
    )
    args = parser.parse_args()

    analyze_lerobot_episodes(
        repo_id=args.repo_id,
        revision=args.revision,
        episode_indices=args.episodes,
        output_dir=args.output,
        root=args.root,
        dataset_subdir=args.subdir,
    )


if __name__ == "__main__":
    main()
