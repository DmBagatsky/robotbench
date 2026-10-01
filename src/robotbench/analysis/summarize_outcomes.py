"""Aggregate RobotBench episode outcome annotations."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from robotbench.domain import (
    EpisodeAnnotation,
    EpisodeOutcome,
    OutcomeSummary,
)


def _episode_index_from_manifest(
    manifest: dict[str, Any],
) -> int:
    episode_index = manifest.get("episode_index")

    if episode_index is None:
        episode_index = manifest.get(
            "source",
            {},
        ).get("episode_index")

    if episode_index is None:
        raise ValueError("Run manifest does not contain an episode index")

    return int(episode_index)


def _load_expected_episode_indices(
    runs_dir: Path,
) -> set[int]:
    manifest_paths = sorted(runs_dir.glob("episode_*.json"))

    if not manifest_paths:
        raise FileNotFoundError(f"No episode manifests found in {runs_dir}")

    episode_indices: set[int] = set()

    for path in manifest_paths:
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in run manifest: {path}") from error

        episode_index = _episode_index_from_manifest(manifest)

        if episode_index in episode_indices:
            raise ValueError(
                f"Duplicate episode index {episode_index} " f"in {runs_dir}"
            )

        episode_indices.add(episode_index)

    return episode_indices


def _load_annotations(
    annotations_dir: Path,
) -> dict[int, EpisodeAnnotation]:
    annotation_paths = sorted(annotations_dir.glob("*.annotation.json"))

    annotations: dict[int, EpisodeAnnotation] = {}

    for path in annotation_paths:
        try:
            annotation = EpisodeAnnotation.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except ValueError as error:
            raise ValueError(f"Invalid annotation: {path}") from error

        if annotation.episode_index in annotations:
            raise ValueError(
                f"Duplicate annotation for episode " f"{annotation.episode_index}"
            )

        annotations[annotation.episode_index] = annotation

    return annotations


def summarize_outcomes(
    *,
    runs_dir: Path,
    annotations_dir: Path,
    name: str | None = None,
    low_confidence_threshold: float = 0.75,
) -> OutcomeSummary:
    """Aggregate annotations and calculate outcome metrics."""
    expected_episodes = _load_expected_episode_indices(runs_dir)
    annotations = _load_annotations(annotations_dir)

    unexpected_episodes = set(annotations) - expected_episodes

    if unexpected_episodes:
        raise ValueError(
            "Annotations reference unknown episodes: " f"{sorted(unexpected_episodes)}"
        )

    annotated_episodes = set(annotations)
    missing_annotations = sorted(expected_episodes - annotated_episodes)

    outcome_counts: Counter[str] = Counter()
    failure_mode_counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    low_confidence_episodes: list[int] = []

    for episode_index in sorted(annotations):
        annotation = annotations[episode_index]

        outcome_counts[annotation.outcome.value] += 1

        if annotation.failure_mode is not None:
            failure_mode_counts[annotation.failure_mode.value] += 1

        if annotation.phase is not None:
            phase_counts[annotation.phase.value] += 1

        if annotation.confidence < low_confidence_threshold:
            low_confidence_episodes.append(episode_index)

    total_count = len(expected_episodes)
    annotated_count = len(annotations)
    success_count = outcome_counts[EpisodeOutcome.SUCCESS.value]

    coverage_rate = annotated_count / total_count if total_count else 0.0

    success_rate = success_count / annotated_count if annotated_count else 0.0

    normalized_outcomes = {
        outcome.value: outcome_counts[outcome.value] for outcome in EpisodeOutcome
    }

    return OutcomeSummary(
        name=name or annotations_dir.name,
        total_episodes=total_count,
        annotated_episodes=annotated_count,
        missing_annotations=missing_annotations,
        coverage_rate=round(coverage_rate, 6),
        success_rate=round(success_rate, 6),
        outcome_counts=normalized_outcomes,
        failure_mode_counts=dict(sorted(failure_mode_counts.items())),
        phase_counts=dict(sorted(phase_counts.items())),
        low_confidence_episodes=(low_confidence_episodes),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--runs",
        type=Path,
        required=True,
        help="Directory containing episode run manifests",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Directory containing outcome annotations",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the outcome summary JSON",
    )
    parser.add_argument(
        "--name",
        help="Human-readable name for this collection",
    )
    parser.add_argument(
        "--low-confidence-threshold",
        type=float,
        default=0.75,
    )

    args = parser.parse_args()

    if not 0.0 <= args.low_confidence_threshold <= 1.0:
        parser.error("--low-confidence-threshold must be " "between 0 and 1")

    summary = summarize_outcomes(
        runs_dir=args.runs,
        annotations_dir=args.annotations,
        name=args.name,
        low_confidence_threshold=(args.low_confidence_threshold),
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        summary.model_dump_json(indent=2),
        encoding="utf-8",
    )

    print(f"Outcome summary: {summary.name}")
    print(
        f"Coverage: {summary.annotated_episodes}/"
        f"{summary.total_episodes} "
        f"({summary.coverage_rate:.1%})"
    )
    print(f"Success rate: {summary.success_rate:.1%}")

    if summary.missing_annotations:
        print(
            "Missing annotations: "
            + ", ".join(str(value) for value in summary.missing_annotations)
        )

    if summary.low_confidence_episodes:
        print(
            "Low-confidence episodes: "
            + ", ".join(str(value) for value in summary.low_confidence_episodes)
        )

    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
