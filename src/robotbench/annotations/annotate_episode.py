"""Create a human outcome annotation for a RobotBench episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from robotbench.domain import (
    EpisodeAnnotation,
    EpisodeOutcome,
    FailureMode,
    TaskPhase,
)


def load_manifest(path: Path) -> dict[str, Any]:
    """Load an imported RobotBench run manifest."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Run manifest does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Run manifest contains invalid JSON: {path}") from error


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


def create_annotation(
    manifest: dict[str, Any],
    *,
    outcome: str,
    failure_mode: str | None = None,
    phase: str | None = None,
    timestamp_seconds: float | None = None,
    confidence: float = 1.0,
    notes: str | None = None,
) -> EpisodeAnnotation:
    """Create and validate an annotation from a run manifest."""
    run_id = manifest.get("run_id")

    if not run_id:
        raise ValueError("Run manifest does not contain run_id")

    episode_index = _episode_index_from_manifest(manifest)

    duration_seconds = manifest.get(
        "timing",
        {},
    ).get("duration_seconds")

    if (
        timestamp_seconds is not None
        and duration_seconds is not None
        and timestamp_seconds > float(duration_seconds)
    ):
        raise ValueError(
            f"Annotation timestamp {timestamp_seconds} exceeds "
            f"episode duration {duration_seconds}"
        )

    return EpisodeAnnotation(
        run_id=str(run_id),
        episode_index=episode_index,
        outcome=outcome,
        failure_mode=failure_mode,
        phase=phase,
        timestamp_seconds=timestamp_seconds,
        confidence=confidence,
        notes=notes,
    )


def save_annotation(
    annotation: EpisodeAnnotation,
    output_path: Path,
) -> None:
    """Write an annotation to JSON."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        annotation.model_dump_json(indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to an episode run manifest",
    )
    parser.add_argument(
        "--outcome",
        required=True,
        choices=[value.value for value in EpisodeOutcome],
    )
    parser.add_argument(
        "--failure-mode",
        choices=[value.value for value in FailureMode],
    )
    parser.add_argument(
        "--phase",
        choices=[value.value for value in TaskPhase],
    )
    parser.add_argument(
        "--timestamp",
        type=float,
        dest="timestamp_seconds",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=1.0,
    )
    parser.add_argument("--notes")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path; generated automatically when omitted",
    )

    args = parser.parse_args()

    if args.outcome == EpisodeOutcome.SUCCESS and args.failure_mode is not None:
        parser.error("--failure-mode cannot be used with a successful outcome")

    if args.outcome != EpisodeOutcome.SUCCESS and args.failure_mode is None:
        parser.error("--failure-mode is required for non-successful outcomes")

    manifest = load_manifest(args.manifest)

    annotation = create_annotation(
        manifest,
        outcome=args.outcome,
        failure_mode=args.failure_mode,
        phase=args.phase,
        timestamp_seconds=args.timestamp_seconds,
        confidence=args.confidence,
        notes=args.notes,
    )

    output_path = args.output

    if output_path is None:
        output_path = Path("artifacts/annotations") / (
            f"episode_" f"{annotation.episode_index:06d}" f".annotation.json"
        )

    save_annotation(
        annotation,
        output_path,
    )

    print(f"Episode {annotation.episode_index}: " f"{annotation.outcome.value.upper()}")

    if annotation.failure_mode is not None:
        print(f"Failure mode: " f"{annotation.failure_mode.value}")

    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
