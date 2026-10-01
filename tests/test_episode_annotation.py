from pathlib import Path

import pytest
from pydantic import ValidationError

from robotbench.annotations.annotate_episode import (
    create_annotation,
    save_annotation,
)
from robotbench.domain import (
    EpisodeAnnotation,
    EpisodeOutcome,
    FailureMode,
    TaskPhase,
)

MANIFEST = {
    "schema_version": "0.1.0",
    "run_id": "test-episode-000003",
    "source": {
        "episode_index": 3,
    },
    "timing": {
        "duration_seconds": 11.0,
    },
}


def test_creates_success_annotation() -> None:
    annotation = create_annotation(
        MANIFEST,
        outcome="success",
        notes="Object placed into the tray",
    )

    assert annotation.episode_index == 3
    assert annotation.outcome == EpisodeOutcome.SUCCESS
    assert annotation.failure_mode is None


def test_creates_failure_annotation() -> None:
    annotation = create_annotation(
        MANIFEST,
        outcome="failure",
        failure_mode="object_dropped",
        phase="transport",
        timestamp_seconds=6.42,
        confidence=0.9,
    )

    assert annotation.outcome == EpisodeOutcome.FAILURE
    assert annotation.failure_mode == FailureMode.OBJECT_DROPPED
    assert annotation.phase == TaskPhase.TRANSPORT
    assert annotation.timestamp_seconds == 6.42


@pytest.mark.parametrize(
    ("outcome", "failure_mode"),
    [
        ("success", "collision"),
        ("failure", None),
    ],
)
def test_rejects_inconsistent_outcome(
    outcome: str,
    failure_mode: str | None,
) -> None:
    with pytest.raises(ValidationError):
        EpisodeAnnotation(
            run_id="test",
            episode_index=0,
            outcome=outcome,
            failure_mode=failure_mode,
        )


def test_rejects_timestamp_after_episode_end() -> None:
    with pytest.raises(
        ValueError,
        match="exceeds episode duration",
    ):
        create_annotation(
            MANIFEST,
            outcome="failure",
            failure_mode="timeout",
            timestamp_seconds=20.0,
        )


def test_saves_annotation(
    tmp_path: Path,
) -> None:
    annotation = create_annotation(
        MANIFEST,
        outcome="success",
    )

    output_path = tmp_path / "annotation.json"
    save_annotation(
        annotation,
        output_path,
    )

    assert output_path.exists()
    assert '"outcome": "success"' in output_path.read_text()
