"""Local RobotBench episode review interface."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from robotbench.analysis.episode_health import _episode_columns
from robotbench.annotations.annotate_episode import (
    create_annotation,
    save_annotation,
)
from robotbench.domain import (
    EpisodeOutcome,
    FailureMode,
    TaskPhase,
)
from robotbench.importers.lerobot_loader import (
    DEFAULT_DATASET_SUBDIR,
    DEFAULT_REPO_ID,
    DEFAULT_REVISION,
    load_lerobot_dataset,
)

PROJECT_ROOT = Path.cwd()

RUNS_DIR = PROJECT_ROOT / "artifacts/runs"
HEALTH_DIR = PROJECT_ROOT / "artifacts/health"
ANNOTATIONS_DIR = PROJECT_ROOT / "artifacts/annotations"
VIDEOS_DIR = PROJECT_ROOT / "artifacts/source_videos"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_manifests() -> dict[int, Path]:
    manifests: dict[int, Path] = {}

    for path in sorted(RUNS_DIR.glob("episode_*.json")):
        manifest = load_json(path)
        episode_index = int(
            manifest.get(
                "episode_index",
                manifest["source"]["episode_index"],
            )
        )
        manifests[episode_index] = path

    return manifests


def calculate_episode_windows(
    manifests: dict[int, Path],
) -> dict[int, tuple[float, float]]:
    """Calculate episode offsets inside a concatenated video chunk."""
    windows: dict[int, tuple[float, float]] = {}
    cursor = 0.0

    for episode_index in sorted(manifests):
        manifest = load_json(manifests[episode_index])
        duration = float(manifest["timing"]["duration_seconds"])

        windows[episode_index] = (
            cursor,
            cursor + duration,
        )
        cursor += duration

    return windows


def find_camera_video(
    camera: str,
) -> Path | None:
    extensions = {
        ".mp4",
        ".mkv",
        ".mov",
        ".avi",
    }

    candidates = [
        path
        for path in VIDEOS_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in extensions
        and f"observation.images.{camera}" in str(path)
    ]

    return sorted(candidates)[0] if candidates else None


@st.cache_resource(show_spinner="Loading LeRobot dataset…")
def get_dataset(
    episode_indices: tuple[int, ...],
):
    return load_lerobot_dataset(
        repo_id=DEFAULT_REPO_ID,
        revision=DEFAULT_REVISION,
        episodes=list(episode_indices),
        root=None,
        dataset_subdir=DEFAULT_DATASET_SUBDIR,
    )


@st.cache_data
def load_health_report(
    episode_index: int,
) -> dict[str, Any] | None:
    path = HEALTH_DIR / f"episode_{episode_index:06d}.health.json"

    if not path.exists():
        return None

    return load_json(path)


def load_annotation(
    episode_index: int,
) -> dict[str, Any] | None:
    path = ANNOTATIONS_DIR / f"episode_{episode_index:06d}.annotation.json"

    if not path.exists():
        return None

    return load_json(path)


def show_videos(
    episode_index: int,
    windows: dict[int, tuple[float, float]],
) -> None:
    st.subheader("Video")

    start_time, end_time = windows[episode_index]

    st.caption(f"Chunk interval: {start_time:.2f}s – " f"{end_time:.2f}s")

    front_column, wrist_column = st.columns(2)

    for column, camera in (
        (front_column, "front"),
        (wrist_column, "wrist"),
    ):
        with column:
            st.markdown(f"#### {camera.title()} camera")

            video_path = find_camera_video(camera)

            if video_path is None:
                st.warning(f"No local video found for {camera}")
                continue

            st.video(
                str(video_path),
                start_time=math.floor(start_time),
                end_time=math.ceil(end_time),
            )

            st.caption(video_path.name)


def show_signals(
    dataset: Any,
    manifest: dict[str, Any],
    episode_index: int,
) -> None:
    st.subheader("Actions and states")

    columns = _episode_columns(
        dataset,
        episode_index,
    )

    actions = columns["action"]
    states = columns["state"]

    fps = float(manifest["timing"]["fps"])
    timestamps = np.arange(len(actions), dtype=np.float64) / fps

    labels = manifest["signals"]["actions"].get("labels")

    if not labels:
        labels = [f"joint_{index}" for index in range(actions.shape[1])]

    tabs = st.tabs(labels)

    for joint_index, tab in enumerate(tabs):
        with tab:
            frame = pd.DataFrame(
                {
                    "time_seconds": timestamps,
                    "action": actions[:, joint_index],
                    "state": states[:, joint_index],
                }
            ).set_index("time_seconds")

            st.line_chart(
                frame,
                x_label="Time, seconds",
                y_label="Position",
            )

            difference = np.abs(actions[:, joint_index] - states[:, joint_index])

            metric_left, metric_right = st.columns(2)

            metric_left.metric(
                "Mean tracking error",
                f"{np.mean(difference):.4f}",
            )
            metric_right.metric(
                "Maximum tracking error",
                f"{np.max(difference):.4f}",
            )


def show_health(
    episode_index: int,
) -> None:
    st.subheader("Episode health")

    report = load_health_report(episode_index)

    if report is None:
        st.warning("Health report not found")
        return

    first, second, third, fourth = st.columns(4)

    first.metric(
        "Overall",
        report["overall_status"].upper(),
    )
    second.metric(
        "Passed",
        report["passed_checks"],
    )
    third.metric(
        "Warnings",
        report["warning_checks"],
    )
    fourth.metric(
        "Failed",
        report["failed_checks"],
    )

    rows = [
        {
            "status": check["status"],
            "code": check["code"],
            "message": check["message"],
        }
        for check in report["checks"]
    ]

    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        use_container_width=True,
    )


def show_annotation_form(
    manifest: dict[str, Any],
    episode_index: int,
) -> None:
    st.subheader("Outcome annotation")

    current = load_annotation(episode_index) or {}

    outcome_values = [value.value for value in EpisodeOutcome]
    failure_values = [
        "",
        *(value.value for value in FailureMode),
    ]
    phase_values = [
        "",
        *(value.value for value in TaskPhase),
    ]

    current_outcome = current.get(
        "outcome",
        EpisodeOutcome.SUCCESS.value,
    )
    current_failure = current.get("failure_mode") or ""
    current_phase = current.get("phase") or ""

    with st.form(f"annotation-form-{episode_index}"):
        outcome = st.selectbox(
            "Outcome",
            options=outcome_values,
            index=outcome_values.index(current_outcome),
        )

        failure_mode = st.selectbox(
            "Failure mode",
            options=failure_values,
            index=failure_values.index(current_failure),
            help=("Leave empty for successful episodes"),
        )

        phase = st.selectbox(
            "Task phase",
            options=phase_values,
            index=phase_values.index(current_phase),
        )

        duration = float(manifest["timing"]["duration_seconds"])

        timestamp_seconds = st.number_input(
            "Failure timestamp, seconds",
            min_value=0.0,
            max_value=duration,
            value=float(current.get("timestamp_seconds") or 0.0),
            step=0.1,
            help=("Relative to the beginning " "of this episode"),
        )

        confidence = st.slider(
            "Confidence",
            min_value=0.0,
            max_value=1.0,
            value=float(
                current.get(
                    "confidence",
                    1.0,
                )
            ),
            step=0.05,
        )

        notes = st.text_area(
            "Notes",
            value=current.get("notes") or "",
        )

        submitted = st.form_submit_button(
            "Save annotation",
            type="primary",
        )

    if not submitted:
        return

    try:
        annotation = create_annotation(
            manifest,
            outcome=outcome,
            failure_mode=(failure_mode or None),
            phase=phase or None,
            timestamp_seconds=(
                None if outcome == EpisodeOutcome.SUCCESS.value else timestamp_seconds
            ),
            confidence=confidence,
            notes=notes or None,
        )

        output_path = ANNOTATIONS_DIR / (
            f"episode_" f"{episode_index:06d}" f".annotation.json"
        )

        save_annotation(
            annotation,
            output_path,
        )

        st.success(f"Annotation saved to {output_path}")

    except ValueError as error:
        st.error(str(error))


def main() -> None:
    st.set_page_config(
        page_title="RobotBench",
        page_icon="🤖",
        layout="wide",
    )

    st.title("RobotBench")
    st.caption("Reliability and diagnostics " "for real-world robot learning")

    manifests = discover_manifests()

    if not manifests:
        st.error(f"No run manifests found in {RUNS_DIR}")
        st.stop()

    episode_indices = tuple(sorted(manifests))
    windows = calculate_episode_windows(manifests)

    selected_episode = st.sidebar.selectbox(
        "Episode",
        options=episode_indices,
        format_func=lambda value: (f"Episode {value}"),
    )

    manifest = load_json(manifests[selected_episode])

    st.sidebar.markdown("### Run")

    st.sidebar.code(
        manifest["run_id"],
        language=None,
    )

    st.sidebar.metric(
        "Frames",
        manifest["timing"]["num_frames"],
    )
    st.sidebar.metric(
        "Duration",
        (f"{manifest['timing']['duration_seconds']}" " s"),
    )
    st.sidebar.metric(
        "FPS",
        manifest["timing"]["fps"],
    )

    annotation = load_annotation(selected_episode)

    status_columns = st.columns(3)

    status_columns[0].metric(
        "Episode",
        selected_episode,
    )
    status_columns[1].metric(
        "Task",
        manifest["task"]["name"],
    )
    status_columns[2].metric(
        "Outcome",
        (annotation["outcome"].upper() if annotation else "NOT ANNOTATED"),
    )

    show_videos(
        selected_episode,
        windows,
    )

    dataset = get_dataset(episode_indices)

    show_signals(
        dataset,
        manifest,
        selected_episode,
    )

    show_health(selected_episode)

    show_annotation_form(
        manifest,
        selected_episode,
    )


if __name__ == "__main__":
    main()
