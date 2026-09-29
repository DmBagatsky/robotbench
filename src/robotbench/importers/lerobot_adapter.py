"""Convert LeRobot episodes into RobotBench ``RobotRun`` manifests."""

from __future__ import annotations

import argparse
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lerobot.datasets import LeRobotDataset

from robotbench.domain import (
    RobotInfo,
    RobotRun,
    RunSignals,
    RunSource,
    RunTiming,
    SignalSpec,
    TaskInfo,
    VideoStreamSpec,
)
from robotbench.importers.lerobot_loader import (
    DEFAULT_DATASET_SUBDIR,
    DEFAULT_REPO_ID,
    DEFAULT_REVISION,
    load_lerobot_dataset,
    parse_episode_indices,
)


def _as_mapping(value: Any) -> dict[str, Any]:
    """Convert dictionaries, Pydantic models, and pandas rows to a dict."""
    if isinstance(value, Mapping):
        return dict(value)

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return dict(model_dump())

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())

    raise TypeError(f"Cannot convert {type(value).__name__} to a mapping")


def _get_episode_metadata(
    dataset: LeRobotDataset, episode_index: int
) -> dict[str, Any]:
    """Return metadata for an episode across supported LeRobot versions."""
    episodes = dataset.meta.episodes
    if episodes is None:
        raise ValueError("The dataset does not contain episode metadata")

    # LeRobot versions that expose episode metadata as a pandas DataFrame.
    if hasattr(episodes, "iloc"):
        columns = getattr(episodes, "columns", ())
        if "episode_index" in columns:
            matches = episodes.loc[episodes["episode_index"] == episode_index]
            if len(matches) == 0:
                raise IndexError(f"Episode {episode_index} does not exist")
            return _as_mapping(matches.iloc[0])

        try:
            return _as_mapping(episodes.iloc[episode_index])
        except IndexError as error:
            raise IndexError(f"Episode {episode_index} does not exist") from error

    # Newer LeRobot versions expose metadata as a Hugging Face Dataset.
    try:
        episode_indices = [int(value) for value in episodes["episode_index"]]
        position = episode_indices.index(episode_index)
    except (KeyError, ValueError) as error:
        raise IndexError(f"Episode {episode_index} does not exist") from error

    return _as_mapping(episodes[position])


def _get_feature(dataset: LeRobotDataset, key: str) -> dict[str, Any]:
    """Read one feature declaration from LeRobot metadata."""
    features = dataset.meta.features
    if key not in features:
        raise KeyError(f"Required feature {key!r} is missing from the dataset")
    return _as_mapping(features[key])


def _get_shape(feature: Mapping[str, Any], key: str) -> list[int]:
    shape = feature.get("shape")
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
        raise ValueError(f"Feature {key!r} does not have a valid shape")
    return [int(dimension) for dimension in shape]


def _get_labels(feature: Mapping[str, Any]) -> list[str] | None:
    names = feature.get("names")
    if names is None:
        return None
    if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
        return None
    return [str(name) for name in names]


def _build_signal(dataset: LeRobotDataset, key: str) -> SignalSpec:
    feature = _get_feature(dataset, key)
    return SignalSpec(
        source_key=key,
        dtype=str(feature["dtype"]),
        shape=_get_shape(feature, key),
        labels=_get_labels(feature),
    )


def _video_streams(dataset: LeRobotDataset) -> list[VideoStreamSpec]:
    streams: list[VideoStreamSpec] = []

    for key, raw_feature in dataset.meta.features.items():
        feature = _as_mapping(raw_feature)
        if feature.get("dtype") != "video":
            continue

        shape = _get_shape(feature, key)
        if len(shape) < 2:
            raise ValueError(f"Video feature {key!r} must contain height and width")

        raw_info = feature.get("info") or {}
        info = _as_mapping(raw_info) if raw_info else {}
        height = int(info.get("video.height", info.get("height", shape[0])))
        width = int(info.get("video.width", info.get("width", shape[1])))
        fps = float(info.get("video.fps", info.get("fps", dataset.meta.fps)))
        codec = info.get("video.codec", info.get("codec"))

        streams.append(
            VideoStreamSpec(
                source_key=key,
                camera=key.removeprefix("observation.images."),
                width=width,
                height=height,
                fps=fps,
                codec=str(codec) if codec is not None else None,
            )
        )

    return streams


def _task_info(dataset: LeRobotDataset, episode: Mapping[str, Any]) -> TaskInfo:
    raw_tasks = episode.get("tasks")
    if isinstance(raw_tasks, str):
        tasks = [raw_tasks]
    elif isinstance(raw_tasks, Sequence):
        tasks = [str(task) for task in raw_tasks]
    else:
        tasks = []

    if not tasks:
        raise ValueError(
            f"Episode {episode.get('episode_index')} does not contain a task description"
        )

    task_name = tasks[0]
    task_index = None
    get_task_index = getattr(dataset.meta, "get_task_index", None)
    if callable(get_task_index):
        result = get_task_index(task_name)
        task_index = int(result) if result is not None else None

    return TaskInfo(name=task_name, task_index=task_index)


def _robot_type(dataset: LeRobotDataset) -> str:
    robot_type = getattr(dataset.meta, "robot_type", None)
    if robot_type:
        return str(robot_type)

    info = _as_mapping(dataset.meta.info)
    robot_type = info.get("robot_type")
    if not robot_type:
        raise ValueError("The dataset does not declare robot_type")
    return str(robot_type)


def _run_id(repo_id: str, episode_index: int) -> str:
    repo_slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", repo_id).strip("-")
    return f"{repo_slug}-episode-{episode_index:06d}"


def adapt_lerobot_episode(
    dataset: LeRobotDataset,
    episode_index: int,
    *,
    repo_id: str,
    revision: str,
    dataset_subdir: str | None,
) -> RobotRun:
    """Convert one LeRobot episode into a validated RobotRun."""
    episode = _get_episode_metadata(dataset, episode_index)

    num_frames = episode.get("length")
    if num_frames is None:
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        num_frames = end - start

    return RobotRun(
        run_id=_run_id(repo_id, episode_index),
        source=RunSource(
            repo_id=repo_id,
            revision=revision,
            episode_index=episode_index,
            dataset_subdir=dataset_subdir,
        ),
        robot=RobotInfo(type=_robot_type(dataset)),
        task=_task_info(dataset, episode),
        timing=RunTiming(
            fps=float(dataset.meta.fps),
            num_frames=int(num_frames),
        ),
        signals=RunSignals(
            actions=_build_signal(dataset, "action"),
            states=_build_signal(dataset, "observation.state"),
        ),
        video_streams=_video_streams(dataset),
    )


def export_lerobot_runs(
    *,
    repo_id: str,
    revision: str,
    episode_indices: Sequence[int],
    output_dir: Path,
    root: Path | None = None,
    dataset_subdir: str | None = None,
) -> list[Path]:
    """Load selected episodes and write one RobotRun manifest per episode."""
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
        output_path = output_dir / f"episode_{int(episode_index):06d}.json"
        output_path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
        paths.append(output_path)
        print(
            f"Created {output_path}: "
            f"{run.timing.num_frames} frames, "
            f"{run.timing.duration_seconds:.2f} seconds"
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
        default=Path("artifacts/runs"),
        help="Directory for generated RobotRun JSON manifests",
    )
    args = parser.parse_args()

    export_lerobot_runs(
        repo_id=args.repo_id,
        revision=args.revision,
        episode_indices=args.episodes,
        output_dir=args.output,
        root=args.root,
        dataset_subdir=args.subdir,
    )


if __name__ == "__main__":
    main()
