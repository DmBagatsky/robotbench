"""Load and inspect LeRobot datasets from the Hugging Face Hub.

The module deliberately avoids the filename ``lerobot.py`` because that name
would shadow the installed third-party ``lerobot`` package when run directly.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from huggingface_hub import snapshot_download
from lerobot.datasets import LeRobotDataset

DEFAULT_REPO_ID = "upna/pickup_blue_block_into_the_yellow_tray"
DEFAULT_REVISION = "main"
DEFAULT_DATASET_SUBDIR = "blue_block_yellow_tray"


def resolve_dataset_root(
    repo_id: str,
    *,
    root: str | Path | None,
    revision: str | None,
    dataset_subdir: str | None,
) -> Path | None:
    """Resolve a usable dataset root, including repositories with nested data.

    The UPNA example repository stores ``data/``, ``meta/``, and ``videos/``
    under ``blue_block_yellow_tray/`` instead of at the repository root.
    LeRobot does not support such a Hub subdirectory directly, so the snapshot
    is downloaded first and the nested directory is passed as a local root.
    """
    if root is not None:
        return Path(root).expanduser()

    effective_subdir = dataset_subdir
    if effective_subdir is None and repo_id == DEFAULT_REPO_ID:
        effective_subdir = DEFAULT_DATASET_SUBDIR

    if effective_subdir is None:
        return None

    snapshot_root = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
        )
    )
    dataset_root = snapshot_root / effective_subdir
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(
            f"LeRobot metadata not found at {info_path}. "
            "Check --repo-id, --revision, and --subdir."
        )

    return dataset_root


def load_lerobot_dataset(
    repo_id: str = DEFAULT_REPO_ID,
    *,
    root: str | Path | None = None,
    episodes: Sequence[int] | None = None,
    revision: str | None = DEFAULT_REVISION,
    dataset_subdir: str | None = None,
) -> LeRobotDataset:
    """Load a LeRobot dataset from the Hub or an optional local cache path.

    Args:
        repo_id: Hugging Face dataset repository identifier.
        root: Optional local directory used to store or read dataset files.
        episodes: Optional episode indices to load, for example ``[0, 1, 2]``.
        revision: Hub branch, tag, or commit. The default dataset is read from
            ``main`` because its owner did not create the expected ``v3.0`` tag.
        dataset_subdir: Optional directory containing the dataset inside the
            Hub repository. It is detected automatically for the default repo.

    Returns:
        The loaded LeRobot dataset.
    """
    dataset_root = resolve_dataset_root(
        repo_id,
        root=root,
        revision=revision,
        dataset_subdir=dataset_subdir,
    )
    options: dict[str, object] = {"repo_id": repo_id}

    if dataset_root is not None:
        options["root"] = dataset_root

    if episodes is not None:
        options["episodes"] = list(episodes)

    if revision is not None:
        options["revision"] = revision

    return LeRobotDataset(**options)


def print_dataset_summary(dataset: LeRobotDataset) -> None:
    """Print the dataset metadata and its basic size information."""
    print(dataset.meta)
    print(f"{len(dataset)} frames, {dataset.num_episodes} episodes")


def parse_episode_indices(value: str) -> list[int]:
    """Parse comma-separated episode indices supplied through the CLI."""
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Episode indices must be comma-separated integers, for example: 0,1,2"
        ) from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--root", type=Path)
    parser.add_argument(
        "--subdir",
        help=(
            "Directory containing data/, meta/, and videos/ inside the Hub "
            "repository (auto-detected for the default dataset)"
        ),
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Hub branch, tag, or commit to load (default: main)",
    )
    parser.add_argument(
        "--episodes",
        type=parse_episode_indices,
        help="Optional comma-separated episode indices, for example: 0,1,2,3,4",
    )
    args = parser.parse_args()

    dataset = load_lerobot_dataset(
        repo_id=args.repo_id,
        root=args.root,
        episodes=args.episodes,
        revision=args.revision,
        dataset_subdir=args.subdir,
    )
    print_dataset_summary(dataset)


if __name__ == "__main__":
    main()
