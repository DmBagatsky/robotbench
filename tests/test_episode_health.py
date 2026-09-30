from types import SimpleNamespace

import numpy as np

from robotbench.analysis.episode_health import (
    _action_jump_check,
    _finite_values_check,
    _frame_sequence_check,
    _low_variation_check,
    _signal_shape_check,
    _timing_check,
)


def test_detects_nan_in_actions() -> None:
    actions = np.zeros((4, 2), dtype=np.float64)
    actions[2, 1] = np.nan

    states = np.zeros((4, 2), dtype=np.float64)
    timestamps = np.arange(4, dtype=np.float64) / 30.0

    check = _finite_values_check(actions, states, timestamps)

    assert check.status == "fail"
    assert check.metrics["action_non_finite"] == 1


def test_detects_missing_frame_index() -> None:
    frame_indices = np.array([0, 1, 3, 4], dtype=np.int64)

    check = _frame_sequence_check(frame_indices)

    assert check.status == "fail"
    assert check.metrics["mismatch_count"] > 0


def test_detects_repeated_timestamp() -> None:
    timestamps = np.array(
        [
            0.0,
            1.0 / 30.0,
            1.0 / 30.0,
            3.0 / 30.0,
        ],
        dtype=np.float64,
    )

    check = _timing_check(timestamps, expected_fps=30.0)

    assert check.status == "fail"
    assert check.metrics["non_increasing_count"] == 1


def test_detects_wrong_state_shape() -> None:
    run = SimpleNamespace(
        signals=SimpleNamespace(
            actions=SimpleNamespace(shape=[6]),
            states=SimpleNamespace(shape=[6]),
        )
    )

    actions = np.zeros((10, 6), dtype=np.float64)
    states = np.zeros((10, 5), dtype=np.float64)

    check = _signal_shape_check(actions, states, run)

    assert check.status == "fail"
    assert check.metrics["state_shape"] == "(5,)"
    assert check.metrics["expected_state_shape"] == "(6,)"


def test_detects_frozen_joint() -> None:
    actions = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.1],
            [0.0, 0.2],
            [0.0, 0.3],
        ],
        dtype=np.float64,
    )
    states = actions.copy()

    check = _low_variation_check(
        actions,
        states,
        labels=["frozen_joint", "moving_joint"],
    )

    assert check.status == "warning"
    assert check.metrics["inactive_joint_count"] == 1
    assert "frozen_joint" in check.metrics["inactive_joints"]


def test_detects_extreme_action_jump() -> None:
    actions = np.array(
        [
            [0.0],
            [0.1],
            [0.9],
            [1.0],
        ],
        dtype=np.float64,
    )

    check = _action_jump_check(actions)

    assert check.status == "warning"
    assert check.metrics["discontinuity_count"] == 1
    assert check.metrics["max_relative_jump"] > 0.5
