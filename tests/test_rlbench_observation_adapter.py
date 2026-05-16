from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pg3d.envs.rlbench_adapter import Observation, RobotState, adapt_rlbench_observation


def make_fake_observation(**overrides: object) -> SimpleNamespace:
    point_cloud = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[np.nan, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    rgb = np.asarray(
        [
            [[10, 0, 0], [20, 0, 0]],
            [[30, 0, 0], [40, 0, 0]],
        ],
        dtype=np.uint8,
    )
    mask = np.asarray([[100, 200], [300, 400]], dtype=np.int64)
    values: dict[str, object] = {
        "left_shoulder_point_cloud": point_cloud,
        "left_shoulder_rgb": rgb,
        "left_shoulder_mask": mask,
        "joint_positions": np.arange(7, dtype=np.float32),
        "joint_velocities": np.arange(7, dtype=np.float32) * 0.1,
        "joint_forces": np.arange(7, dtype=np.float32) * 0.2,
        "gripper_open": 1.0,
        "gripper_pose": np.arange(7, dtype=np.float32),
        "gripper_matrix": np.eye(4, dtype=np.float32),
        "gripper_joint_positions": np.asarray([0.01, 0.02], dtype=np.float32),
        "gripper_touch_forces": np.arange(6, dtype=np.float32),
        "task_low_dim_state": np.asarray([0.4, 0.5, 0.6], dtype=np.float32),
        "misc": {"variation_index": 3},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_adapt_rlbench_observation_preserves_point_mask_alignment() -> None:
    adapted = adapt_rlbench_observation(
        make_fake_observation(),
        descriptions=["reach the red target"],
        camera_names=("left_shoulder",),
        robot_mask_ids={200},
        object_mask_ids={"target": {400}},
    )

    np.testing.assert_allclose(
        adapted.point_cloud,
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    )
    assert adapted.point_features["rgb"].tolist() == [[10, 0, 0], [20, 0, 0], [40, 0, 0]]
    assert adapted.point_features["instance_id"].tolist() == [100, 200, 400]
    assert adapted.robot_mask is not None
    assert adapted.robot_mask.tolist() == [False, True, False]
    assert adapted.object_masks["target"].tolist() == [False, False, True]
    assert adapted.sim_gt is not None
    np.testing.assert_allclose(adapted.sim_gt.target_position, [0.4, 0.5, 0.6])
    assert adapted.sim_gt.descriptions == ("reach the red target",)


def test_policy_inputs_exclude_masks_and_sim_ground_truth_by_default() -> None:
    adapted = adapt_rlbench_observation(
        make_fake_observation(),
        camera_names=("left_shoulder",),
        robot_mask_ids={200},
        object_mask_ids={"target": {400}},
    )

    policy_inputs = adapted.as_policy_inputs()

    assert set(policy_inputs) == {"point_cloud", "agent_pos"}
    assert policy_inputs["point_cloud"].shape == (3, 3)
    assert policy_inputs["agent_pos"].tolist() == list(np.arange(7, dtype=np.float32))

    rgb_policy_inputs = adapted.as_policy_inputs(include_rgb=True)
    assert rgb_policy_inputs["point_cloud"].shape == (3, 6)
    assert np.max(rgb_policy_inputs["point_cloud"][:, 3:]) <= 1.0


def test_rgb_coded_masks_are_decoded_to_instance_ids() -> None:
    rgb_mask = np.asarray([[[1, 0, 0], [0, 1, 0]]], dtype=np.uint8)
    point_cloud = np.asarray([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float32)
    adapted = adapt_rlbench_observation(
        make_fake_observation(
            left_shoulder_point_cloud=point_cloud,
            left_shoulder_mask=rgb_mask,
            left_shoulder_rgb=None,
        ),
        camera_names=("left_shoulder",),
        robot_mask_ids={1},
        object_mask_ids={"encoded_256": {256}},
    )

    assert "rgb" not in adapted.point_features
    assert adapted.point_features["instance_id"].tolist() == [1, 256]
    assert adapted.robot_mask is not None
    assert adapted.robot_mask.tolist() == [True, False]
    assert adapted.object_masks["encoded_256"].tolist() == [False, True]


def test_model_validation_rejects_feature_length_mismatch() -> None:
    with pytest.raises(ValueError, match="point_features\\['rgb'\\] first dimension"):
        Observation(
            point_cloud=np.zeros((3, 3), dtype=np.float32),
            point_features={"rgb": np.zeros((2, 3), dtype=np.uint8)},
            robot_state=RobotState(joint_positions=np.zeros(7, dtype=np.float32)),
        )


def test_model_validation_rejects_robot_mask_length_mismatch() -> None:
    with pytest.raises(ValueError, match="robot_mask must have shape"):
        Observation(
            point_cloud=np.zeros((3, 3), dtype=np.float32),
            point_features={},
            robot_mask=np.zeros(2, dtype=bool),
            robot_state=RobotState(joint_positions=np.zeros(7, dtype=np.float32)),
        )


def test_adapter_requires_joint_positions() -> None:
    with pytest.raises(ValueError, match="missing joint_positions"):
        adapt_rlbench_observation(
            make_fake_observation(joint_positions=None),
            camera_names=("left_shoulder",),
        )
