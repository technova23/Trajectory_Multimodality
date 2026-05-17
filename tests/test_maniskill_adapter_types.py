from __future__ import annotations

import numpy as np
import pytest

from pg3d.envs.maniskill_adapter import Observation, RobotState, SimGroundTruth


def test_policy_inputs_exclude_masks_and_sim_ground_truth_by_default() -> None:
    observation = Observation(
        point_cloud=np.zeros((2, 3), dtype=np.float32),
        point_features={
            "rgb": np.asarray([[255, 0, 0], [0, 255, 0]], dtype=np.uint8),
            "segmentation": np.asarray([1, 2], dtype=np.int64),
        },
        robot_mask=np.asarray([True, False]),
        object_masks={"cube": np.asarray([False, True])},
        robot_state=RobotState(
            joint_positions=np.arange(7, dtype=np.float32),
            joint_velocities=np.ones(7, dtype=np.float32),
            tcp_pose=np.zeros(7, dtype=np.float32),
        ),
        sim_gt=SimGroundTruth(
            task_name="PickCube-v1",
            target_position=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
            success=False,
        ),
    )

    policy_inputs = observation.as_policy_inputs()

    assert set(policy_inputs) == {"point_cloud", "agent_pos"}
    assert policy_inputs["point_cloud"].shape == (2, 3)
    assert policy_inputs["agent_pos"].tolist() == list(np.arange(7, dtype=np.float32))

    rgb_policy_inputs = observation.as_policy_inputs(include_rgb=True)
    assert rgb_policy_inputs["point_cloud"].shape == (2, 6)
    assert np.max(rgb_policy_inputs["point_cloud"][:, 3:]) <= 1.0


def test_observation_validation_rejects_feature_length_mismatch() -> None:
    with pytest.raises(ValueError, match="point_features\\['rgb'\\] first dimension"):
        Observation(
            point_cloud=np.zeros((3, 3), dtype=np.float32),
            point_features={"rgb": np.zeros((2, 3), dtype=np.uint8)},
            robot_state=RobotState(joint_positions=np.zeros(7, dtype=np.float32)),
        )


def test_observation_validation_rejects_robot_mask_length_mismatch() -> None:
    with pytest.raises(ValueError, match="robot_mask must have shape"):
        Observation(
            point_cloud=np.zeros((3, 3), dtype=np.float32),
            point_features={},
            robot_mask=np.zeros(2, dtype=bool),
            robot_state=RobotState(joint_positions=np.zeros(7, dtype=np.float32)),
        )
