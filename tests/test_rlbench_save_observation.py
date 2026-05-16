from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np
import pytest

from pg3d.envs.rlbench_adapter import Observation, RobotState, SimGroundTruth
from scripts import rlbench_save_observation


def test_parse_camera_names_accepts_comma_separated_subset() -> None:
    assert rlbench_save_observation.parse_camera_names("front,wrist") == ("front", "wrist")


def test_parse_camera_names_rejects_unknown_camera() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="unknown camera names"):
        rlbench_save_observation.parse_camera_names("front,elbow")


def test_build_npz_payload_contains_policy_masks_and_sim_gt_arrays() -> None:
    observation = Observation(
        point_cloud=np.zeros((2, 3), dtype=np.float32),
        point_features={
            "rgb": np.zeros((2, 3), dtype=np.uint8),
            "instance_id": np.asarray([10, 20], dtype=np.int64),
        },
        robot_mask=np.asarray([True, False]),
        object_masks={"target": np.asarray([False, True])},
        robot_state=RobotState(
            joint_positions=np.arange(7, dtype=np.float32),
            joint_velocities=np.ones(7, dtype=np.float32),
            gripper_open=1.0,
        ),
        sim_gt=SimGroundTruth(
            task_name="ReachTarget",
            target_position=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
            task_low_dim_state=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        ),
    )

    payload = rlbench_save_observation.build_npz_payload(observation)

    assert payload["point_cloud"].shape == (2, 3)
    assert payload["agent_pos"].tolist() == list(np.arange(7, dtype=np.float32))
    assert payload["point_feature_rgb"].shape == (2, 3)
    assert payload["point_feature_instance_id"].tolist() == [10, 20]
    assert payload["robot_mask"].tolist() == [True, False]
    assert payload["object_mask_target"].tolist() == [False, True]
    assert payload["sim_gt_target_position"].tolist() == pytest.approx([0.1, 0.2, 0.3])


class FakeObject:
    def __init__(self, handle: int) -> None:
        self.handle = handle

    def get_handle(self) -> int:
        return self.handle


class FakeOwner:
    def __init__(self, handles: list[int]) -> None:
        self.handles = handles

    def get_objects_in_tree(self, *, object_type: object) -> list[FakeObject]:
        _ = object_type
        return [FakeObject(handle) for handle in self.handles]


def test_discover_robot_mask_ids_uses_scene_arm_and_gripper_shapes() -> None:
    task_env = SimpleNamespace(
        _scene=SimpleNamespace(
            _robot_shapes=[FakeObject(1), FakeObject(2)],
            robot=SimpleNamespace(
                arm=FakeOwner([5]),
                gripper=FakeOwner([6]),
            ),
        ),
        _robot=SimpleNamespace(
            arm=FakeOwner([2, 3]),
            gripper=FakeOwner([4]),
        ),
    )
    object_type = SimpleNamespace(SHAPE=object())

    handles = rlbench_save_observation.discover_robot_mask_ids(task_env, object_type)

    assert handles == {1, 2, 3, 4, 5, 6}


def test_discover_reach_object_mask_ids_reads_named_task_shapes() -> None:
    task_env = SimpleNamespace(
        _task=SimpleNamespace(
            target=FakeObject(10),
            distractor0=FakeObject(11),
            distractor1=FakeObject(12),
        )
    )

    assert rlbench_save_observation.discover_reach_object_mask_ids(task_env) == {
        "target": {10},
        "distractor0": {11},
        "distractor1": {12},
    }


def test_disable_waypoint_validation_for_observation_replaces_validate() -> None:
    from pg3d.envs.rlbench_adapter.setup import disable_waypoint_validation_for_observation

    calls = {"count": 0}

    def validate() -> None:
        calls["count"] += 1

    task = SimpleNamespace(validate=validate)
    disable_waypoint_validation_for_observation(SimpleNamespace(_task=task))

    task.validate()

    assert calls["count"] == 0
