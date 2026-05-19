from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import scripts.write_maniskill_reach_dataset as writer
from pg3d.envs.maniskill_adapter import Observation, RobotState, SimGroundTruth
from pg3d.envs.maniskill_adapter.dataset import PointCloudCropConfig
from pg3d.envs.maniskill_adapter.reach_config import REACH_TASK_SPECS, reach_task_metadata


def test_collect_episode_appends_hold_chunk_after_success(monkeypatch) -> None:
    env = _FakeReachEnv()
    monkeypatch.setattr(writer, "adapt_observation", _fake_adapt_observation)

    episode = writer._collect_episode(
        env=env,
        seed=3,
        env_id="PG3DReach-Narrow-v0",
        action_mode="abs_joint",
        crop_config=PointCloudCropConfig(
            bounds=np.asarray([[-1, 1], [-1, 1], [0, 1]], dtype=np.float32),
            num_points=4,
        ),
        max_steps=10,
        hold_steps=3,
        gripper_open=0.04,
        sapien=SimpleNamespace(Pose=_FakePose),
        planner_cls=_FakePlanner,
    )

    assert episode is not None
    assert episode.state.shape[0] == 5
    assert episode.metadata["first_success_step"] == 2
    assert episode.metadata["hold_steps_requested"] == 3
    assert episode.metadata["hold_steps_recorded"] == 3
    assert episode.metadata["success"] is True
    np.testing.assert_allclose(episode.sim_action[2:, :7], episode.state[2:, :7])
    np.testing.assert_allclose(episode.sim_action[2:, 7], 0.04)


def test_dataset_stats_reports_hold_coverage() -> None:
    episode = writer.ReachEpisodeData(
        state=np.zeros((2, 9), dtype=np.float32),
        action=np.ones((2, 7), dtype=np.float32),
        sim_action=np.ones((2, 8), dtype=np.float32),
        point_cloud=np.zeros((2, 4, 3), dtype=np.float32),
        robot_mask=np.asarray([[True, False, False, False], [True, True, False, False]]),
        point_valid_mask=np.ones((2, 4), dtype=bool),
        target_position=np.zeros((2, 3), dtype=np.float32),
        tcp_pose=np.zeros((2, 7), dtype=np.float32),
        success=np.asarray([True, True]),
        metadata={
            "success": True,
            "final_distance": 0.01,
            "hold_steps_requested": 2,
            "hold_steps_recorded": 2,
        },
    )

    stats = writer._dataset_stats([episode])

    assert stats["num_episodes"] == 1
    assert stats["success_rate"] == 1.0
    assert stats["hold_coverage"] == 1.0
    assert stats["robot_mask_points"]["mean"] == 1.5


def test_workspace_reach_task_defaults_are_simulator_free() -> None:
    spec = REACH_TASK_SPECS["PG3DReach-Workspace-v0"]

    assert spec.max_episode_steps == 100
    assert spec.goal_center == (0.05, 0.0, 0.45)
    assert spec.goal_half_extents == (0.35, 0.35, 0.30)
    assert spec.goal_bounds == ((-0.3, 0.4), (-0.35, 0.35), (0.15, 0.75))

    metadata = reach_task_metadata("PG3DReach-Workspace-v0")
    assert metadata["goal_center"] == [0.05, 0.0, 0.45]
    assert metadata["goal_half_extents"] == [0.35, 0.35, 0.3]
    assert metadata["goal_bounds"] == [[-0.3, 0.4], [-0.35, 0.35], [0.15, 0.75]]


def test_balanced_workspace_reach_task_metadata_has_weighted_regions() -> None:
    metadata = reach_task_metadata("PG3DReach-BalancedWorkspace-v0")

    assert metadata["goal_bounds"] == [[-0.26, 0.34], [-0.3, 0.3], [0.2, 0.68]]
    assert metadata["goal_regions"][0]["name"] == "core_practical"
    assert metadata["goal_regions"][0]["weight"] == 0.7
    assert metadata["goal_regions"][0]["bounds"] == [
        [-0.14, 0.24],
        [-0.2, 0.2],
        [0.28, 0.56],
    ]
    assert metadata["goal_regions"][1]["name"] == "outer_practical"
    assert metadata["goal_regions"][1]["weight"] == 0.3


class _FakePose:
    def __init__(self, p=None, q=None) -> None:
        self.p = np.asarray([0.0, 0.0, 0.2] if p is None else p, dtype=np.float32)
        self.q = np.asarray([1.0, 0.0, 0.0, 0.0] if q is None else q, dtype=np.float32)
        self.raw_pose = np.concatenate([self.p, self.q]).astype(np.float32)


class _FakePlanner:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def move_to_pose_with_screw(self, goal_pose, *, dry_run: bool):
        return {
            "position": np.asarray(
                [
                    [0.1, 0.2, 0.3, -1.0, 0.0, 1.0, 0.5, 0.04],
                    [0.2, 0.3, 0.4, -0.9, 0.1, 1.1, 0.6, 0.04],
                ],
                dtype=np.float32,
            ),
            "status": "Success",
        }

    def close(self) -> None:
        pass


class _FakeReachEnv:
    def __init__(self) -> None:
        self.action_space = SimpleNamespace(shape=(8,))
        self.step_count = 0
        self.unwrapped = self
        self.goal_site = SimpleNamespace(pose=_FakePose(p=[0.2, 0.0, 0.3]))
        self.agent = SimpleNamespace(
            robot=SimpleNamespace(qpos=np.zeros(9, dtype=np.float32), pose=_FakePose()),
            tcp=SimpleNamespace(pose=_FakePose()),
        )

    def reset(self, *, seed: int, options: dict[str, object]):
        self.step_count = 0
        self.agent.robot.qpos = np.zeros(9, dtype=np.float32)
        return {"step": 0}, {"success": np.asarray([False])}

    def step(self, action: np.ndarray):
        self.step_count += 1
        self.agent.robot.qpos[:7] = np.asarray(action, dtype=np.float32)[:7]
        success = self.step_count >= 2
        distance = 0.01 if success else 0.2
        info = {
            "success": np.asarray([success]),
            "tcp_to_goal_dist": np.asarray([distance], dtype=np.float32),
        }
        return {"step": self.step_count}, 0.0, np.asarray([success]), np.asarray([False]), info


def _fake_adapt_observation(obs, *, info, env, task_name):
    success = bool(np.asarray(info.get("success", [False])).reshape(-1)[0])
    return Observation(
        point_cloud=np.asarray(
            [[0.0, 0.0, 0.2], [0.1, 0.0, 0.2], [0.2, 0.0, 0.2], [0.3, 0.0, 0.2]],
            dtype=np.float32,
        ),
        point_features={},
        robot_mask=np.asarray([True, True, False, False]),
        robot_state=RobotState(
            joint_positions=env.unwrapped.agent.robot.qpos.copy(),
            tcp_pose=np.asarray([0, 0, 0.2, 1, 0, 0, 0], dtype=np.float32),
        ),
        sim_gt=SimGroundTruth(
            task_name=task_name,
            target_position=np.asarray([0.2, 0.0, 0.3], dtype=np.float32),
            success=success,
        ),
    )
