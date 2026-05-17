from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from pg3d.envs.maniskill_adapter import adapt_observation, register_pg3d_reach_envs
from pg3d.envs.maniskill_adapter.dataset import (
    DEFAULT_WORKSPACE_BOUNDS,
    ActionMode,
    PointCloudCropConfig,
    ReachEpisodeData,
    git_commit_info,
    observation_to_dataset_row,
    write_reach_zarr,
)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import gymnasium as gym
        import mani_skill
        import mani_skill.envs  # noqa: F401
        import sapien
        from mani_skill.examples.motionplanning.panda.motionplanner import (
            PandaArmMotionPlanningSolver,
        )
    except Exception as exc:
        print(
            f"Failed to import ManiSkill motion-planning stack: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print(
            "Install with: "
            "uv sync --extra cu129 --extra maniskill --group dev --group notebooks",
            file=sys.stderr,
        )
        return 2

    register_pg3d_reach_envs()
    crop_config = PointCloudCropConfig(
        bounds=np.asarray(args.workspace_bounds),
        num_points=args.num_points,
    )
    env_kwargs = _env_kwargs(args)
    env: Any | None = None
    episodes: list[ReachEpisodeData] = []
    skipped: list[dict[str, Any]] = []
    try:
        env = gym.make(args.env_id, **env_kwargs)
        attempt = 0
        while len(episodes) < args.num_demos and attempt < args.max_attempts:
            seed = args.seed_start + attempt
            attempt += 1
            episode = _collect_episode(
                env=env,
                seed=seed,
                env_id=args.env_id,
                action_mode=args.action_mode,
                crop_config=crop_config,
                max_steps=args.max_steps_per_demo,
                hold_steps=args.hold_steps,
                gripper_open=args.gripper_open,
                sapien=sapien,
                planner_cls=PandaArmMotionPlanningSolver,
            )
            if episode is None:
                skipped.append({"seed": seed, "reason": "planner_failed_or_empty"})
                continue
            if not args.keep_failures and not bool(episode.metadata.get("success", False)):
                skipped.append(
                    {
                        "seed": seed,
                        "reason": "unsuccessful_replay",
                        "final_distance": episode.metadata.get("final_distance"),
                    }
                )
                continue
            episodes.append(episode)
            print(
                "demo "
                f"{len(episodes)}/{args.num_demos}: seed={seed} "
                f"steps={episode.state.shape[0]} "
                f"hold={episode.metadata.get('hold_steps_recorded')} "
                f"success={episode.metadata.get('success')} "
                f"final_distance={episode.metadata.get('final_distance'):.4f}"
            )
    except Exception as exc:
        print(f"Failed to write reach dataset: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if env is not None:
            env.close()

    if not episodes:
        print("No usable reach demonstrations were collected.", file=sys.stderr)
        return 1

    repo_root = Path(__file__).resolve().parents[1]
    dataset_stats = _dataset_stats(episodes)
    metadata = {
        "env_id": args.env_id,
        "env_kwargs": env_kwargs,
        "num_requested_demos": args.num_demos,
        "num_collected_demos": len(episodes),
        "num_attempts": attempt,
        "skipped": skipped,
        "seed_start": args.seed_start,
        "action_mode": args.action_mode,
        "control_mode": args.control_mode,
        "hold_steps": args.hold_steps,
        "crop": crop_config.to_json(),
        "dataset_stats": dataset_stats,
        "camera": {
            "obs_mode": args.obs_mode,
            "shader": args.shader,
            "source": "ManiSkill default sensor config for PG3DReach",
        },
        "versions": {
            "mani_skill": getattr(mani_skill, "__version__", None),
            "sapien": getattr(sapien, "__version__", None),
        },
        "git": {
            "pg3d": git_commit_info(repo_root),
            "external_dp3": git_commit_info(repo_root / "external" / "dp3"),
        },
    }
    summary = write_reach_zarr(args.output, episodes, metadata=metadata, overwrite=args.overwrite)
    print(f"saved dataset: {args.output}")
    print(f"summary: {summary}")
    print("dataset_stats: " + json_dumps(dataset_stats))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a smoke-scale pg3d ManiSkill reach dataset."
    )
    parser.add_argument("--env-id", default="PG3DReach-Narrow-v0")
    parser.add_argument("--num-demos", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=25)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--obs-mode", default="pointcloud", choices=["pointcloud"])
    parser.add_argument("--action-mode", default="abs_joint", choices=["abs_joint", "delta_joint"])
    parser.add_argument("--control-mode", default="pd_joint_pos")
    parser.add_argument("--robot-uid", default="panda")
    parser.add_argument("--num-points", type=int, default=512)
    parser.add_argument(
        "--workspace-bounds",
        type=float,
        nargs=6,
        default=DEFAULT_WORKSPACE_BOUNDS.reshape(-1).tolist(),
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"),
    )
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--render-backend", default="gpu")
    parser.add_argument("--shader", default="default")
    parser.add_argument("--max-steps-per-demo", type=int, default=80)
    parser.add_argument("--hold-steps", type=int, default=8)
    parser.add_argument("--gripper-open", type=float, default=0.04)
    parser.add_argument("--keep-failures", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/pg3d_reach_narrow.zarr"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    args.workspace_bounds = np.asarray(args.workspace_bounds, dtype=np.float32).reshape(3, 2)
    args.action_mode = _action_mode(args.action_mode)
    if args.hold_steps < 0:
        raise ValueError("--hold-steps must be non-negative")
    if args.max_steps_per_demo <= 0:
        raise ValueError("--max-steps-per-demo must be positive")
    return args


def _env_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "render_mode": None,
        "robot_uids": args.robot_uid,
        "num_envs": 1,
        "sim_backend": args.sim_backend,
        "render_backend": args.render_backend,
        "sensor_configs": {"shader_pack": args.shader},
    }


def _collect_episode(
    *,
    env: Any,
    seed: int,
    env_id: str,
    action_mode: ActionMode,
    crop_config: PointCloudCropConfig,
    max_steps: int,
    hold_steps: int,
    gripper_open: float,
    sapien: Any,
    planner_cls: Any,
) -> ReachEpisodeData | None:
    obs, info = env.reset(seed=seed, options={"reconfigure": True})
    unwrapped = env.unwrapped
    goal_pose = _goal_pose(unwrapped, sapien)
    planner = planner_cls(
        env,
        debug=False,
        vis=False,
        base_pose=unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )
    try:
        plan = planner.move_to_pose_with_screw(goal_pose, dry_run=True)
    finally:
        planner.close()
    if plan == -1 or "position" not in plan:
        return None

    rows: list[dict[str, np.ndarray]] = []
    successes: list[bool] = []
    distances: list[float] = []
    first_success_step: int | None = None
    pre_hold_final_distance: float | None = None
    hold_steps_recorded = 0
    positions = np.asarray(plan["position"], dtype=np.float32)
    for planned_qpos in positions[:max_steps]:
        sim_action = _format_sim_action(env, planned_qpos)
        row = _dataset_row_from_obs(
            obs=obs,
            info=info,
            env=env,
            env_id=env_id,
            sim_action=sim_action,
            action_mode=action_mode,
            crop_config=crop_config,
        )
        obs, _reward, terminated, truncated, info = env.step(sim_action)
        success = _bool_info(info, "success")
        distance = _float_info(info, "tcp_to_goal_dist", default=_tcp_to_goal_distance(unwrapped))
        row["success"] = np.asarray(success, dtype=bool)
        rows.append(row)
        successes.append(success)
        distances.append(distance)
        if success:
            first_success_step = len(rows)
            pre_hold_final_distance = distance
            break
        if _bool_any(terminated) or _bool_any(truncated):
            break

    if not rows:
        return None

    while (
        first_success_step is not None
        and hold_steps_recorded < hold_steps
        and len(rows) < max_steps
    ):
        sim_action = _hold_sim_action(env, gripper_open=gripper_open)
        row = _dataset_row_from_obs(
            obs=obs,
            info=info,
            env=env,
            env_id=env_id,
            sim_action=sim_action,
            action_mode=action_mode,
            crop_config=crop_config,
        )
        obs, _reward, _terminated, truncated, info = env.step(sim_action)
        success = _bool_info(info, "success")
        distance = _float_info(info, "tcp_to_goal_dist", default=_tcp_to_goal_distance(unwrapped))
        row["success"] = np.asarray(success, dtype=bool)
        rows.append(row)
        successes.append(success)
        distances.append(distance)
        hold_steps_recorded += 1
        if _bool_any(truncated):
            break

    final_distance = _float_info(info, "tcp_to_goal_dist", default=_tcp_to_goal_distance(unwrapped))
    metadata = {
        "seed": seed,
        "length": len(rows),
        "planner_status": str(plan.get("status", "unknown")),
        "first_success_step": first_success_step,
        "hold_steps_requested": hold_steps,
        "hold_steps_recorded": hold_steps_recorded,
        "pre_hold_final_distance": pre_hold_final_distance,
        "final_distance": final_distance,
        "min_distance": float(np.min(distances)) if distances else final_distance,
        "success": bool(successes[-1]) if successes else False,
    }
    return ReachEpisodeData(
        state=np.stack([row["state"] for row in rows], axis=0),
        action=np.stack([row["action"] for row in rows], axis=0),
        sim_action=np.stack([row["sim_action"] for row in rows], axis=0),
        point_cloud=np.stack([row["point_cloud"] for row in rows], axis=0),
        robot_mask=np.stack([row["robot_mask"] for row in rows], axis=0),
        point_valid_mask=np.stack([row["point_valid_mask"] for row in rows], axis=0),
        target_position=np.stack([row["target_position"] for row in rows], axis=0),
        tcp_pose=np.stack([row["tcp_pose"] for row in rows], axis=0),
        success=np.asarray(successes, dtype=bool),
        metadata=metadata,
    )


def _dataset_row_from_obs(
    *,
    obs: Any,
    info: Any,
    env: Any,
    env_id: str,
    sim_action: np.ndarray,
    action_mode: ActionMode,
    crop_config: PointCloudCropConfig,
) -> dict[str, np.ndarray]:
    adapted = adapt_observation(obs, info=info, env=env, task_name=env_id)
    return observation_to_dataset_row(
        adapted,
        sim_action=sim_action,
        action_mode=action_mode,
        crop_config=crop_config,
    )


def _goal_pose(unwrapped_env: Any, sapien: Any) -> Any:
    goal_pos = _to_numpy(unwrapped_env.goal_site.pose.p).reshape(-1, 3)[0]
    tcp_pose = _to_numpy(unwrapped_env.agent.tcp.pose.raw_pose).reshape(-1, 7)[0]
    return sapien.Pose(p=goal_pos, q=tcp_pose[3:])


def _format_sim_action(env: Any, planned_qpos: np.ndarray) -> np.ndarray:
    action_dim = int(np.prod(env.action_space.shape))
    planned_qpos = np.asarray(planned_qpos, dtype=np.float32).reshape(-1)
    if planned_qpos.shape[0] == action_dim:
        return planned_qpos
    if planned_qpos.shape[0] >= 9 and action_dim == 8:
        return np.concatenate([planned_qpos[:7], [np.mean(planned_qpos[7:9])]]).astype(np.float32)
    if planned_qpos.shape[0] == 7 and action_dim == 8:
        return np.concatenate([planned_qpos, [0.04]]).astype(np.float32)
    raise ValueError(
        f"cannot convert planned qpos shape {planned_qpos.shape} to action_dim={action_dim}"
    )


def _hold_sim_action(env: Any, *, gripper_open: float) -> np.ndarray:
    """Return a simulator action that asks Panda to hold the current arm qpos."""
    qpos = _to_numpy(env.unwrapped.agent.robot.qpos).reshape(-1)
    action_dim = int(np.prod(env.action_space.shape))
    if qpos.shape[0] < 7:
        raise ValueError(f"robot qpos must have at least 7 values, got {qpos.shape}")
    if action_dim == 7:
        return qpos[:7].astype(np.float32, copy=True)
    if action_dim == 8:
        return np.concatenate([qpos[:7], [gripper_open]]).astype(np.float32)
    raise ValueError(f"unsupported action_dim={action_dim} for hold action")


def _dataset_stats(episodes: list[ReachEpisodeData]) -> dict[str, Any]:
    lengths = np.asarray([episode.state.shape[0] for episode in episodes], dtype=np.int64)
    final_distances = np.asarray(
        [episode.metadata.get("final_distance", np.nan) for episode in episodes],
        dtype=np.float32,
    )
    action_norms = np.concatenate(
        [np.linalg.norm(episode.action, axis=1).astype(np.float32) for episode in episodes],
        axis=0,
    )
    robot_counts = np.concatenate(
        [episode.robot_mask.sum(axis=1).astype(np.float32) for episode in episodes],
        axis=0,
    )
    valid_counts = np.concatenate(
        [episode.point_valid_mask.sum(axis=1).astype(np.float32) for episode in episodes],
        axis=0,
    )
    hold_requested = np.asarray(
        [episode.metadata.get("hold_steps_requested", 0) for episode in episodes],
        dtype=np.int64,
    )
    hold_recorded = np.asarray(
        [episode.metadata.get("hold_steps_recorded", 0) for episode in episodes],
        dtype=np.int64,
    )
    successes = np.asarray([bool(episode.metadata.get("success", False)) for episode in episodes])
    return {
        "num_episodes": int(len(episodes)),
        "num_steps": int(lengths.sum()) if lengths.size else 0,
        "success_rate": float(successes.mean()) if successes.size else 0.0,
        "episode_length": _summary_stats(lengths),
        "final_distance": _summary_stats(final_distances),
        "action_norm": _summary_stats(action_norms),
        "robot_mask_points": _summary_stats(robot_counts),
        "valid_points": _summary_stats(valid_counts),
        "hold_steps_requested": _summary_stats(hold_requested),
        "hold_steps_recorded": _summary_stats(hold_recorded),
        "hold_coverage": float(
            hold_recorded.sum() / max(int(hold_requested.sum()), 1)
        ),
    }


def _summary_stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"min": 0, "mean": 0.0, "max": 0}
    return {
        "min": float(np.min(finite)),
        "mean": float(np.mean(finite)),
        "max": float(np.max(finite)),
    }


def json_dumps(value: Any) -> str:
    import json

    return json.dumps(_jsonable(value), sort_keys=True)


def _tcp_to_goal_distance(unwrapped_env: Any) -> float:
    goal_pos = _to_numpy(unwrapped_env.goal_site.pose.p).reshape(-1, 3)[0]
    tcp_pos = _to_numpy(unwrapped_env.agent.tcp.pose.p).reshape(-1, 3)[0]
    return float(np.linalg.norm(goal_pos - tcp_pos))


def _bool_info(info: dict[str, Any], key: str) -> bool:
    return bool(np.asarray(_to_numpy(info[key])).reshape(-1)[0]) if key in info else False


def _float_info(info: dict[str, Any], key: str, *, default: float) -> float:
    if key not in info:
        return float(default)
    return float(np.asarray(_to_numpy(info[key])).reshape(-1)[0])


def _bool_any(value: Any) -> bool:
    return bool(np.any(_to_numpy(value)))


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _action_mode(value: str) -> ActionMode:
    if value not in {"abs_joint", "delta_joint"}:
        raise ValueError(f"unsupported action mode {value!r}")
    return value  # type: ignore[return-value]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
