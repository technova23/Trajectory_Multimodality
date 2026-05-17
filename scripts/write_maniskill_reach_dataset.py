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
        "crop": crop_config.to_json(),
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
    parser.add_argument("--keep-failures", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/pg3d_reach_narrow.zarr"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    args.workspace_bounds = np.asarray(args.workspace_bounds, dtype=np.float32).reshape(3, 2)
    args.action_mode = _action_mode(args.action_mode)
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
    positions = np.asarray(plan["position"], dtype=np.float32)
    for planned_qpos in positions[:max_steps]:
        sim_action = _format_sim_action(env, planned_qpos)
        adapted = adapt_observation(obs, info=info, env=env, task_name=env_id)
        row = observation_to_dataset_row(
            adapted,
            sim_action=sim_action,
            action_mode=action_mode,
            crop_config=crop_config,
        )
        obs, _reward, terminated, truncated, info = env.step(sim_action)
        success = _bool_info(info, "success")
        row["success"] = np.asarray(success, dtype=bool)
        rows.append(row)
        successes.append(success)
        if _bool_any(terminated) or _bool_any(truncated):
            break

    if not rows:
        return None
    final_distance = _float_info(info, "tcp_to_goal_dist", default=_tcp_to_goal_distance(unwrapped))
    metadata = {
        "seed": seed,
        "length": len(rows),
        "planner_status": str(plan.get("status", "unknown")),
        "final_distance": final_distance,
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


if __name__ == "__main__":
    raise SystemExit(main())
