from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path
from typing import Any, Literal

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
from pg3d.envs.maniskill_adapter.reach_config import REACH_TASK_SPECS, reach_task_metadata
from pg3d.utils.arrays import (
    bool_any as _bool_any,
)
from pg3d.utils.arrays import (
    bool_info as _bool_info,
)
from pg3d.utils.arrays import (
    float_info as _float_info,
)
from pg3d.utils.arrays import (
    to_numpy as _to_numpy,
)
from pg3d.utils.serialization import jsonable as _jsonable


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
        robot_point_fraction=args.robot_point_fraction,
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
            new_episodes = _collect_multimodal_episodes(
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
                variants_per_reset=args.trajectory_variants_per_reset,
                waypoint_attempts=args.waypoint_attempts,
                min_base_clearance=args.min_base_clearance,
                table_margin=args.table_margin,
                waypoint_xy_noise=args.waypoint_xy_noise,
                waypoint_z_noise=args.waypoint_z_noise,
                lateral_z_offset=args.lateral_z_offset,
                vertical_lateral_offset=args.vertical_lateral_offset,
                randomize_start=args.randomize_start,
                start_bounds=_start_workspace_bounds(args.env_id, args.start_bounds),
                start_sample_attempts=args.start_sample_attempts,
                min_start_goal_distance=args.min_start_goal_distance,
                require_complete_variant_set=not args.allow_partial_variant_sets,
                suppress_planner_output=not args.show_planner_output,
                viewer_step_delay=args.viewer_step_delay if args.viewer else 0.0,
            )
            if not new_episodes:
                skipped.append({"seed": seed, "reason": "planner_failed_or_empty"})
                continue
            for episode in new_episodes:
                if len(episodes) >= args.num_demos:
                    break
                if not args.keep_failures and not bool(episode.metadata.get("success", False)):
                    skipped.append(
                        {
                            "seed": seed,
                            "reason": "unsuccessful_replay",
                            "trajectory_family": episode.metadata.get("trajectory_family"),
                            "final_distance": episode.metadata.get("final_distance"),
                        }
                    )
                    continue
                episodes.append(episode)
                print(
                    "demo "
                    f"{len(episodes)}/{args.num_demos}: seed={seed} "
                    f"variant={episode.metadata.get('trajectory_family')} "
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
            _hold_viewer(env, args.viewer_hold_seconds if args.viewer else 0.0)
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
        "start_sampling": {
            "randomize_start": args.randomize_start,
            "start_bounds": _start_workspace_bounds(args.env_id, args.start_bounds).tolist(),
            "start_sample_attempts": args.start_sample_attempts,
            "min_start_goal_distance": args.min_start_goal_distance,
            "min_base_clearance": args.min_base_clearance,
            "table_margin": args.table_margin,
            "note": (
                "Starts are sampled as Cartesian TCP poses, accepted only when the ManiSkill "
                "Panda motion planner can reach them from the reset configuration. The script "
                "also rejects starts and goals that violate the configured base clearance or "
                "XY table-margin inset."
            ),
        },
        "trajectory_generation": {
            "type": "multimodal_waypoint_planning",
            "variants_per_reset": args.trajectory_variants_per_reset,
            "waypoint_attempts": args.waypoint_attempts,
            "min_base_clearance": args.min_base_clearance,
            "table_margin": args.table_margin,
            "waypoint_xy_noise": args.waypoint_xy_noise,
            "waypoint_z_noise": args.waypoint_z_noise,
            "lateral_z_offset": args.lateral_z_offset,
            "vertical_lateral_offset": args.vertical_lateral_offset,
            "allow_partial_variant_sets": args.allow_partial_variant_sets,
            "show_planner_output": args.show_planner_output,
            "planner": "PandaArmMotionPlanningSolver.move_to_pose_with_screw",
            "note": (
                "Only trajectory generation is changed; env, observations, crop, "
                "and zarr writer are unchanged."
            ),
        },
        "task": reach_task_metadata(args.env_id),
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
    alias_arrays = _ensure_goal_observation_aliases(args.output)
    summary.get("arrays", {}).update(alias_arrays)
    print(f"saved dataset: {args.output}")
    print(f"summary: {summary}")
    print("dataset_stats: " + json_dumps(dataset_stats))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a smoke-scale pg3d ManiSkill reach dataset."
    )
    parser.add_argument("--env-id", default="PG3DReach-BalancedWorkspace-v0")
    parser.add_argument("--num-demos", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=25)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--obs-mode", default="pointcloud", choices=["pointcloud"])
    parser.add_argument("--action-mode", default="abs_joint", choices=["abs_joint", "delta_joint"])
    parser.add_argument("--control-mode", default="pd_joint_pos")
    parser.add_argument("--robot-uid", default="panda")
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument(
        "--robot-point-fraction",
        type=float,
        default=0.25,
        help=(
            "minimum fraction of saved point-cloud slots reserved for robot-mask points "
            "when enough robot points are available"
        ),
    )
    parser.add_argument(
        "--workspace-bounds",
        type=float,
        nargs=6,
        default=DEFAULT_WORKSPACE_BOUNDS.reshape(-1).tolist(),
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"),
    )
    parser.add_argument(
        "--start-bounds",
        type=float,
        nargs=6,
        default=None,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"),
        help=(
            "Cartesian TCP start sampling bounds. Defaults to the selected task's goal bounds, "
            "so broad tasks scatter both starts and goals across the table workspace."
        ),
    )
    parser.add_argument(
        "--randomize-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="sample a reachable TCP start pose before generating variants",
    )
    parser.add_argument(
        "--start-sample-attempts",
        type=int,
        default=30,
        help="maximum reachable start pose samples to try per reset",
    )
    parser.add_argument(
        "--min-start-goal-distance",
        type=float,
        default=0.12,
        help="minimum Euclidean distance between sampled TCP start and goal in meters",
    )
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--render-backend", default="gpu")
    parser.add_argument("--shader", default="default")
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="open the ManiSkill human viewer and render frames during collection",
    )
    parser.add_argument(
        "--viewer-step-delay",
        type=float,
        default=0.0,
        help="seconds to sleep after each viewer frame; useful when watching collection live",
    )
    parser.add_argument(
        "--viewer-hold-seconds",
        type=float,
        default=0.0,
        help="seconds to keep the viewer open before closing the environment",
    )
    parser.add_argument("--max-steps-per-demo", type=int, default=100)
    parser.add_argument("--hold-steps", type=int, default=8)
    parser.add_argument("--gripper-open", type=float, default=0.04)
    parser.add_argument(
        "--trajectory-variants-per-reset",
        type=int,
        default=4,
        help=(
            "number of waypoint-conditioned trajectory variants to try for each identical "
            "environment reset/start/goal"
        ),
    )
    parser.add_argument(
        "--allow-partial-variant-sets",
        action="store_true",
        help=(
            "keep successful variants from a seed even if one requested trajectory family fails; "
            "by default, incomplete seed/start groups are skipped so datasets do not silently "
            "miss e.g. downward_arc for a seed"
        ),
    )
    parser.add_argument(
        "--show-planner-output",
        action="store_true",
        help=(
            "show ManiSkill planner stdout/stderr during expected retry failures; by default "
            "the writer suppresses repeated messages such as 'screw plan failed'"
        ),
    )
    parser.add_argument(
        "--waypoint-attempts",
        type=int,
        default=40,
        help="maximum waypoint samples to try per trajectory family",
    )
    parser.add_argument(
        "--min-base-clearance",
        type=float,
        default=0.10,
        help=(
            "minimum horizontal XY distance from the robot base for sampled starts, goals, "
            "and waypoints, in meters"
        ),
    )
    parser.add_argument(
        "--table-margin",
        type=float,
        default=0.10,
        help=(
            "XY margin inset from the task/workspace bounds for sampled starts, goals, "
            "and waypoints, in meters"
        ),
    )
    parser.add_argument(
        "--waypoint-xy-noise",
        type=float,
        default=0.04,
        help="half-width of uniform XY perturbation added to sampled waypoints, in meters",
    )
    parser.add_argument(
        "--waypoint-z-noise",
        type=float,
        default=0.025,
        help="half-width of uniform Z perturbation added to sampled waypoints, in meters",
    )
    parser.add_argument(
        "--lateral-z-offset",
        type=float,
        default=0.15,
        help="maximum extra absolute Z offset for lateral curve waypoints, in meters",
    )
    parser.add_argument(
        "--vertical-lateral-offset",
        type=float,
        default=0.10,
        help="maximum lateral XY offset for upward/downward arc waypoints, in meters",
    )
    parser.add_argument("--keep-failures", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/pg3d_reach_balanced.zarr"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    args.workspace_bounds = np.asarray(args.workspace_bounds, dtype=np.float32).reshape(3, 2)
    if args.start_bounds is not None:
        args.start_bounds = np.asarray(args.start_bounds, dtype=np.float32).reshape(3, 2)
    args.action_mode = _action_mode(args.action_mode)
    if args.hold_steps < 0:
        raise ValueError("--hold-steps must be non-negative")
    if args.max_steps_per_demo <= 0:
        raise ValueError("--max-steps-per-demo must be positive")
    if args.num_points <= 0:
        raise ValueError("--num-points must be positive")
    if not 0.0 <= args.robot_point_fraction <= 1.0:
        raise ValueError("--robot-point-fraction must be between 0 and 1")
    if args.trajectory_variants_per_reset <= 0:
        raise ValueError("--trajectory-variants-per-reset must be positive")
    if args.waypoint_attempts <= 0:
        raise ValueError("--waypoint-attempts must be positive")
    if args.min_base_clearance < 0:
        raise ValueError("--min-base-clearance must be non-negative")
    if args.table_margin < 0:
        raise ValueError("--table-margin must be non-negative")
    if args.waypoint_xy_noise < 0:
        raise ValueError("--waypoint-xy-noise must be non-negative")
    if args.waypoint_z_noise < 0:
        raise ValueError("--waypoint-z-noise must be non-negative")
    if args.lateral_z_offset < 0:
        raise ValueError("--lateral-z-offset must be non-negative")
    if args.vertical_lateral_offset < 0:
        raise ValueError("--vertical-lateral-offset must be non-negative")
    if args.start_sample_attempts <= 0:
        raise ValueError("--start-sample-attempts must be positive")
    if args.min_start_goal_distance < 0:
        raise ValueError("--min-start-goal-distance must be non-negative")
    if args.viewer_step_delay < 0:
        raise ValueError("--viewer-step-delay must be non-negative")
    if args.viewer_hold_seconds < 0:
        raise ValueError("--viewer-hold-seconds must be non-negative")
    return args


def _env_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "render_mode": "human" if args.viewer else None,
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
    viewer_step_delay: float = 0.0,
    suppress_planner_output: bool = True,
) -> ReachEpisodeData | None:
    obs, info = env.reset(seed=seed, options={"reconfigure": True})
    _render_viewer_frame(env, viewer_step_delay)
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
        plan = _move_to_pose_with_screw(
            planner,
            goal_pose,
            suppress_output=suppress_planner_output,
        )
    finally:
        planner.close()
    if plan == -1 or "position" not in plan:
        return None

    return _replay_planned_positions_as_episode(
        env=env,
        env_id=env_id,
        action_mode=action_mode,
        crop_config=crop_config,
        max_steps=max_steps,
        hold_steps=hold_steps,
        gripper_open=gripper_open,
        obs=obs,
        info=info,
        positions=np.asarray(plan["position"], dtype=np.float32),
        viewer_step_delay=viewer_step_delay,
        metadata={
            "seed": seed,
            "planner_status": str(plan.get("status", "unknown")),
            "trajectory_family": "direct",
            "trajectory_type": -1,
            "trajectory_waypoints": [],
        },
    )


def _collect_multimodal_episodes(
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
    variants_per_reset: int,
    waypoint_attempts: int,
    min_base_clearance: float,
    table_margin: float,
    waypoint_xy_noise: float,
    waypoint_z_noise: float,
    lateral_z_offset: float,
    vertical_lateral_offset: float,
    randomize_start: bool,
    start_bounds: np.ndarray,
    start_sample_attempts: int,
    min_start_goal_distance: float,
    require_complete_variant_set: bool,
    suppress_planner_output: bool,
    viewer_step_delay: float = 0.0,
) -> list[ReachEpisodeData]:
    obs, info = env.reset(seed=seed, options={"reconfigure": True})
    _render_viewer_frame(env, viewer_step_delay)
    unwrapped = env.unwrapped
    goal_pose = _goal_pose(unwrapped, sapien)
    reset_tcp_pose = _tcp_pose(unwrapped)
    reset_qpos = _get_robot_qpos(env)
    rng = np.random.default_rng(seed)
    robot_base_position = _robot_base_position(unwrapped)
    waypoint_bounds = _inset_xy_bounds(
        _waypoint_workspace_bounds(env_id, crop_config),
        table_margin,
    )
    start_sampling_bounds = _inset_xy_bounds(start_bounds, table_margin)
    if waypoint_bounds is None or start_sampling_bounds is None:
        return []
    goal_xyz = np.asarray(goal_pose.p, dtype=np.float64).reshape(-1, 3)[0]
    if not _is_waypoint_valid(
        waypoint=goal_xyz,
        workspace_bounds=waypoint_bounds,
        robot_base_position=robot_base_position,
        min_base_clearance=min_base_clearance,
    ):
        return []

    planner = planner_cls(
        env,
        debug=False,
        vis=False,
        base_pose=unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )
    try:
        start_sample = _sample_reachable_start(
            env=env,
            planner=planner,
            sapien=sapien,
            rng=rng,
            reset_qpos=reset_qpos,
            reset_tcp_pose=reset_tcp_pose,
            goal_pose=goal_pose,
            start_bounds=start_sampling_bounds,
            randomize_start=randomize_start,
            max_attempts=start_sample_attempts,
            min_start_goal_distance=min_start_goal_distance,
            min_base_clearance=min_base_clearance,
            suppress_planner_output=suppress_planner_output,
        )
        if start_sample is None:
            return []
        start_qpos, start_tcp_pose, start_metadata = start_sample
        _set_robot_qpos(env, start_qpos)
        _set_start_site_pose(env, start_tcp_pose[:3])
        variants = generate_multimodal_waypoints(
            current_tcp_pose=start_tcp_pose,
            goal_pose=goal_pose,
            workspace_bounds=waypoint_bounds,
            robot_base_position=robot_base_position,
            planner=planner,
            env=env,
            sapien=sapien,
            rng=rng,
            variants_per_reset=variants_per_reset,
            max_attempts=waypoint_attempts,
            min_base_clearance=min_base_clearance,
            waypoint_xy_noise=waypoint_xy_noise,
            waypoint_z_noise=waypoint_z_noise,
            lateral_z_offset=lateral_z_offset,
            vertical_lateral_offset=vertical_lateral_offset,
            start_qpos=start_qpos,
            suppress_planner_output=suppress_planner_output,
        )
    finally:
        planner.close()

    if require_complete_variant_set and not _has_complete_variant_set(
        variants,
        variants_per_reset=variants_per_reset,
    ):
        return []

    episodes: list[ReachEpisodeData] = []
    for variant in variants:
        obs, info = env.reset(seed=seed, options={"reconfigure": True})
        _set_robot_qpos(env, start_qpos)
        _set_start_site_pose(env, start_tcp_pose[:3])
        obs, info = _refresh_obs_after_manual_qpos(
            env,
            info=info,
            gripper_open=gripper_open,
        )
        _render_viewer_frame(env, viewer_step_delay)
        episode = _replay_planned_positions_as_episode(
            env=env,
            env_id=env_id,
            action_mode=action_mode,
            crop_config=crop_config,
            max_steps=max_steps,
            hold_steps=hold_steps,
            gripper_open=gripper_open,
            obs=obs,
            info=info,
            positions=variant["positions"],
            viewer_step_delay=viewer_step_delay,
            metadata={
                "seed": seed,
                "planner_status": variant["planner_status"],
                "trajectory_family": variant["name"],
                "trajectory_type": variant["trajectory_type"],
                "trajectory_waypoints": variant["waypoints"],
                "trajectory_waypoint_metadata": variant["waypoint_metadata"],
                "start_tcp_pose": start_tcp_pose.astype(np.float32).tolist(),
                "goal_pose": _pose_to_list(goal_pose),
                "start_sampling": start_metadata,
            },
        )
        if episode is not None:
            episodes.append(episode)
    return episodes


def _replay_planned_positions_as_episode(
    *,
    env: Any,
    env_id: str,
    action_mode: ActionMode,
    crop_config: PointCloudCropConfig,
    max_steps: int,
    hold_steps: int,
    gripper_open: float,
    obs: Any,
    info: Any,
    positions: np.ndarray,
    metadata: dict[str, Any],
    viewer_step_delay: float = 0.0,
) -> ReachEpisodeData | None:
    unwrapped = env.unwrapped
    rows: list[dict[str, np.ndarray]] = []
    successes: list[bool] = []
    distances: list[float] = []
    first_success_step: int | None = None
    pre_hold_final_distance: float | None = None
    hold_steps_recorded = 0
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
        _render_viewer_frame(env, viewer_step_delay)
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
        _render_viewer_frame(env, viewer_step_delay)
        success = _bool_info(info, "success")
        distance = _float_info(info, "tcp_to_goal_dist", default=_tcp_to_goal_distance(unwrapped))
        row["success"] = np.asarray(success, dtype=bool)
        rows.append(row)
        successes.append(success)
        distances.append(distance)
        hold_steps_recorded += 1
        if _bool_any(truncated):
            break

    final_distance = _float_info(
        info,
        "tcp_to_goal_dist",
        default=_tcp_to_goal_distance(unwrapped),
    )
    metadata = {
        **metadata,
        "length": len(rows),
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


WaypointMode = Literal["lateral", "wide_curve", "upward_arc", "downward_arc"]


def generate_multimodal_waypoints(
    *,
    current_tcp_pose: np.ndarray,
    goal_pose: Any,
    workspace_bounds: np.ndarray,
    robot_base_position: np.ndarray,
    planner: Any,
    env: Any,
    sapien: Any,
    rng: np.random.Generator,
    variants_per_reset: int,
    max_attempts: int,
    min_base_clearance: float,
    waypoint_xy_noise: float,
    waypoint_z_noise: float,
    lateral_z_offset: float,
    vertical_lateral_offset: float,
    start_qpos: np.ndarray,
    suppress_planner_output: bool = True,
) -> list[dict[str, Any]]:
    """Sample waypoint-conditioned variants while keeping the official planner in charge."""
    start = np.asarray(current_tcp_pose[:3], dtype=np.float64)
    goal = np.asarray(goal_pose.p, dtype=np.float64).reshape(-1, 3)[0]
    goal_quat = np.asarray(goal_pose.q, dtype=np.float64).reshape(4)
    delta = goal - start
    distance = float(np.linalg.norm(delta))
    if distance < 1e-6:
        return []

    specs = _trajectory_variant_specs(variants_per_reset)
    variants: list[dict[str, Any]] = []
    for trajectory_type, name, mode, lateral_sign in specs:
        for _ in range(max_attempts):
            waypoint, waypoint_metadata = _sample_waypoint(
                start=start,
                goal=goal,
                mode=mode,
                lateral_sign=lateral_sign,
                workspace_bounds=workspace_bounds,
                robot_base_position=robot_base_position,
                min_base_clearance=min_base_clearance,
                xy_noise=waypoint_xy_noise,
                z_noise=waypoint_z_noise,
                lateral_z_offset=lateral_z_offset,
                vertical_lateral_offset=vertical_lateral_offset,
                rng=rng,
            )
            if waypoint is None:
                continue

            waypoint_pose = sapien.Pose(
                p=waypoint.astype(np.float32),
                q=goal_quat.astype(np.float32),
            )
            plan_result = _plan_multisegment_trajectory(
                planner=planner,
                env=env,
                poses=[waypoint_pose, goal_pose],
                start_qpos=start_qpos,
                suppress_planner_output=suppress_planner_output,
            )
            if plan_result is None:
                continue
            positions, planner_status = plan_result
            variants.append(
                {
                    "name": name,
                    "trajectory_type": trajectory_type,
                    "positions": positions,
                    "planner_status": planner_status,
                    "waypoints": [waypoint.astype(np.float32).tolist()],
                    "waypoint_metadata": [waypoint_metadata],
                }
            )
            break
    _set_robot_qpos(env, start_qpos)
    return variants


def _sample_reachable_start(
    *,
    env: Any,
    planner: Any,
    sapien: Any,
    rng: np.random.Generator,
    reset_qpos: np.ndarray,
    reset_tcp_pose: np.ndarray,
    goal_pose: Any,
    start_bounds: np.ndarray,
    randomize_start: bool,
    max_attempts: int,
    min_start_goal_distance: float,
    min_base_clearance: float,
    suppress_planner_output: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    if not randomize_start:
        return (
            reset_qpos,
            reset_tcp_pose,
            {
                "randomized": False,
                "sampled_position": reset_tcp_pose[:3].astype(np.float32).tolist(),
                "actual_position": reset_tcp_pose[:3].astype(np.float32).tolist(),
                "attempt": 0,
            },
        )

    goal_xyz = np.asarray(goal_pose.p, dtype=np.float64).reshape(-1, 3)[0]
    reset_quat = np.asarray(reset_tcp_pose[3:7], dtype=np.float32)
    bounds = np.asarray(start_bounds, dtype=np.float64).reshape(3, 2)
    robot_base_position = _robot_base_position(env.unwrapped)
    for attempt in range(1, max_attempts + 1):
        sampled_xyz = rng.uniform(bounds[:, 0], bounds[:, 1]).astype(np.float64)
        if not _is_waypoint_valid(
            waypoint=sampled_xyz,
            workspace_bounds=bounds,
            robot_base_position=robot_base_position,
            min_base_clearance=min_base_clearance,
        ):
            continue
        if float(np.linalg.norm(sampled_xyz - goal_xyz)) < min_start_goal_distance:
            continue
        sampled_pose = sapien.Pose(p=sampled_xyz.astype(np.float32), q=reset_quat)
        plan = _plan_to_pose(
            planner=planner,
            env=env,
            pose=sampled_pose,
            start_qpos=reset_qpos,
            suppress_planner_output=suppress_planner_output,
        )
        if plan is None:
            continue
        start_qpos, planner_status = plan
        _set_robot_qpos(env, start_qpos)
        actual_tcp_pose = _tcp_pose(env.unwrapped)
        _set_robot_qpos(env, reset_qpos)
        if float(np.linalg.norm(actual_tcp_pose[:3] - goal_xyz)) < min_start_goal_distance:
            continue
        if not _is_waypoint_valid(
            waypoint=actual_tcp_pose[:3],
            workspace_bounds=bounds,
            robot_base_position=robot_base_position,
            min_base_clearance=min_base_clearance,
        ):
            continue
        return (
            start_qpos,
            actual_tcp_pose,
            {
                "randomized": True,
                "attempt": attempt,
                "sampled_position": sampled_xyz.astype(np.float32).tolist(),
                "actual_position": actual_tcp_pose[:3].astype(np.float32).tolist(),
                "planner_status": planner_status,
                "horizontal_base_clearance": float(
                    np.linalg.norm(actual_tcp_pose[:2] - robot_base_position[:2])
                ),
                "distance_to_goal": float(np.linalg.norm(actual_tcp_pose[:3] - goal_xyz)),
            },
        )
    _set_robot_qpos(env, reset_qpos)
    return None


def _trajectory_variant_specs(
    variants_per_reset: int,
) -> list[tuple[int, str, WaypointMode, float | None]]:
    base_specs: list[tuple[int, str, WaypointMode, float | None]] = [
        (0, "lateral_left", "lateral", 1.0),
        (1, "lateral_right", "lateral", -1.0),
        (2, "upward_arc", "upward_arc", None),
        (3, "downward_arc", "downward_arc", None),
        (4, "wide_curve", "wide_curve", None),
    ]
    if variants_per_reset <= len(base_specs):
        return base_specs[:variants_per_reset]
    specs = list(base_specs)
    for idx in range(len(base_specs), variants_per_reset):
        specs.append((idx, f"wide_curve_{idx - len(base_specs) + 2}", "wide_curve", None))
    return specs


def _has_complete_variant_set(variants: list[dict[str, Any]], *, variants_per_reset: int) -> bool:
    expected_types = {spec[0] for spec in _trajectory_variant_specs(variants_per_reset)}
    actual_types = {int(variant["trajectory_type"]) for variant in variants}
    return expected_types.issubset(actual_types)


def _sample_waypoint(
    *,
    start: np.ndarray,
    goal: np.ndarray,
    mode: WaypointMode,
    lateral_sign: float | None,
    workspace_bounds: np.ndarray,
    robot_base_position: np.ndarray,
    min_base_clearance: float,
    xy_noise: float,
    z_noise: float,
    lateral_z_offset: float,
    vertical_lateral_offset: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]] | tuple[None, dict[str, Any]]:
    delta = goal - start
    horizontal = delta[:2]
    horizontal_norm = float(np.linalg.norm(horizontal))
    if horizontal_norm < 1e-6:
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        perp = np.asarray([np.cos(angle), np.sin(angle), 0.0], dtype=np.float64)
    else:
        perp = np.asarray([-horizontal[1], horizontal[0], 0.0], dtype=np.float64)
        perp /= np.linalg.norm(perp[:2])

    path_ratio = float(rng.uniform(0.15, 0.85))
    base_point = start + path_ratio * delta
    waypoint = base_point.copy()
    offset = np.zeros(3, dtype=np.float64)
    if mode == "lateral":
        sign = float(1.0 if lateral_sign is None else lateral_sign)
        lateral_mag = float(rng.uniform(0.05, 0.25))
        offset = sign * lateral_mag * perp
        offset[2] = float(rng.uniform(-lateral_z_offset, lateral_z_offset))
    elif mode == "wide_curve":
        sign = float(rng.choice(np.asarray([-1.0, 1.0], dtype=np.float64)))
        lateral_mag = float(rng.uniform(0.18, 0.35))
        offset = sign * lateral_mag * perp
        offset[2] = float(rng.uniform(-lateral_z_offset, lateral_z_offset))
    elif mode == "upward_arc":
        sign = float(rng.choice(np.asarray([-1.0, 1.0], dtype=np.float64)))
        lateral_mag = float(rng.uniform(0.0, vertical_lateral_offset))
        offset = sign * lateral_mag * perp
        offset[2] = float(rng.uniform(0.10, 0.35))
    elif mode == "downward_arc":
        sign = float(rng.choice(np.asarray([-1.0, 1.0], dtype=np.float64)))
        lateral_mag = float(rng.uniform(0.0, vertical_lateral_offset))
        offset = sign * lateral_mag * perp
        offset[2] = -float(rng.uniform(0.05, 0.20))
    else:
        raise ValueError(f"unsupported waypoint mode {mode!r}")

    perturbation = np.asarray(
        [
            rng.uniform(-xy_noise, xy_noise),
            rng.uniform(-xy_noise, xy_noise),
            rng.uniform(-z_noise, z_noise),
        ],
        dtype=np.float64,
    )
    waypoint = waypoint + offset + perturbation
    if not _is_waypoint_valid(
        waypoint=waypoint,
        workspace_bounds=workspace_bounds,
        robot_base_position=robot_base_position,
        min_base_clearance=min_base_clearance,
    ):
        return None, {}
    metadata = {
        "mode": mode,
        "path_ratio": path_ratio,
        "offset": offset.astype(np.float32).tolist(),
        "perturbation": perturbation.astype(np.float32).tolist(),
        "xy_noise": float(xy_noise),
        "z_noise": float(z_noise),
        "lateral_z_offset": float(lateral_z_offset),
        "vertical_lateral_offset": float(vertical_lateral_offset),
    }
    return waypoint, metadata


def _inset_xy_bounds(bounds: np.ndarray, margin: float) -> np.ndarray | None:
    inset = np.asarray(bounds, dtype=np.float64).reshape(3, 2).copy()
    inset[:2, 0] += margin
    inset[:2, 1] -= margin
    if np.any(inset[:, 0] > inset[:, 1]):
        return None
    return inset


def _is_waypoint_valid(
    *,
    waypoint: np.ndarray,
    workspace_bounds: np.ndarray,
    robot_base_position: np.ndarray,
    min_base_clearance: float,
) -> bool:
    if not np.all(np.isfinite(waypoint)):
        return False
    bounds = np.asarray(workspace_bounds, dtype=np.float64).reshape(3, 2)
    if not np.all((waypoint >= bounds[:, 0]) & (waypoint <= bounds[:, 1])):
        return False
    horizontal_clearance = float(
        np.linalg.norm(waypoint[:2] - np.asarray(robot_base_position[:2], dtype=np.float64))
    )
    return horizontal_clearance >= min_base_clearance


def _plan_multisegment_trajectory(
    *,
    planner: Any,
    env: Any,
    poses: list[Any],
    start_qpos: np.ndarray,
    suppress_planner_output: bool = True,
) -> tuple[np.ndarray, str] | None:
    _set_robot_qpos(env, start_qpos)
    segments: list[np.ndarray] = []
    statuses: list[str] = []
    try:
        for pose in poses:
            plan = _move_to_pose_with_screw(
                planner,
                pose,
                suppress_output=suppress_planner_output,
            )
            if not _is_valid_plan(plan):
                return None
            positions = np.asarray(plan["position"], dtype=np.float32)
            if segments and np.allclose(segments[-1][-1], positions[0], atol=1e-5):
                positions = positions[1:]
            if positions.size == 0:
                return None
            segments.append(positions)
            statuses.append(str(plan.get("status", "unknown")))
            _set_robot_qpos(env, positions[-1])
    finally:
        _set_robot_qpos(env, start_qpos)
    if not segments:
        return None
    return np.concatenate(segments, axis=0).astype(np.float32), "+".join(statuses)


def _plan_to_pose(
    *,
    planner: Any,
    env: Any,
    pose: Any,
    start_qpos: np.ndarray,
    suppress_planner_output: bool = True,
) -> tuple[np.ndarray, str] | None:
    _set_robot_qpos(env, start_qpos)
    try:
        plan = _move_to_pose_with_screw(
            planner,
            pose,
            suppress_output=suppress_planner_output,
        )
        if not _is_valid_plan(plan):
            return None
        positions = np.asarray(plan["position"], dtype=np.float32)
        return positions[-1].astype(np.float32, copy=True), str(plan.get("status", "unknown"))
    finally:
        _set_robot_qpos(env, start_qpos)


def _is_valid_plan(plan: Any) -> bool:
    if plan == -1 or not isinstance(plan, dict) or "position" not in plan:
        return False
    positions = np.asarray(plan["position"], dtype=np.float32)
    if positions.ndim != 2 or positions.shape[0] == 0:
        return False
    if not np.all(np.isfinite(positions)):
        return False
    status = str(plan.get("status", "")).lower()
    return "fail" not in status and "error" not in status


def _move_to_pose_with_screw(
    planner: Any,
    pose: Any,
    *,
    suppress_output: bool = True,
) -> Any:
    if not suppress_output:
        return planner.move_to_pose_with_screw(pose, dry_run=True)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return planner.move_to_pose_with_screw(pose, dry_run=True)


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


def _render_viewer_frame(env: Any, delay_seconds: float = 0.0) -> None:
    if not _is_human_render_env(env):
        return
    env.render()
    if delay_seconds > 0.0:
        import time

        time.sleep(delay_seconds)


def _hold_viewer(env: Any, seconds: float) -> None:
    if seconds <= 0.0 or not _is_human_render_env(env):
        return
    import time

    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        env.render()
        time.sleep(1.0 / 30.0)


def _is_human_render_env(env: Any) -> bool:
    return getattr(env, "render_mode", None) == "human" or (
        hasattr(env, "unwrapped") and getattr(env.unwrapped, "render_mode", None) == "human"
    )


def _refresh_obs_after_manual_qpos(
    env: Any,
    *,
    info: Any,
    gripper_open: float,
) -> tuple[Any, Any]:
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "get_obs"):
        return unwrapped.get_obs(info), info
    if hasattr(env, "get_obs"):
        return env.get_obs(info), info
    action = _hold_sim_action(env, gripper_open=gripper_open)
    obs, _reward, _terminated, _truncated, info = env.step(action)
    return obs, info


def _set_start_site_pose(env: Any, position: np.ndarray) -> None:
    start_site = getattr(env.unwrapped, "start_site", None)
    if start_site is None:
        return
    from mani_skill.utils.structs.pose import Pose

    start_site.set_pose(Pose.create_from_pq(np.asarray(position, dtype=np.float32).reshape(1, 3)))


def _goal_pose(unwrapped_env: Any, sapien: Any) -> Any:
    goal_pos = _to_numpy(unwrapped_env.goal_site.pose.p).reshape(-1, 3)[0]
    tcp_pose = _to_numpy(unwrapped_env.agent.tcp.pose.raw_pose).reshape(-1, 7)[0]
    return sapien.Pose(p=goal_pos, q=tcp_pose[3:])


def _tcp_pose(unwrapped_env: Any) -> np.ndarray:
    tcp = getattr(unwrapped_env.agent, "tcp_pose", None)
    if tcp is not None and hasattr(tcp, "raw_pose"):
        return _to_numpy(tcp.raw_pose).reshape(-1, 7)[0].astype(np.float32)
    return _to_numpy(unwrapped_env.agent.tcp.pose.raw_pose).reshape(-1, 7)[0].astype(np.float32)


def _robot_base_position(unwrapped_env: Any) -> np.ndarray:
    pose = getattr(unwrapped_env.agent.robot, "pose", None)
    if pose is None or not hasattr(pose, "p"):
        return np.zeros(3, dtype=np.float32)
    return _to_numpy(pose.p).reshape(-1, 3)[0].astype(np.float32)


def _waypoint_workspace_bounds(env_id: str, crop_config: PointCloudCropConfig) -> np.ndarray:
    task = reach_task_metadata(env_id)
    goal_bounds = task.get("goal_bounds")
    if goal_bounds is not None:
        return np.asarray(goal_bounds, dtype=np.float32).reshape(3, 2)
    return np.asarray(crop_config.bounds, dtype=np.float32).reshape(3, 2)


def _start_workspace_bounds(env_id: str, start_bounds: np.ndarray | None) -> np.ndarray:
    if start_bounds is not None:
        return np.asarray(start_bounds, dtype=np.float32).reshape(3, 2)
    task = reach_task_metadata(env_id)
    goal_bounds = task.get("goal_bounds")
    if goal_bounds is not None:
        return np.asarray(goal_bounds, dtype=np.float32).reshape(3, 2)
    return np.asarray(REACH_TASK_SPECS["PG3DReach-BalancedWorkspace-v0"].goal_bounds).reshape(3, 2)


def _pose_to_list(pose: Any) -> list[float]:
    return (
        np.concatenate(
            [
                np.asarray(pose.p, dtype=np.float32).reshape(-1)[:3],
                np.asarray(pose.q, dtype=np.float32).reshape(-1)[:4],
            ],
            axis=0,
        )
        .astype(float)
        .tolist()
    )


def _get_robot_qpos(env: Any) -> np.ndarray:
    robot = env.unwrapped.agent.robot
    if hasattr(robot, "get_qpos"):
        qpos = _to_numpy(robot.get_qpos())
    else:
        qpos = _to_numpy(robot.qpos)
    return qpos.reshape(-1).astype(np.float32, copy=True)


def _set_robot_qpos(env: Any, qpos: np.ndarray) -> None:
    robot = env.unwrapped.agent.robot
    qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
    if hasattr(robot, "get_qpos"):
        current = _to_numpy(robot.get_qpos()).astype(np.float32, copy=True)
    else:
        current = _to_numpy(robot.qpos).astype(np.float32, copy=True)
    current_shape = current.shape
    current = current.reshape(-1)
    if qpos.shape[0] > current.shape[0]:
        raise ValueError(
            f"planned qpos has {qpos.shape[0]} values, robot qpos has {current.shape[0]}"
        )
    next_qpos = current.copy()
    next_qpos[: qpos.shape[0]] = qpos
    next_qpos = next_qpos.reshape(current_shape)
    if hasattr(robot, "set_qpos"):
        robot.set_qpos(next_qpos)
    else:
        robot.qpos = next_qpos.reshape(-1)
    if hasattr(robot, "set_qvel"):
        robot.set_qvel(np.zeros_like(next_qpos, dtype=np.float32))


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


def _ensure_goal_observation_aliases(dataset_path: Path) -> dict[str, dict[str, Any]]:
    """Add compatibility aliases requested by multimodal reach training configs."""
    import zarr

    root = zarr.open_group(str(dataset_path), mode="a")
    data = root["data"]
    goal_pos = np.asarray(data["target_position"][:], dtype=np.float32)
    tcp_pose = np.asarray(data["tcp_pose"][:], dtype=np.float32)
    eef_pos = tcp_pose[:, :3].astype(np.float32, copy=True)
    arrays = {
        "goal_pos": goal_pos,
        "goal_relative": (goal_pos - eef_pos).astype(np.float32, copy=False),
        "eef_pos": eef_pos,
    }
    summaries: dict[str, dict[str, Any]] = {}
    for key, value in arrays.items():
        if key in data:
            del data[key]
        chunks = (min(max(1, value.shape[0]), 1024),) + value.shape[1:]
        data.array(name=key, data=value, chunks=chunks)
        summaries[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    return summaries


def _tcp_to_goal_distance(unwrapped_env: Any) -> float:
    goal_pos = _to_numpy(unwrapped_env.goal_site.pose.p).reshape(-1, 3)[0]
    tcp_pos = _to_numpy(unwrapped_env.agent.tcp.pose.p).reshape(-1, 3)[0]
    return float(np.linalg.norm(goal_pos - tcp_pos))


def _action_mode(value: str) -> ActionMode:
    if value not in {"abs_joint", "delta_joint"}:
        raise ValueError(f"unsupported action mode {value!r}")
    return value  # type: ignore[return-value]


if __name__ == "__main__":
    raise SystemExit(main())
