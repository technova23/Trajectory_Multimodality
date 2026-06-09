from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pg3d.composition.scoring import (  # noqa: E402
    consensus_deviations,
    goal_distance as _goal_distance,
    primary_constraint_penalty,
    trajectory_smoothness,
)
from pg3d.constraints.core import SceneContext  # noqa: E402
from pg3d.envs.maniskill_adapter import register_pg3d_reach_envs  # noqa: E402
from pg3d.envs.maniskill_adapter.dataset import load_reach_metadata  # noqa: E402
from pg3d.envs.maniskill_adapter.geometry import (  # noqa: E402
    ManiSkillGhostPandaGeometryProvider,
)
from pg3d.envs.maniskill_adapter.types import Observation, RobotState, SimGroundTruth  # noqa: E402
from pg3d.eval import AvoidOverlayConfig, direct_path_avoid_region  # noqa: E402
from pg3d.policies.dp3.checkpoint import load_reach_policy_from_checkpoint  # noqa: E402
from pg3d.utils.devices import select_device  # noqa: E402
from pg3d.utils.serialization import jsonable as _jsonable  # noqa: E402
from pg3d.viz.constraints import avoid_region_line_visuals  # noqa: E402
from pg3d.world_model import ActionChunk, GeometricWorldModel  # noqa: E402
from scripts.rollout_dp3_reach_policy import (  # noqa: E402
    _action_mode,
    crop_config_from_metadata,
    make_initial_obs_window,
    obs_window_to_torch,
    rollout_observation_entry,
)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        import rerun as rr
    except Exception as exc:
        print(
            f"Failed to import visualization stack: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print(
            "Install with: "
            "uv sync --extra cu129 --extra maniskill --extra viz --group dev --group notebooks",
            file=sys.stderr,
        )
        return 2

    register_pg3d_reach_envs()
    device = select_device(args.device)
    policy = load_reach_policy_from_checkpoint(
        args.checkpoint,
        device=device,
        prefer_ema=args.checkpoint_model == "ema",
    )

    metadata = load_reach_metadata(args.dataset)
    episode_seeds = [
        int(episode["seed"]) for episode in metadata.get("episodes", []) if "seed" in episode
    ]
    if args.episode_index < 0 or args.episode_index >= len(episode_seeds):
        raise IndexError(
            f"--episode-index {args.episode_index} is outside dataset episode range "
            f"[0, {len(episode_seeds) - 1}]"
        )
    rollout_seed = episode_seeds[args.episode_index]
    crop_config = crop_config_from_metadata(metadata)
    action_mode = _action_mode(str(metadata.get("action_mode", "abs_joint")))
    env_kwargs = dict(metadata["env_kwargs"])
    env_kwargs["obs_mode"] = "pointcloud"
    env_kwargs["render_mode"] = "rgb_array"
    env_kwargs["num_envs"] = 1

    args.rerun.parent.mkdir(parents=True, exist_ok=True)
    if args.video is not None:
        args.video.parent.mkdir(parents=True, exist_ok=True)

    env: Any | None = None
    ghost_env: Any | None = None
    try:
        env = gym.make(str(metadata["env_id"]), **env_kwargs)
        ghost_env = gym.make(str(metadata["env_id"]), **env_kwargs)
        obs, info = env.reset(seed=rollout_seed, options={"reconfigure": True})
        initial_entry = rollout_observation_entry(obs, info, env=env, crop_config=crop_config)

        provider = ManiSkillGhostPandaGeometryProvider(
            ghost_env,
            task_name=str(metadata["env_id"]),
            crop_bounds=crop_config.bounds,
        )
        provider.reset(seed=rollout_seed, options={"reconfigure": True})
        provider.set_robot_point_budget_from_mask(
            np.asarray(initial_entry["robot_mask"], dtype=bool),
            point_valid_mask=np.asarray(initial_entry["point_valid_mask"], dtype=bool),
        )
        world_model = GeometricWorldModel(provider)

        observation = _world_model_observation(
            initial_entry=initial_entry,
            task_name=str(metadata["env_id"]),
        )
        start_tcp = np.asarray(initial_entry["tcp_pose"], dtype=np.float32).reshape(-1)[:3]
        target = np.asarray(initial_entry["target_position"], dtype=np.float32).reshape(3)
        constraint = direct_path_avoid_region(
            start_tcp=start_tcp,
            target_position=target,
            config=AvoidOverlayConfig(
                radius=args.avoid_radius,
                min_radius=args.avoid_min_radius,
                margin=args.avoid_margin,
                weight=1.0,
            ),
        )
        scene = SceneContext(
            target_position=target,
            regions={constraint.name: constraint.region},
            metadata={"rollout_seed": rollout_seed},
        )
        action_chunks = _sample_action_chunks(
            policy=policy,
            initial_entry=initial_entry,
            device=device,
            action_mode=action_mode,
            candidates=args.candidates,
            seed=args.seed,
            dt=args.dt,
        )
        candidates = _score_world_model_candidates(
            world_model=world_model,
            observation=observation,
            action_chunks=action_chunks,
            constraint=constraint,
            scene=scene,
            args=args,
        )
    finally:
        if env is not None:
            env.close()
        if ghost_env is not None:
            ghost_env.close()

    selected = _select_candidate(candidates)
    print(
        "world-model sphere-cost selection: "
        f"episode_index={args.episode_index} rollout_seed={rollout_seed} "
        f"selected={selected['candidate_index']} feasible={selected['feasible']} "
        f"score={selected['total_score']:.5f} "
        f"constraint_penalty={selected['constraint_penalty']:.5f} "
        f"goal_distance={selected['goal_distance']:.5f}",
        flush=True,
    )
    _write_rerun(
        rr=rr,
        output=args.rerun,
        initial_entry=initial_entry,
        constraint=constraint,
        candidates=candidates,
        selected_index=int(selected["candidate_index"]),
    )
    if args.video is not None:
        _write_video(
            output=args.video,
            initial_entry=initial_entry,
            constraint=constraint,
            candidates=candidates,
            selected_index=int(selected["candidate_index"]),
            fps=args.video_fps,
        )

    summary = {
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint),
        "episode_index": args.episode_index,
        "rollout_seed": rollout_seed,
        "candidate_count": len(candidates),
        "selection": selected,
        "score_weights": {
            "constraint": args.constraint_weight,
            "goal": args.goal_weight,
            "smoothness": args.smoothness_weight,
            "consensus": args.consensus_weight,
        },
        "candidates": candidates,
        "avoid_region": _constraint_summary(constraint),
        "rerun": str(args.rerun),
        "video": str(args.video) if args.video is not None else None,
    }
    summary_path = args.summary or args.rerun.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"saved rerun: {args.rerun}")
    if args.video is not None:
        print(f"saved video: {args.video}")
    print(f"saved summary: {summary_path}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample DP3 action chunks, imagine them with the geometric world model, "
            "score them with a virtual avoid sphere, and visualize the selected rollout."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-model", choices=["ema", "raw"], default="ema")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.125)
    parser.add_argument("--avoid-radius", type=float, default=0.08)
    parser.add_argument("--avoid-min-radius", type=float, default=0.08)
    parser.add_argument("--avoid-margin", type=float, default=0.0)
    parser.add_argument("--constraint-weight", type=float, default=10.0)
    parser.add_argument("--goal-weight", type=float, default=1.0)
    parser.add_argument("--smoothness-weight", type=float, default=0.01)
    parser.add_argument("--consensus-weight", type=float, default=0.0)
    parser.add_argument(
        "--rerun",
        type=Path,
        default=Path("artifacts/world_model_cost_rollouts/sphere_cost_selection.rrd"),
    )
    parser.add_argument(
        "--video",
        type=str,
        default="artifacts/world_model_cost_rollouts/sphere_cost_selection.mp4",
        help="MP4 output path; pass none to disable",
    )
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--video-fps", type=int, default=10)
    args = parser.parse_args(argv)
    args.video = None if args.video.lower() in {"", "none", "null", "off"} else Path(args.video)
    if args.candidates <= 0:
        raise ValueError("--candidates must be positive")
    if args.dt <= 0.0:
        raise ValueError("--dt must be positive")
    if args.avoid_radius <= 0.0 or args.avoid_min_radius <= 0.0:
        raise ValueError("avoid radii must be positive")
    if args.avoid_margin < 0.0:
        raise ValueError("--avoid-margin must be non-negative")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive")
    return args


def _world_model_observation(*, initial_entry: dict[str, Any], task_name: str) -> Observation:
    valid = np.asarray(initial_entry["point_valid_mask"], dtype=bool)
    points = np.asarray(initial_entry["point_cloud"], dtype=np.float32)[valid]
    robot_mask = np.asarray(initial_entry["robot_mask"], dtype=bool)[valid]
    return Observation(
        point_cloud=points,
        point_features={},
        robot_mask=robot_mask,
        robot_state=RobotState(
            joint_positions=np.asarray(initial_entry["agent_pos"], dtype=np.float32).reshape(-1),
            tcp_pose=np.asarray(initial_entry["tcp_pose"], dtype=np.float32).reshape(-1),
        ),
        sim_gt=SimGroundTruth(
            task_name=task_name,
            target_position=np.asarray(initial_entry["target_position"], dtype=np.float32).reshape(3),
            success=bool(initial_entry.get("success", False)),
        ),
        metadata={"source": "dataset_reset_observation"},
    )


def _sample_action_chunks(
    *,
    policy: Any,
    initial_entry: dict[str, Any],
    device: torch.device,
    action_mode: str,
    candidates: int,
    seed: int,
    dt: float,
) -> list[ActionChunk]:
    obs_window = make_initial_obs_window(initial_entry, n_obs_steps=int(policy.n_obs_steps))
    chunks: list[ActionChunk] = []
    for idx in range(candidates):
        sample_seed = int(seed + idx)
        torch.manual_seed(sample_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sample_seed)
        with torch.no_grad():
            policy_input = obs_window_to_torch(
                obs_window,
                device=device,
                goal_marker_points=int(policy.goal_marker_points),
                goal_marker_radius=float(policy.goal_marker_radius),
            )
            actions = policy.predict_action(policy_input)["action"][0].detach().cpu().numpy()
        chunks.append(
            ActionChunk(
                actions=np.asarray(actions, dtype=np.float32),
                action_mode=action_mode,
                dt=dt,
                metadata={"candidate_index": idx, "sample_seed": sample_seed},
            )
        )
    return chunks


def _score_world_model_candidates(
    *,
    world_model: GeometricWorldModel,
    observation: Observation,
    action_chunks: list[ActionChunk],
    constraint: Any,
    scene: SceneContext,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    consensus = consensus_deviations(action_chunks)
    candidates: list[dict[str, Any]] = []
    for idx, chunk in enumerate(action_chunks):
        rollout = world_model.imagine(
            observation,
            chunk,
            metadata={
                "candidate_index": idx,
                "sample_seed": chunk.metadata.get("sample_seed"),
            },
        )
        costs = constraint.cost(rollout, scene)
        feasible = bool(constraint.satisfied(rollout, scene))
        constraint_penalty = primary_constraint_penalty(costs)
        distance = _goal_distance(rollout, scene.target_position)
        smoothness = trajectory_smoothness(rollout, order=2)
        total_score = (
            float(args.constraint_weight) * constraint_penalty
            + float(args.goal_weight) * (0.0 if distance is None else distance)
            + float(args.smoothness_weight) * smoothness
            + float(args.consensus_weight) * consensus[idx]
        )
        candidates.append(
            {
                "candidate_index": idx,
                "sample_seed": chunk.metadata.get("sample_seed"),
                "feasible": feasible,
                "total_score": float(total_score),
                "constraint_penalty": float(constraint_penalty),
                "goal_distance": float(0.0 if distance is None else distance),
                "smoothness": float(smoothness),
                "consensus_deviation": float(consensus[idx]),
                "constraint_costs": costs,
                "eef_path": rollout.eef_path,
                "q": rollout.q,
                "action_horizon": int(chunk.horizon),
                "action_dim": int(chunk.action_dim),
            }
        )
    return candidates


def _select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    feasible = [candidate for candidate in candidates if bool(candidate["feasible"])]
    pool = feasible or candidates
    return min(pool, key=lambda candidate: float(candidate["total_score"]))


def _write_rerun(
    *,
    rr: Any,
    output: Path,
    initial_entry: dict[str, Any],
    constraint: Any,
    candidates: list[dict[str, Any]],
    selected_index: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rr.init("pg3d_world_model_cost_rollouts", spawn=False)
    rr.save(str(output))
    rr.set_time_sequence("step", 0)

    points = _scene_points(initial_entry)
    if points.size:
        rr.log("world/point_cloud", rr.Points3D(points, colors=[150, 150, 150], radii=0.003))

    target = np.asarray(initial_entry["target_position"], dtype=np.float32).reshape(1, 3)
    start = np.asarray(initial_entry["tcp_pose"], dtype=np.float32).reshape(-1)[:3].reshape(1, 3)
    rr.log("world/goal", rr.Points3D(target, colors=[0, 255, 0], radii=0.018))
    rr.log("world/start_tcp", rr.Points3D(start, colors=[255, 220, 0], radii=0.014))
    for visual in avoid_region_line_visuals([constraint]):
        rr.log(
            f"world/constraints/{visual.name}",
            rr.LineStrips3D(visual.line_strips, colors=visual.color),
            static=True,
        )

    palette = _palette(len(candidates))
    for candidate in candidates:
        idx = int(candidate["candidate_index"])
        path = np.asarray(candidate["eef_path"], dtype=np.float32)
        if path.shape[0] < 2:
            continue
        color = (255, 40, 20) if idx == selected_index else palette[idx % len(palette)]
        prefix = "selected" if idx == selected_index else "candidate"
        rr.log(
            f"world/world_model/{prefix}_{idx:02d}",
            rr.LineStrips3D([path], colors=color),
            static=True,
        )
        rr.log(
            f"world/world_model/end_{idx:02d}",
            rr.Points3D(path[-1:].astype(np.float32), colors=color, radii=0.011),
            static=True,
        )
    rr.disconnect()


def _write_video(
    *,
    output: Path,
    initial_entry: dict[str, Any],
    constraint: Any,
    candidates: list[dict[str, Any]],
    selected_index: int,
    fps: int,
) -> None:
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    points = _scene_points(initial_entry)
    paths = [np.asarray(candidate["eef_path"], dtype=np.float32) for candidate in candidates]
    target = np.asarray(initial_entry["target_position"], dtype=np.float32).reshape(3)
    start = np.asarray(initial_entry["tcp_pose"], dtype=np.float32).reshape(-1)[:3]
    center = np.asarray(constraint.region.center, dtype=np.float32).reshape(3)
    radius = float(constraint.region.radius)
    bounds = _plot_bounds(paths=paths, start=start, target=target, center=center, radius=radius)
    palette = plt.cm.tab20(np.linspace(0.0, 1.0, max(len(paths), 1)))
    frames = []
    frame_count = max(24, fps * 4)
    for frame_idx in range(frame_count):
        fig = plt.figure(figsize=(8.0, 6.0), dpi=140)
        ax = fig.add_subplot(111, projection="3d")
        if points.size:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, c="#9ca3af", alpha=0.14)
        _plot_sphere_wire(ax, center=center, radius=radius, color="#ff4010")
        ax.scatter([start[0]], [start[1]], [start[2]], c="gold", s=55, edgecolors="black")
        ax.scatter([target[0]], [target[1]], [target[2]], c="limegreen", s=70, edgecolors="black")
        for idx, path in enumerate(paths):
            if path.shape[0] < 2:
                continue
            if idx == selected_index:
                ax.plot(path[:, 0], path[:, 1], path[:, 2], color="#ff1f12", linewidth=3.4)
                ax.scatter(path[-1:, 0], path[-1:, 1], path[-1:, 2], color="#ff1f12", s=36)
            else:
                color = palette[idx % len(palette)]
                ax.plot(path[:, 0], path[:, 1], path[:, 2], color=color, linewidth=1.5, alpha=0.55)
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
        ax.set_zlim(bounds[2])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("World-model action chunks scored by avoid-sphere cost")
        ax.view_init(elev=24, azim=-70 + 360.0 * frame_idx / frame_count)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(_canvas_rgb_array(fig.canvas))
        plt.close(fig)
    imageio.mimsave(output, frames, fps=fps, macro_block_size=16)


def _scene_points(initial_entry: dict[str, Any]) -> np.ndarray:
    valid = np.asarray(initial_entry["point_valid_mask"], dtype=bool)
    points = np.asarray(initial_entry["point_cloud"], dtype=np.float32)[valid]
    robot_mask = np.asarray(initial_entry["robot_mask"], dtype=bool)[valid]
    return points[~robot_mask]


def _plot_bounds(
    *,
    paths: list[np.ndarray],
    start: np.ndarray,
    target: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    sphere_extents = np.asarray(
        [
            center + np.asarray([radius, 0.0, 0.0], dtype=np.float32),
            center - np.asarray([radius, 0.0, 0.0], dtype=np.float32),
            center + np.asarray([0.0, radius, 0.0], dtype=np.float32),
            center - np.asarray([0.0, radius, 0.0], dtype=np.float32),
            center + np.asarray([0.0, 0.0, radius], dtype=np.float32),
            center - np.asarray([0.0, 0.0, radius], dtype=np.float32),
        ],
        dtype=np.float32,
    )
    points = np.concatenate(
        [*paths, start.reshape(1, 3), target.reshape(1, 3), sphere_extents],
        axis=0,
    )
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    mid = (mins + maxs) * 0.5
    span = max(float(np.max(maxs - mins)) * 1.15, 0.16)
    half = span * 0.5
    return (
        (float(mid[0] - half), float(mid[0] + half)),
        (float(mid[1] - half), float(mid[1] + half)),
        (float(mid[2] - half), float(mid[2] + half)),
    )


def _palette(count: int) -> list[tuple[int, int, int]]:
    base = [
        (70, 130, 255),
        (255, 150, 40),
        (80, 210, 120),
        (210, 90, 255),
        (255, 90, 120),
        (80, 220, 220),
        (235, 205, 70),
        (130, 110, 255),
        (255, 105, 210),
        (105, 190, 90),
        (245, 120, 80),
        (120, 220, 170),
    ]
    return [base[idx % len(base)] for idx in range(max(count, 1))]


def _canvas_rgb_array(canvas: Any) -> np.ndarray:
    width, height = canvas.get_width_height()
    if hasattr(canvas, "buffer_rgba"):
        rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
        return rgba[:, :, :3].copy()
    if hasattr(canvas, "tostring_rgb"):
        return np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
    if hasattr(canvas, "tostring_argb"):
        argb = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8).reshape(height, width, 4)
        return argb[:, :, [1, 2, 3]].copy()
    raise AttributeError("Matplotlib canvas cannot export RGB pixels")


def _plot_sphere_wire(ax: Any, *, center: np.ndarray, radius: float, color: str) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 80)
    circles = [
        np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1),
        np.stack([np.cos(theta), np.zeros_like(theta), np.sin(theta)], axis=1),
        np.stack([np.zeros_like(theta), np.cos(theta), np.sin(theta)], axis=1),
    ]
    for circle in circles:
        pts = center.reshape(1, 3) + radius * circle
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=1.2, alpha=0.9)


def _constraint_summary(constraint: Any) -> dict[str, Any]:
    region = constraint.region
    return {
        "name": constraint.name,
        "center": np.asarray(region.center, dtype=np.float32).tolist(),
        "radius": float(region.radius),
        "margin": float(constraint.margin),
    }


if __name__ == "__main__":
    raise SystemExit(main())
