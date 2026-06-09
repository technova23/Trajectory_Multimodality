from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pg3d.envs.maniskill_adapter import register_pg3d_reach_envs
from pg3d.envs.maniskill_adapter.dataset import (
    PointCloudCropConfig,
    crop_point_cloud,
    load_reach_metadata,
)
from pg3d.envs.maniskill_adapter.geometry import ManiSkillGhostPandaGeometryProvider
from pg3d.envs.maniskill_adapter.types import Observation, RobotState, SimGroundTruth
from pg3d.policies.dp3.checkpoint import load_reach_policy_from_checkpoint
from pg3d.policies.dp3.goal_markers import insert_goal_marker_points
from pg3d.utils.devices import select_device
from pg3d.utils.serialization import jsonable
from pg3d.world_model import ActionChunk, GeometricWorldModel


@dataclass
class EpisodeSeedObservation:
    episode_index: int
    seed: int | None
    point_cloud: np.ndarray
    robot_mask: np.ndarray
    point_valid_mask: np.ndarray
    policy_point_cloud: np.ndarray
    joint_state: np.ndarray
    tcp_pose: np.ndarray
    goal_position: np.ndarray
    crop_config: PointCloudCropConfig
    action_mode: str
    env_id: str
    env_kwargs: dict[str, Any]


@dataclass
class TreeNode:
    node_id: int
    parent_id: int | None
    root_id: int
    depth: int
    timestamp: float
    observation_point_cloud: np.ndarray
    robot_mask: np.ndarray
    point_valid_mask: np.ndarray
    joint_state: np.ndarray
    tcp_pose: np.ndarray
    goal_position: np.ndarray
    action_sequence: np.ndarray | None = None
    q_trajectory: np.ndarray | None = None
    ee_trajectory: np.ndarray | None = None
    future_point_clouds: list[np.ndarray] = field(default_factory=list)
    future_robot_masks: list[np.ndarray] = field(default_factory=list)
    child_ids: list[int] = field(default_factory=list)


@dataclass
class TreeBuildContext:
    policy: Any
    device: torch.device
    world_model: GeometricWorldModel
    geometry_provider: ManiSkillGhostPandaGeometryProvider
    crop_config: PointCloudCropConfig
    action_mode: str
    hz: float
    replan_step: int
    branching_factor: int
    tree_depth: int
    seed: int
    nodes: list[TreeNode] = field(default_factory=list)

    @property
    def dt(self) -> float:
        return 1.0 / self.hz

    def next_id(self) -> int:
        return len(self.nodes)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "ManiSkill/Gymnasium are required. Install with the repo's maniskill extras."
        ) from exc

    register_pg3d_reach_envs()
    device = select_device(args.device)
    policy = load_reach_policy_from_checkpoint(
        args.checkpoint,
        device=device,
        prefer_ema=args.checkpoint_model == "ema",
    )
    episode = load_episode(args.dataset, episode_index=args.episode_index)
    ghost_env = None
    try:
        ghost_env = gym.make(episode.env_id, **_ghost_env_kwargs(episode.env_kwargs))
        geometry_provider = ManiSkillGhostPandaGeometryProvider(
            ghost_env,
            task_name=episode.env_id,
            max_robot_points=int(np.count_nonzero(episode.robot_mask & episode.point_valid_mask)),
            crop_bounds=episode.crop_config.bounds,
        )
        if episode.seed is not None:
            geometry_provider.reset(seed=int(episode.seed), options={"reconfigure": True})
        geometry_provider.set_robot_point_budget_from_mask(
            episode.robot_mask,
            point_valid_mask=episode.point_valid_mask,
            min_points=1,
        )
        context = TreeBuildContext(
            policy=policy,
            device=device,
            world_model=GeometricWorldModel(geometry_provider),
            geometry_provider=geometry_provider,
            crop_config=episode.crop_config,
            action_mode=episode.action_mode,
            hz=args.hz,
            replan_step=args.replan_step,
            branching_factor=args.branching_factor,
            tree_depth=args.tree_depth,
            seed=args.seed,
        )
        root = _root_node(episode)
        context.nodes.append(root)
        expand_tree_node(
            context,
            root,
            candidate_count=args.root_candidates,
            sample_seed_base=args.seed,
        )
    finally:
        if ghost_env is not None:
            ghost_env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_tree_json(
        args.output.with_suffix(".json"),
        dataset=args.dataset,
        checkpoint=args.checkpoint,
        episode=episode,
        context=context,
    )
    visualize_tree(args.output, args.video, context.nodes, hz=args.hz)
    print(f"saved rerun: {args.output}")
    print(f"saved video: {args.video}")
    print(f"saved json: {args.output.with_suffix('.json')}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic geometry world-model rollout tree from DP3 samples."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-model", choices=["ema", "raw"], default="ema")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--root-candidates", type=int, default=32)
    parser.add_argument("--branching-factor", type=int, default=16)
    parser.add_argument("--tree-depth", type=int, default=3)
    parser.add_argument("--replan-step", type=int, default=8)
    parser.add_argument("--hz", type=float, default=16.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.root_candidates <= 0:
        raise ValueError("--root-candidates must be positive")
    if args.branching_factor <= 0:
        raise ValueError("--branching-factor must be positive")
    if args.tree_depth <= 0:
        raise ValueError("--tree-depth must be positive")
    if args.replan_step <= 0:
        raise ValueError("--replan-step must be positive")
    if args.hz <= 0.0:
        raise ValueError("--hz must be positive")
    return args


def load_episode(dataset_path: Path, *, episode_index: int) -> EpisodeSeedObservation:
    """Load timestep zero of one dataset episode and its metadata."""
    import zarr

    metadata = load_reach_metadata(dataset_path)
    root = zarr.open_group(str(dataset_path), mode="r")
    data = root["data"]
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    if episode_index < 0 or episode_index >= episode_ends.shape[0]:
        raise IndexError(f"episode_index={episode_index} out of range")
    episode_starts = np.concatenate([np.asarray([0], dtype=np.int64), episode_ends[:-1]])
    row = int(episode_starts[episode_index])
    valid = (
        np.asarray(data["point_valid_mask"][row], dtype=bool)
        if "point_valid_mask" in data
        else np.ones(data["point_cloud"].shape[1], dtype=bool)
    )
    robot_mask_full = np.asarray(data["robot_mask"][row], dtype=bool)
    point_cloud_full = np.asarray(data["point_cloud"][row], dtype=np.float32)
    target_key = "target_position" if "target_position" in data else "goal_pos"
    crop = metadata.get("crop", {})
    crop_config = PointCloudCropConfig(
        bounds=np.asarray(crop.get("bounds", [[-0.9, 0.7], [-0.6, 0.6], [0.0, 1.1]])),
        num_points=int(crop.get("num_points", point_cloud_full.shape[0])),
        robot_point_fraction=float(crop.get("robot_point_fraction", 0.25)),
    )
    episode_meta = metadata.get("episodes", [])
    seed = None
    if episode_index < len(episode_meta) and "seed" in episode_meta[episode_index]:
        seed = int(episode_meta[episode_index]["seed"])
    return EpisodeSeedObservation(
        episode_index=episode_index,
        seed=seed,
        point_cloud=point_cloud_full[valid].astype(np.float32, copy=True),
        robot_mask=robot_mask_full.astype(bool, copy=True),
        point_valid_mask=valid.astype(bool, copy=True),
        policy_point_cloud=point_cloud_full.astype(np.float32, copy=True),
        joint_state=np.asarray(data["state"][row], dtype=np.float32),
        tcp_pose=np.asarray(data["tcp_pose"][row], dtype=np.float32),
        goal_position=np.asarray(data[target_key][row], dtype=np.float32),
        crop_config=crop_config,
        action_mode=str(metadata.get("action_mode", "abs_joint")),
        env_id=str(metadata["env_id"]),
        env_kwargs=dict(metadata["env_kwargs"]),
    )


def sample_dp3_candidates(
    policy: Any,
    *,
    point_cloud: np.ndarray,
    joint_state: np.ndarray,
    goal_position: np.ndarray,
    device: torch.device,
    count: int,
    seed_base: int,
) -> list[np.ndarray]:
    """Sample full-horizon DP3 action predictions without executing them."""
    policy.eval()
    point_cloud_window = np.repeat(
        point_cloud.reshape(1, *point_cloud.shape),
        int(policy.n_obs_steps),
        axis=0,
    )
    if int(getattr(policy, "goal_marker_points", 0)) > 0:
        point_cloud_window = insert_goal_marker_points(
            point_cloud_window,
            np.repeat(goal_position.reshape(1, 3), point_cloud_window.shape[0], axis=0),
            num_points=int(policy.goal_marker_points),
            radius=float(policy.goal_marker_radius),
        )
    joint_window = np.repeat(
        joint_state.reshape(1, -1),
        int(policy.n_obs_steps),
        axis=0,
    )
    obs = {
        "point_cloud": torch.from_numpy(point_cloud_window.astype(np.float32))
        .unsqueeze(0)
        .to(device),
        "agent_pos": torch.from_numpy(joint_window.astype(np.float32)).unsqueeze(0).to(device),
    }
    candidates: list[np.ndarray] = []
    for idx in range(count):
        sample_seed = int(seed_base + idx)
        torch.manual_seed(sample_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sample_seed)
        with torch.no_grad():
            output = policy.predict_action(obs)
        action_sequence = output["action_pred"][0].detach().cpu().numpy().astype(np.float32)
        candidates.append(action_sequence)
    return candidates


def rollout_world_model(
    context: TreeBuildContext,
    *,
    node: TreeNode,
    action_sequence: np.ndarray,
    candidate_index: int,
) -> tuple[Any, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Roll one sampled action horizon through the geometry-only world model."""
    observation = Observation(
        point_cloud=node.observation_point_cloud[node.point_valid_mask],
        robot_mask=node.robot_mask[node.point_valid_mask],
        point_features={},
        robot_state=RobotState(
            joint_positions=node.joint_state,
            tcp_pose=node.tcp_pose,
        ),
        sim_gt=SimGroundTruth(task_name="trajectory_tree", target_position=node.goal_position),
    )
    action_chunk = ActionChunk(
        actions=action_sequence,
        action_mode=context.action_mode,  # type: ignore[arg-type]
        dt=context.dt,
        metadata={"candidate_index": int(candidate_index), "node_id": int(node.node_id)},
    )
    rollout = context.world_model.imagine(
        observation,
        action_chunk,
        start_q=node.joint_state,
        metadata={"node_id": int(node.node_id), "candidate_index": int(candidate_index)},
    )
    future_clouds: list[np.ndarray] = []
    future_masks: list[np.ndarray] = []
    future_valid_masks: list[np.ndarray] = []
    for scene, mask in zip(rollout.scene_point_clouds, rollout.robot_masks, strict=True):
        cloud, robot_mask, valid_mask = compose_future_pointcloud(
            scene,
            mask,
            crop_config=context.crop_config,
        )
        future_clouds.append(cloud)
        future_masks.append(robot_mask)
        future_valid_masks.append(valid_mask)
    return rollout, future_clouds, future_masks, future_valid_masks


def compose_future_pointcloud(
    scene_point_cloud: np.ndarray,
    robot_mask: np.ndarray,
    *,
    crop_config: PointCloudCropConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop/pad a future static+robot scene back to the policy point-cloud shape."""
    cropped = crop_point_cloud(
        scene_point_cloud,
        robot_mask=robot_mask,
        config=crop_config,
    )
    return (
        cropped["point_cloud"].astype(np.float32, copy=True),
        np.asarray(cropped["robot_mask"], dtype=bool),
        np.asarray(cropped["point_valid_mask"], dtype=bool),
    )


def expand_tree_node(
    context: TreeBuildContext,
    node: TreeNode,
    *,
    candidate_count: int,
    sample_seed_base: int,
) -> None:
    """Recursively sample DP3 branches, imagine them, and attach child nodes."""
    if node.depth >= context.tree_depth:
        return
    candidates = sample_dp3_candidates(
        context.policy,
        point_cloud=node.observation_point_cloud,
        joint_state=node.joint_state,
        goal_position=node.goal_position,
        device=context.device,
        count=candidate_count,
        seed_base=sample_seed_base,
    )
    for candidate_index, action_sequence in enumerate(candidates):
        rollout, future_clouds, future_masks, future_valid_masks = rollout_world_model(
            context,
            node=node,
            action_sequence=action_sequence,
            candidate_index=candidate_index,
        )
        replan_idx = min(context.replan_step - 1, rollout.q.shape[0] - 1)
        child_id = context.next_id()
        root_id = candidate_index if node.parent_id is None else node.root_id
        child = TreeNode(
            node_id=child_id,
            parent_id=node.node_id,
            root_id=root_id,
            depth=node.depth + 1,
            timestamp=node.timestamp + context.replan_step * context.dt,
            observation_point_cloud=future_clouds[replan_idx],
            robot_mask=future_masks[replan_idx],
            point_valid_mask=future_valid_masks[replan_idx],
            joint_state=rollout.q[replan_idx].astype(np.float32, copy=True),
            tcp_pose=_tcp_pose_from_eef(rollout.eef_path[replan_idx], node.tcp_pose),
            goal_position=node.goal_position,
            action_sequence=action_sequence,
            q_trajectory=rollout.q,
            ee_trajectory=rollout.eef_path,
            future_point_clouds=future_clouds,
            future_robot_masks=future_masks,
        )
        context.nodes.append(child)
        node.child_ids.append(child.node_id)
        if child.depth < context.tree_depth:
            expand_tree_node(
                context,
                child,
                candidate_count=context.branching_factor,
                sample_seed_base=sample_seed_base + 10_000 * (child.node_id + 1),
            )


def visualize_tree(
    output: Path,
    video: Path,
    nodes: list[TreeNode],
    *,
    hz: float,
) -> None:
    """Write Rerun and MP4 visualizations of the imagined trajectory tree."""
    _write_rerun(output, nodes, hz=hz)
    _write_video(video, nodes)


def save_tree_json(
    path: Path,
    *,
    dataset: Path,
    checkpoint: Path,
    episode: EpisodeSeedObservation,
    context: TreeBuildContext,
) -> None:
    """Save topology and trajectory arrays in JSON-safe form."""
    payload = {
        "dataset": str(dataset),
        "checkpoint": str(checkpoint),
        "episode_index": episode.episode_index,
        "episode_seed": episode.seed,
        "hz": context.hz,
        "dt": context.dt,
        "replan_step": context.replan_step,
        "tree_depth": context.tree_depth,
        "branching_factor": context.branching_factor,
        "num_nodes": len(context.nodes),
        "nodes": [_node_to_json(node) for node in context.nodes],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _root_node(episode: EpisodeSeedObservation) -> TreeNode:
    return TreeNode(
        node_id=0,
        parent_id=None,
        root_id=0,
        depth=0,
        timestamp=0.0,
        observation_point_cloud=episode.policy_point_cloud,
        robot_mask=episode.robot_mask,
        point_valid_mask=episode.point_valid_mask,
        joint_state=episode.joint_state,
        tcp_pose=episode.tcp_pose,
        goal_position=episode.goal_position,
    )


def _node_to_json(node: TreeNode) -> dict[str, Any]:
    point_counts = [int(np.count_nonzero(mask)) for mask in node.future_robot_masks]
    return {
        "node_id": node.node_id,
        "parent_id": node.parent_id,
        "root_id": node.root_id,
        "depth": node.depth,
        "timestamp": node.timestamp,
        "child_ids": list(node.child_ids),
        "observation_point_count": int(node.observation_point_cloud.shape[0]),
        "observation_robot_points": int(np.count_nonzero(node.robot_mask)),
        "joint_state": node.joint_state.tolist(),
        "tcp_pose": node.tcp_pose.tolist(),
        "goal_position": node.goal_position.tolist(),
        "action_sequence": None if node.action_sequence is None else node.action_sequence.tolist(),
        "predicted_joint_trajectory": (
            None if node.q_trajectory is None else node.q_trajectory.tolist()
        ),
        "predicted_ee_trajectory": (
            None if node.ee_trajectory is None else node.ee_trajectory.tolist()
        ),
        "future_point_counts": [int(cloud.shape[0]) for cloud in node.future_point_clouds],
        "future_robot_point_counts": point_counts,
    }


def _write_rerun(path: Path, nodes: list[TreeNode], *, hz: float) -> None:
    try:
        import rerun as rr
    except Exception as exc:
        raise RuntimeError("Rerun export requires the repo viz extras") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    rr.init("pg3d_trajectory_tree_world_model", spawn=False)
    rr.save(str(path))
    palette = _palette()
    root = nodes[0]
    rr.set_time_seconds("time", 0.0)
    valid = root.point_valid_mask
    rr.log(
        "world/current/static_points",
        rr.Points3D(root.observation_point_cloud[valid & ~root.robot_mask], colors=[170, 170, 170]),
        static=True,
    )
    rr.log(
        "world/current/robot_points",
        rr.Points3D(root.observation_point_cloud[valid & root.robot_mask], colors=[0, 128, 255]),
        static=True,
    )
    rr.log(
        "world/goal",
        rr.Points3D(root.goal_position.reshape(1, 3), colors=[0, 255, 0], radii=0.018),
        static=True,
    )
    for node in nodes[1:]:
        if node.ee_trajectory is None:
            continue
        color = palette[node.root_id % len(palette)]
        rr.set_time_seconds("time", float(node.timestamp))
        rr.log(
            f"world/tree/root_{node.root_id:02d}/node_{node.node_id:05d}/ee",
            rr.LineStrips3D([node.ee_trajectory.astype(np.float32)], colors=color),
            static=True,
        )
        replan_idx = min(max(0, node.ee_trajectory.shape[0] - 1), node.ee_trajectory.shape[0] - 1)
        if node.future_point_clouds:
            scene = node.future_point_clouds[replan_idx]
            robot = node.future_robot_masks[replan_idx]
            rr.log(
                f"world/tree/root_{node.root_id:02d}/node_{node.node_id:05d}/future_robot",
                rr.Points3D(scene[robot], colors=color, radii=0.003),
                static=True,
            )
    rr.disconnect()


def _write_video(path: Path, nodes: list[TreeNode]) -> None:
    try:
        import imageio.v2 as imageio
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except Exception as exc:
        raise RuntimeError("MP4 export requires matplotlib and imageio") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    palette = np.asarray(_palette(), dtype=np.float32) / 255.0
    root = nodes[0]
    static = root.observation_point_cloud[root.point_valid_mask & ~root.robot_mask]
    robot = root.observation_point_cloud[root.point_valid_mask & root.robot_mask]
    paths = [
        (node.root_id, node.ee_trajectory)
        for node in nodes[1:]
        if node.ee_trajectory is not None and node.ee_trajectory.shape[0] >= 2
    ]
    all_points = [static, robot, root.goal_position.reshape(1, 3)]
    all_points.extend(path for _, path in paths[:512])
    limits = _axis_limits(np.concatenate([pts for pts in all_points if pts.size], axis=0))
    frames = []
    for frame_idx in range(96):
        fig = plt.figure(figsize=(9.0, 7.0), dpi=120)
        ax = fig.add_subplot(111, projection="3d")
        if static.size:
            ax.scatter(static[:, 0], static[:, 1], static[:, 2], s=1.5, c="#b8b8b8", alpha=0.20)
        if robot.size:
            ax.scatter(robot[:, 0], robot[:, 1], robot[:, 2], s=3.0, c="#1681d9", alpha=0.60)
        ax.scatter(*root.goal_position.tolist(), s=90, c="#00b050")
        for root_id, path_points in paths:
            color = palette[root_id % len(palette)]
            ax.plot(
                path_points[:, 0],
                path_points[:, 1],
                path_points[:, 2],
                color=color,
                alpha=0.28,
                linewidth=1.2,
            )
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        ax.set_zlim(*limits[2])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=24, azim=35 + 360.0 * frame_idx / 96)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(fig)
    imageio.mimsave(path, frames, fps=24)


def _ghost_env_kwargs(env_kwargs: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(env_kwargs)
    kwargs["obs_mode"] = "pointcloud"
    kwargs["render_mode"] = None
    kwargs["num_envs"] = 1
    kwargs.setdefault("robot_uids", "panda")
    kwargs.setdefault("control_mode", "pd_joint_pos")
    return kwargs


def _tcp_pose_from_eef(eef_position: np.ndarray, previous_tcp_pose: np.ndarray) -> np.ndarray:
    tcp = np.asarray(previous_tcp_pose, dtype=np.float32).reshape(-1).copy()
    if tcp.shape[0] < 7:
        padded = np.zeros((7,), dtype=np.float32)
        padded[: min(tcp.shape[0], 7)] = tcp[:7]
        tcp = padded
    tcp[:3] = np.asarray(eef_position, dtype=np.float32).reshape(3)
    return tcp.astype(np.float32, copy=False)


def _axis_limits(points: np.ndarray) -> list[tuple[float, float]]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    mins = np.nanmin(points, axis=0)
    maxs = np.nanmax(points, axis=0)
    center = (mins + maxs) * 0.5
    span = float(np.max(maxs - mins))
    half = max(0.15, span * 0.60)
    return [(float(value - half), float(value + half)) for value in center]


def _palette() -> list[tuple[int, int, int]]:
    return [
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


if __name__ == "__main__":
    raise SystemExit(main())
