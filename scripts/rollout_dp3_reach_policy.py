from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from pg3d.envs.maniskill_adapter import adapt_observation, register_pg3d_reach_envs
from pg3d.envs.maniskill_adapter.dataset import (
    DEFAULT_WORKSPACE_BOUNDS,
    PointCloudCropConfig,
    crop_point_cloud,
    load_reach_metadata,
)
from pg3d.policies.dp3 import SimpleDP3
from pg3d.policies.dp3.normalizer import LinearNormalizer

Source = Literal["dataset", "fresh"]
ActionMode = Literal["abs_joint", "delta_joint"]


@dataclass(frozen=True)
class RolloutSpec:
    """One policy rollout seed and optional source episode."""

    output_index: int
    seed: int
    source: Source
    dataset_episode_index: int | None = None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
    except Exception as exc:
        print(
            f"Failed to import ManiSkill/Gymnasium: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print(
            "Install with: "
            "uv sync --extra cu129 --extra maniskill --extra viz --group dev --group notebooks",
            file=sys.stderr,
        )
        return 2

    register_pg3d_reach_envs()
    device = _select_device(args.device)
    policy = load_reach_policy(args.checkpoint, device=device)
    metadata = load_reach_metadata(args.dataset)
    dataset_episode_seeds = [
        int(episode["seed"]) for episode in metadata.get("episodes", []) if "seed" in episode
    ]
    specs = select_rollout_specs(
        source=args.source,
        dataset_episode_seeds=dataset_episode_seeds,
        episodes=args.episodes,
        episode_indices=args.episode_indices,
        seed_start=args.seed_start,
    )
    if not specs:
        raise RuntimeError("no rollout episodes selected")

    crop_config = crop_config_from_metadata(metadata)
    action_mode = _action_mode(str(metadata.get("action_mode", "abs_joint")))
    env_kwargs = dict(metadata["env_kwargs"])
    env_kwargs["render_mode"] = "rgb_array"
    env_kwargs.setdefault("obs_mode", "pointcloud")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env: Any | None = None
    summaries: list[dict[str, Any]] = []
    metrics_path = args.output_dir / "metrics.jsonl"
    try:
        env = gym.make(str(metadata["env_id"]), **env_kwargs)
        with metrics_path.open("w", encoding="utf-8") as metrics_file:
            for spec in specs:
                summary = run_policy_rollout(
                    env=env,
                    policy=policy,
                    spec=spec,
                    action_mode=action_mode,
                    crop_config=crop_config,
                    output_dir=args.output_dir,
                    device=device,
                    max_steps=args.max_steps,
                    replan_stride=(
                        args.replan_stride
                        if args.replan_stride is not None
                        else int(policy.n_action_steps)
                    ),
                    gripper_open=args.gripper_open,
                    video_fps=args.video_fps,
                    metrics_file=metrics_file,
                )
                summaries.append(summary)
                print(
                    f"episode={spec.output_index} seed={spec.seed} "
                    f"success={summary['success']} final_distance={summary['final_distance']:.4f} "
                    f"steps={summary['steps']}"
                )
    except Exception as exc:
        print(f"Failed to roll out DP3 reach policy: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if env is not None:
            env.close()

    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "source": args.source,
        "env_id": metadata["env_id"],
        "env_kwargs": env_kwargs,
        "action_mode": action_mode,
        "episodes": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    failures = sum(0 if episode["success"] else 1 for episode in summaries)
    return 0 if args.allow_failure or failures == 0 else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roll out a trained pg3d-native DP3 reach policy in ManiSkill."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("artifacts/reach-dataset-smoke/pg3d-reach-smoke.zarr"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reach-policy-rollouts"),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--source", choices=["dataset", "fresh"], default="dataset")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--episode-indices", type=int, nargs="+", default=None)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--replan-stride", type=int, default=None)
    parser.add_argument("--gripper-open", type=float, default=0.04)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args(argv)
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.replan_stride is not None and args.replan_stride <= 0:
        raise ValueError("--replan-stride must be positive")
    return args


def run_policy_rollout(
    *,
    env: Any,
    policy: SimpleDP3,
    spec: RolloutSpec,
    action_mode: ActionMode,
    crop_config: PointCloudCropConfig,
    output_dir: Path,
    device: torch.device,
    max_steps: int,
    replan_stride: int,
    gripper_open: float,
    video_fps: int,
    metrics_file: Any,
) -> dict[str, Any]:
    obs, info = env.reset(seed=spec.seed, options={"reconfigure": True})
    frames = [_frame_to_numpy(env.render())]
    timeline: list[dict[str, np.ndarray | bool | float]] = []
    first_entry = rollout_observation_entry(obs, info, env=env, crop_config=crop_config)
    obs_window = make_initial_obs_window(first_entry, n_obs_steps=int(policy.n_obs_steps))
    timeline.append(first_entry)
    steps = 0
    success = False
    final_distance = float("nan")
    action_norms: list[float] = []

    while steps < max_steps:
        with torch.no_grad():
            policy_input = obs_window_to_torch(obs_window, device=device)
            policy_output = policy.predict_action(policy_input)
            action_chunk = policy_output["action"][0].detach().cpu().numpy()

        for policy_action in action_chunk[:replan_stride]:
            current_state = obs_window[-1]["agent_pos"]
            sim_action = policy_action_to_sim_action(
                policy_action,
                current_state,
                action_mode=action_mode,
                sim_action_dim=int(np.prod(env.action_space.shape)),
                low=getattr(env.action_space, "low", None),
                high=getattr(env.action_space, "high", None),
                gripper_open=gripper_open,
            )
            obs, reward, terminated, truncated, info = env.step(sim_action)
            steps += 1
            frames.append(_frame_to_numpy(env.render()))
            entry = rollout_observation_entry(obs, info, env=env, crop_config=crop_config)
            obs_window = append_obs_window(obs_window, entry, n_obs_steps=int(policy.n_obs_steps))
            timeline.append(entry)
            success = _bool_info(info, "success")
            final_distance = _float_info(info, "tcp_to_goal_dist", default=float("nan"))
            action_norm = float(np.linalg.norm(sim_action))
            action_norms.append(action_norm)
            metrics_file.write(
                json.dumps(
                    _jsonable(
                        {
                            "episode": spec.output_index,
                            "seed": spec.seed,
                            "step": steps,
                            "reward": _float_value(reward),
                            "success": success,
                            "final_distance": final_distance,
                            "action_norm": action_norm,
                        }
                    ),
                    sort_keys=True,
                )
                + "\n"
            )
            metrics_file.flush()
            if success or _bool_any(terminated) or _bool_any(truncated) or steps >= max_steps:
                break
        if success or steps >= max_steps:
            break

    video_path = output_dir / f"episode_{spec.output_index:03d}.mp4"
    rerun_path = output_dir / f"episode_{spec.output_index:03d}.rrd"
    save_video(video_path, frames, fps=video_fps)
    save_rerun_timeline(rerun_path, timeline)
    return {
        "episode": spec.output_index,
        "seed": spec.seed,
        "source": spec.source,
        "dataset_episode_index": spec.dataset_episode_index,
        "steps": steps,
        "success": success,
        "final_distance": final_distance,
        "mean_action_norm": float(np.mean(action_norms)) if action_norms else 0.0,
        "video": str(video_path),
        "rerun": str(rerun_path),
    }


def load_reach_policy(path: Path, *, device: torch.device) -> SimpleDP3:
    """Load a reach policy checkpoint written by `scripts/train_dp3_reach.py`."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    policy = SimpleDP3(**checkpoint["policy_kwargs"])
    policy.set_normalizer(LinearNormalizer.from_state_dict(checkpoint["normalizer"]))
    policy.load_state_dict(checkpoint["model"], strict=False)
    policy.to(device)
    policy.eval()
    return policy


def crop_config_from_metadata(metadata: dict[str, Any]) -> PointCloudCropConfig:
    crop = metadata.get("crop", {})
    bounds = np.asarray(crop.get("bounds", DEFAULT_WORKSPACE_BOUNDS), dtype=np.float32)
    num_points = int(crop.get("num_points", 512))
    return PointCloudCropConfig(bounds=bounds, num_points=num_points)


def rollout_observation_entry(
    obs: Any,
    info: Any,
    *,
    env: Any,
    crop_config: PointCloudCropConfig,
) -> dict[str, np.ndarray | bool | float]:
    adapted = adapt_observation(obs, info=info, env=env, task_name=_env_task_name(env))
    cropped = crop_point_cloud(
        adapted.point_cloud,
        robot_mask=adapted.robot_mask,
        config=crop_config,
    )
    target_position = (
        np.zeros((3,), dtype=np.float32)
        if adapted.sim_gt is None or adapted.sim_gt.target_position is None
        else adapted.sim_gt.target_position.astype(np.float32, copy=True)
    )
    tcp_pose = (
        np.zeros((7,), dtype=np.float32)
        if adapted.robot_state.tcp_pose is None
        else adapted.robot_state.tcp_pose.astype(np.float32, copy=True)
    )
    return {
        "point_cloud": cropped["point_cloud"],
        "robot_mask": cropped["robot_mask"],
        "point_valid_mask": cropped["point_valid_mask"],
        "agent_pos": adapted.robot_state.as_agent_pos(),
        "target_position": target_position,
        "tcp_pose": tcp_pose,
        "success": bool(adapted.sim_gt.success) if adapted.sim_gt is not None else False,
        "final_distance": _float_info(info, "tcp_to_goal_dist", default=float("nan")),
    }


def make_initial_obs_window(
    entry: dict[str, np.ndarray | bool | float],
    *,
    n_obs_steps: int,
) -> list[dict[str, np.ndarray | bool | float]]:
    """Pad the initial policy observation window by repeating the first observation."""
    if n_obs_steps <= 0:
        raise ValueError("n_obs_steps must be positive")
    return [_copy_entry(entry) for _ in range(n_obs_steps)]


def append_obs_window(
    window: list[dict[str, np.ndarray | bool | float]],
    entry: dict[str, np.ndarray | bool | float],
    *,
    n_obs_steps: int,
) -> list[dict[str, np.ndarray | bool | float]]:
    """Append one observation and keep the most recent policy-visible window."""
    if n_obs_steps <= 0:
        raise ValueError("n_obs_steps must be positive")
    return [*_copy_window(window), _copy_entry(entry)][-n_obs_steps:]


def obs_window_to_torch(
    window: list[dict[str, np.ndarray | bool | float]],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Convert a rolling observation window into a batched DP3 observation dict."""
    point_cloud = np.stack([entry["point_cloud"] for entry in window], axis=0)
    agent_pos = np.stack([entry["agent_pos"] for entry in window], axis=0)
    return {
        "point_cloud": torch.from_numpy(point_cloud.astype(np.float32)).unsqueeze(0).to(device),
        "agent_pos": torch.from_numpy(agent_pos.astype(np.float32)).unsqueeze(0).to(device),
    }


def policy_action_to_sim_action(
    policy_action: np.ndarray,
    current_state: np.ndarray,
    *,
    action_mode: ActionMode,
    sim_action_dim: int,
    low: np.ndarray | None = None,
    high: np.ndarray | None = None,
    gripper_open: float = 0.04,
) -> np.ndarray:
    """Convert a 7D DP3 arm label into the ManiSkill Panda simulator action."""
    action = np.asarray(policy_action, dtype=np.float32).reshape(-1)
    state = np.asarray(current_state, dtype=np.float32).reshape(-1)
    if action.shape[0] != 7:
        raise ValueError(f"policy action must have shape [7], got {action.shape}")
    if state.shape[0] < 7:
        raise ValueError(f"current state must have at least 7 values, got {state.shape}")
    if action_mode == "abs_joint":
        arm_action = action
    elif action_mode == "delta_joint":
        arm_action = state[:7] + action
    else:
        raise ValueError(f"unsupported action_mode {action_mode!r}")

    if sim_action_dim == 7:
        sim_action = arm_action.astype(np.float32, copy=True)
    elif sim_action_dim == 8:
        sim_action = np.concatenate([arm_action, [gripper_open]]).astype(np.float32)
    else:
        raise ValueError(f"unsupported simulator action dimension {sim_action_dim}")
    if low is not None and high is not None:
        sim_action = np.clip(sim_action, np.asarray(low), np.asarray(high)).astype(np.float32)
    return sim_action


def select_rollout_specs(
    *,
    source: Source,
    dataset_episode_seeds: list[int],
    episodes: int,
    episode_indices: list[int] | None = None,
    seed_start: int = 10000,
) -> list[RolloutSpec]:
    """Select dataset or fresh rollout seeds, skipping training seeds for fresh rollouts."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if source == "dataset":
        if episode_indices is None:
            episode_indices = list(range(min(episodes, len(dataset_episode_seeds))))
        specs = []
        for output_idx, dataset_idx in enumerate(episode_indices):
            if dataset_idx < 0 or dataset_idx >= len(dataset_episode_seeds):
                raise IndexError(f"dataset episode index {dataset_idx} is out of range")
            specs.append(
                RolloutSpec(
                    output_index=output_idx,
                    seed=dataset_episode_seeds[dataset_idx],
                    source="dataset",
                    dataset_episode_index=dataset_idx,
                )
            )
        return specs
    if source == "fresh":
        if episode_indices is not None:
            raise ValueError("--episode-indices is only valid with --source dataset")
        training_seeds = set(dataset_episode_seeds)
        specs = []
        candidate = seed_start
        while len(specs) < episodes:
            if candidate not in training_seeds:
                specs.append(
                    RolloutSpec(
                        output_index=len(specs),
                        seed=candidate,
                        source="fresh",
                    )
                )
            candidate += 1
        return specs
    raise ValueError(f"unsupported source {source!r}")


def save_video(path: Path, frames: list[np.ndarray], *, fps: int) -> None:
    if not frames:
        raise RuntimeError("no frames were captured for video export")
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)


def save_rerun_timeline(path: Path, timeline: list[dict[str, np.ndarray | bool | float]]) -> None:
    try:
        import rerun as rr
    except Exception as exc:
        raise RuntimeError(
            "Rerun export requires: "
            "uv sync --extra cu129 --extra maniskill --extra viz --group dev --group notebooks"
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    rr.init("pg3d_dp3_reach_policy_rollout", spawn=False)
    rr.save(str(path))
    for step_idx, entry in enumerate(timeline):
        rr.set_time_sequence("step", step_idx)
        valid = np.asarray(entry["point_valid_mask"], dtype=bool)
        points = np.asarray(entry["point_cloud"], dtype=np.float32)[valid]
        if points.size:
            rr.log("world/point_cloud", rr.Points3D(points, colors=[180, 180, 180]))
            robot_points = points[np.asarray(entry["robot_mask"], dtype=bool)[valid]]
            if robot_points.size:
                rr.log("world/robot_points", rr.Points3D(robot_points, colors=[0, 128, 255]))
        target = np.asarray(entry["target_position"], dtype=np.float32).reshape(1, 3)
        if np.all(np.isfinite(target)):
            rr.log("world/goal", rr.Points3D(target, colors=[0, 255, 0]))
        tcp = np.asarray(entry["tcp_pose"], dtype=np.float32).reshape(-1)[:3].reshape(1, 3)
        if np.all(np.isfinite(tcp)):
            rr.log("world/tcp", rr.Points3D(tcp, colors=[255, 220, 0]))
    rr.disconnect()


def _copy_window(
    window: list[dict[str, np.ndarray | bool | float]],
) -> list[dict[str, np.ndarray | bool | float]]:
    return [_copy_entry(entry) for entry in window]


def _copy_entry(
    entry: dict[str, np.ndarray | bool | float],
) -> dict[str, np.ndarray | bool | float]:
    return {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in entry.items()
    }


def _select_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but torch.cuda.is_available() is false")
    return torch.device(value)


def _env_task_name(env: Any) -> str:
    unwrapped = getattr(env, "unwrapped", env)
    spec = getattr(unwrapped, "spec", None)
    return str(getattr(spec, "id", "unknown"))


def _action_mode(value: str) -> ActionMode:
    if value not in {"abs_joint", "delta_joint"}:
        raise ValueError(f"unsupported action_mode {value!r}")
    return value  # type: ignore[return-value]


def _bool_info(info: dict[str, Any], key: str) -> bool:
    return bool(np.asarray(_to_numpy(info[key])).reshape(-1)[0]) if key in info else False


def _float_info(info: dict[str, Any], key: str, *, default: float) -> float:
    if key not in info:
        return float(default)
    return float(np.asarray(_to_numpy(info[key])).reshape(-1)[0])


def _float_value(value: Any) -> float:
    return float(np.asarray(_to_numpy(value)).reshape(-1)[0])


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


def _frame_to_numpy(frame: Any) -> np.ndarray:
    array = _to_numpy(frame)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    return array.astype(np.uint8, copy=False)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
