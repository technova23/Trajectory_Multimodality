from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import imageio
import numpy as np

from pg3d.envs.rlbench_adapter import DEFAULT_CAMERA_NAMES, Observation, adapt_rlbench_observation
from pg3d.envs.rlbench_adapter.setup import (
    check_setup,
    disable_waypoint_validation_for_observation,
    load_reach_target_runtime,
    parse_bool,
    print_setup_report,
)


def parse_camera_names(value: str) -> tuple[str, ...]:
    camera_names = tuple(name.strip() for name in value.split(",") if name.strip())
    if not camera_names:
        raise argparse.ArgumentTypeError("at least one camera must be selected")
    unknown = sorted(set(camera_names) - set(DEFAULT_CAMERA_NAMES))
    if unknown:
        raise argparse.ArgumentTypeError(
            "unknown camera names: "
            f"{', '.join(unknown)}; expected one or more of {', '.join(DEFAULT_CAMERA_NAMES)}"
        )
    return camera_names


def build_reach_observation_config(
    runtime: Mapping[str, Any],
    *,
    camera_names: Sequence[str],
    image_size: tuple[int, int],
) -> Any:
    obs_config = runtime["ObservationConfig"]()
    obs_config.set_all_high_dim(False)
    obs_config.set_all_low_dim(False)

    enabled = set(camera_names)
    for camera_name in DEFAULT_CAMERA_NAMES:
        camera_config = getattr(obs_config, f"{camera_name}_camera")
        is_enabled = camera_name in enabled
        camera_config.rgb = is_enabled
        camera_config.depth = is_enabled
        camera_config.point_cloud = is_enabled
        camera_config.mask = is_enabled
        camera_config.image_size = image_size
        camera_config.masks_as_one_channel = True

    obs_config.joint_velocities = True
    obs_config.joint_positions = True
    obs_config.joint_forces = False
    obs_config.gripper_open = True
    obs_config.gripper_pose = True
    obs_config.gripper_matrix = True
    obs_config.gripper_joint_positions = True
    obs_config.gripper_touch_forces = False
    obs_config.task_low_dim_state = True
    return obs_config


def make_action_mode(runtime: Mapping[str, Any]) -> Any:
    return runtime["MoveArmThenGripper"](
        arm_action_mode=runtime["JointVelocity"](),
        gripper_action_mode=runtime["Discrete"](),
    )


def discover_robot_mask_ids(task_env: Any, object_type: Any) -> set[int]:
    handles: set[int] = set()
    scene = getattr(task_env, "_scene", None)
    robots = (
        getattr(task_env, "_robot", None),
        getattr(scene, "robot", None),
        getattr(scene, "_robot", None),
    )

    handles.update(_handles_from_objects(getattr(scene, "_robot_shapes", []) or []))
    for robot in robots:
        for owner in (
            getattr(robot, "arm", None) if robot is not None else None,
            getattr(robot, "gripper", None) if robot is not None else None,
        ):
            if owner is None or not hasattr(owner, "get_objects_in_tree"):
                continue
            try:
                objects = owner.get_objects_in_tree(object_type=object_type.SHAPE)
            except Exception:
                continue
            handles.update(_handles_from_objects(objects))
    return handles


def discover_reach_object_mask_ids(task_env: Any) -> dict[str, set[int]]:
    task = getattr(task_env, "_task", None)
    object_mask_ids: dict[str, set[int]] = {}
    for name in ("target", "distractor0", "distractor1"):
        obj = getattr(task, name, None)
        handle = _handle_from_object(obj)
        if handle is not None:
            object_mask_ids[name] = {handle}
    return object_mask_ids


def save_observation_bundle(
    observation: Observation,
    output_dir: Path,
    *,
    video: bool = False,
    video_frames: int = 72,
    video_fps: int = 12,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "observation.npz"
    summary_path = output_dir / "summary.json"

    np.savez_compressed(npz_path, **build_npz_payload(observation))
    summary = observation.summary()
    summary["artifacts"] = {"npz": npz_path.name}

    artifact_paths = {"npz": str(npz_path), "summary": str(summary_path)}
    if video:
        video_path = output_dir / "observation.mp4"
        save_point_cloud_video(observation, video_path, frames=video_frames, fps=video_fps)
        summary["artifacts"]["video"] = video_path.name
        artifact_paths["video"] = str(video_path)

    summary_path.write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
    return artifact_paths


def build_npz_payload(observation: Observation) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {
        "point_cloud": observation.point_cloud,
        "agent_pos": observation.robot_state.as_agent_pos(),
        "robot_joint_positions": observation.robot_state.joint_positions,
    }
    for key, value in observation.point_features.items():
        payload[f"point_feature_{key}"] = value
    if observation.robot_mask is not None:
        payload["robot_mask"] = observation.robot_mask
    for key, value in observation.object_masks.items():
        payload[f"object_mask_{key}"] = value

    robot_state = observation.robot_state
    for name in (
        "joint_velocities",
        "joint_forces",
        "gripper_pose",
        "gripper_matrix",
        "gripper_joint_positions",
        "gripper_touch_forces",
    ):
        value = getattr(robot_state, name)
        if value is not None:
            payload[f"robot_{name}"] = value
    if robot_state.gripper_open is not None:
        payload["robot_gripper_open"] = np.asarray([robot_state.gripper_open], dtype=np.float32)

    if observation.sim_gt is not None:
        if observation.sim_gt.target_position is not None:
            payload["sim_gt_target_position"] = observation.sim_gt.target_position
        if observation.sim_gt.task_low_dim_state is not None:
            payload["sim_gt_task_low_dim_state"] = observation.sim_gt.task_low_dim_state
    return payload


def save_point_cloud_video(
    observation: Observation,
    path: Path,
    *,
    frames: int = 72,
    fps: int = 12,
    max_points: int = 8000,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = observation.point_cloud
    if points.shape[0] > max_points:
        indices = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
        points = points[indices]
        robot_mask = (
            observation.robot_mask[indices] if observation.robot_mask is not None else None
        )
        rgb = (
            observation.point_features["rgb"][indices]
            if "rgb" in observation.point_features
            else None
        )
    else:
        robot_mask = observation.robot_mask
        rgb = observation.point_features.get("rgb")

    colors = _point_colors(points.shape[0], robot_mask=robot_mask, rgb=rgb)
    center, radius = _axis_center_radius(points)
    target_position = (
        observation.sim_gt.target_position
        if observation.sim_gt is not None and observation.sim_gt.target_position is not None
        else None
    )

    fig = plt.figure(figsize=(7, 6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    with imageio.get_writer(path, fps=fps) as writer:
        for frame_idx in range(frames):
            ax.clear()
            ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c=colors,
                s=2,
                alpha=0.85,
                linewidths=0,
            )
            if target_position is not None:
                ax.scatter(
                    [target_position[0]],
                    [target_position[1]],
                    [target_position[2]],
                    c=np.asarray([[0.0, 0.65, 0.25]]),
                    s=80,
                    marker="*",
                    depthshade=False,
                )
            _set_equal_axes(ax, center=center, radius=radius)
            ax.set_title("RLBench ReachTarget adapted observation")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.view_init(elev=24, azim=360.0 * frame_idx / max(frames, 1))
            fig.tight_layout()
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            writer.append_data(frame)
    plt.close(fig)


def run_save(
    *,
    headless: bool,
    output_dir: Path,
    camera_names: Sequence[str],
    image_size: tuple[int, int],
    visualize: bool,
    allow_missing_robot_mask: bool,
    video_frames: int,
) -> int:
    errors, warnings = check_setup()
    print_setup_report(errors, warnings)
    if errors:
        return 2

    try:
        runtime = load_reach_target_runtime()
    except Exception as exc:
        print(
            "Failed to import RLBench/PyRep after setup checks passed. "
            "This usually means CoppeliaSim shared libraries are not visible.",
            file=sys.stderr,
        )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    obs_config = build_reach_observation_config(
        runtime,
        camera_names=camera_names,
        image_size=image_size,
    )
    env = runtime["Environment"](
        action_mode=make_action_mode(runtime),
        obs_config=obs_config,
        headless=headless,
    )

    adapted: Observation | None = None
    try:
        print(f"launching RLBench ReachTarget observation save; headless={headless}")
        env.launch()
        task = env.get_task(runtime["ReachTarget"])
        disable_waypoint_validation_for_observation(task)
        descriptions, raw_obs = task.reset()

        robot_mask_ids = discover_robot_mask_ids(task, runtime["ObjectType"])
        object_mask_ids = discover_reach_object_mask_ids(task)
        adapted = adapt_rlbench_observation(
            raw_obs,
            task_name="ReachTarget",
            descriptions=descriptions,
            camera_names=camera_names,
            robot_mask_ids=robot_mask_ids,
            object_mask_ids=object_mask_ids,
        )
    except Exception as exc:
        print("RLBench ReachTarget observation save failed.", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "Check CoppeliaSim, camera/mask configuration, DISPLAY/headless "
            "configuration, and PyRep build logs.",
            file=sys.stderr,
        )
        return 1
    finally:
        try:
            env.shutdown()
            print("shutdown ok")
        except Exception as exc:
            print(f"shutdown failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    if adapted is None:
        print("RLBench observation save failed before adaptation.", file=sys.stderr)
        return 1
    if adapted.robot_mask is None and not allow_missing_robot_mask:
        print(
            "Robot mask could not be derived from RLBench masks. "
            "Use --allow-missing-robot-mask only for setup debugging.",
            file=sys.stderr,
        )
        return 1
    if adapted.robot_mask is not None and int(np.count_nonzero(adapted.robot_mask)) == 0:
        print(
            "Warning: robot_mask was derived, but no visible robot points were selected.",
            file=sys.stderr,
        )

    artifacts = save_observation_bundle(
        adapted,
        output_dir,
        video=visualize,
        video_frames=video_frames,
    )
    print("saved adapted observation:")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")
    print(f"points: {adapted.point_cloud.shape[0]}")
    print(f"robot_mask: {adapted.robot_mask is not None}")
    if adapted.sim_gt is not None and adapted.sim_gt.target_position is not None:
        print(f"target_position: {adapted.sim_gt.target_position.tolist()}")
    return 0


def _handles_from_objects(objects: Iterable[Any]) -> set[int]:
    handles: set[int] = set()
    for obj in objects:
        handle = _handle_from_object(obj)
        if handle is not None:
            handles.add(handle)
    return handles


def _handle_from_object(obj: Any) -> int | None:
    if obj is None or not hasattr(obj, "get_handle"):
        return None
    try:
        return int(obj.get_handle())
    except Exception:
        return None


def _point_colors(
    num_points: int,
    *,
    robot_mask: np.ndarray | None,
    rgb: np.ndarray | None,
) -> np.ndarray:
    if rgb is not None:
        colors = rgb.astype(np.float32)
        if colors.size > 0 and np.max(colors) > 1.0:
            colors /= 255.0
        colors = np.clip(colors, 0.0, 1.0)
    else:
        colors = np.full((num_points, 3), 0.55, dtype=np.float32)
    if robot_mask is not None:
        colors = colors.copy()
        colors[robot_mask] = np.asarray([0.85, 0.15, 0.12], dtype=np.float32)
    return colors


def _axis_center_radius(points: np.ndarray) -> tuple[np.ndarray, float]:
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    return center, max(radius, 1e-3)


def _set_equal_axes(ax: Any, *, center: np.ndarray, radius: float) -> None:
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch RLBench ReachTarget and save one adapted pg3d observation."
    )
    parser.add_argument(
        "--headless",
        nargs="?",
        const=True,
        default=True,
        type=parse_bool,
        help="Run CoppeliaSim headless. Accepts true/false; bare --headless means true.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/rlbench_observation"),
        help="Directory for summary.json, observation.npz, and optional observation.mp4.",
    )
    parser.add_argument(
        "--cameras",
        type=parse_camera_names,
        default=DEFAULT_CAMERA_NAMES,
        help="Comma-separated cameras to enable.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(128, 128),
        help="RLBench camera image size.",
    )
    parser.add_argument(
        "--visualize",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        help="Save a rotating point-cloud MP4 artifact.",
    )
    parser.add_argument(
        "--video-frames",
        type=int,
        default=72,
        help="Number of frames for --visualize output.",
    )
    parser.add_argument(
        "--allow-missing-robot-mask",
        action="store_true",
        help="Save even if robot_mask cannot be derived. Intended only for setup debugging.",
    )
    args = parser.parse_args()
    if args.image_size[0] <= 0 or args.image_size[1] <= 0:
        parser.error("--image-size values must be positive")
    if args.video_frames <= 0:
        parser.error("--video-frames must be positive")

    return run_save(
        headless=args.headless,
        output_dir=args.output_dir,
        camera_names=args.cameras,
        image_size=tuple(args.image_size),
        visualize=args.visualize,
        allow_missing_robot_mask=args.allow_missing_robot_mask,
        video_frames=args.video_frames,
    )


if __name__ == "__main__":
    raise SystemExit(main())
