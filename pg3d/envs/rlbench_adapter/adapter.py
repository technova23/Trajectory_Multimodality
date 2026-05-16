from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from pg3d.envs.rlbench_adapter.models import Observation, RobotState, SimGroundTruth

DEFAULT_CAMERA_NAMES = (
    "left_shoulder",
    "right_shoulder",
    "overhead",
    "wrist",
    "front",
)


def adapt_rlbench_observation(
    obs: Any,
    *,
    task_name: str = "ReachTarget",
    descriptions: Sequence[str] | None = None,
    camera_names: Sequence[str] = DEFAULT_CAMERA_NAMES,
    robot_mask_ids: Iterable[int] | None = None,
    object_mask_ids: Mapping[str, Iterable[int]] | None = None,
) -> Observation:
    """Convert an RLBench-like observation into pg3d core objects.

    This function deliberately avoids importing RLBench or PyRep. It consumes
    objects by attribute name so CPU-only tests can exercise the adapter with
    synthetic observations.
    """
    point_chunks: list[np.ndarray] = []
    rgb_chunks: list[np.ndarray] = []
    mask_chunks: list[np.ndarray] = []
    camera_index_chunks: list[np.ndarray] = []
    points_by_camera: dict[str, int] = {}
    missing_rgb_cameras: list[str] = []
    missing_mask_cameras: list[str] = []

    for camera_index, camera_name in enumerate(camera_names):
        point_cloud = getattr(obs, f"{camera_name}_point_cloud", None)
        if point_cloud is None:
            points_by_camera[camera_name] = 0
            continue

        flat_points = _flatten_point_cloud(point_cloud, name=f"{camera_name}_point_cloud")
        finite = np.isfinite(flat_points).all(axis=1)
        flat_points = flat_points[finite]
        point_chunks.append(flat_points)
        points_by_camera[camera_name] = int(flat_points.shape[0])
        camera_index_chunks.append(
            np.full(flat_points.shape[0], camera_index, dtype=np.int16)
        )

        rgb = getattr(obs, f"{camera_name}_rgb", None)
        if rgb is None:
            missing_rgb_cameras.append(camera_name)
        else:
            flat_rgb = _flatten_rgb(
                rgb,
                expected_rows=finite.shape[0],
                name=camera_name,
            )
            rgb_chunks.append(flat_rgb[finite])

        mask = getattr(obs, f"{camera_name}_mask", None)
        if mask is None:
            missing_mask_cameras.append(camera_name)
        else:
            mask_chunks.append(
                _flatten_instance_mask(mask, expected_rows=finite.shape[0], name=camera_name)[
                    finite
                ]
            )

    if not point_chunks:
        raise ValueError("RLBench observation did not contain any configured point clouds")

    point_cloud = np.concatenate(point_chunks, axis=0).astype(np.float32, copy=False)
    point_features: dict[str, np.ndarray] = {
        "camera_index": np.concatenate(camera_index_chunks, axis=0)
    }

    num_point_chunks = len(point_chunks)
    if len(rgb_chunks) == num_point_chunks:
        point_features["rgb"] = np.concatenate(rgb_chunks, axis=0)
    if len(mask_chunks) == num_point_chunks:
        instance_ids = np.concatenate(mask_chunks, axis=0).astype(np.int64, copy=False)
        point_features["instance_id"] = instance_ids
    else:
        instance_ids = None

    metadata: dict[str, Any] = {
        "source": "rlbench",
        "camera_names": list(camera_names),
        "points_by_camera": points_by_camera,
    }
    if missing_rgb_cameras:
        metadata["missing_rgb_cameras"] = missing_rgb_cameras
    if missing_mask_cameras:
        metadata["missing_mask_cameras"] = missing_mask_cameras

    robot_mask = _ids_to_mask(instance_ids, robot_mask_ids)
    if robot_mask is None:
        metadata["robot_mask_status"] = (
            "missing_instance_ids" if robot_mask_ids else "missing_robot_mask_ids"
        )
    else:
        metadata["robot_mask_status"] = "available"

    object_masks: dict[str, np.ndarray] = {}
    if object_mask_ids:
        if instance_ids is None:
            metadata["object_mask_status"] = "missing_instance_ids"
        else:
            for name, ids in object_mask_ids.items():
                mask = _ids_to_mask(instance_ids, ids)
                if mask is not None:
                    object_masks[name] = mask
            metadata["object_mask_status"] = "available"

    return Observation(
        point_cloud=point_cloud,
        point_features=point_features,
        robot_mask=robot_mask,
        object_masks=object_masks,
        robot_state=extract_robot_state(obs),
        sim_gt=extract_sim_ground_truth(
            obs,
            task_name=task_name,
            descriptions=descriptions,
        ),
        metadata=metadata,
    )


def extract_robot_state(obs: Any) -> RobotState:
    joint_positions = getattr(obs, "joint_positions", None)
    if joint_positions is None:
        raise ValueError("RLBench observation is missing joint_positions")
    return RobotState(
        joint_positions=joint_positions,
        joint_velocities=getattr(obs, "joint_velocities", None),
        joint_forces=getattr(obs, "joint_forces", None),
        gripper_open=getattr(obs, "gripper_open", None),
        gripper_pose=getattr(obs, "gripper_pose", None),
        gripper_matrix=getattr(obs, "gripper_matrix", None),
        gripper_joint_positions=getattr(obs, "gripper_joint_positions", None),
        gripper_touch_forces=getattr(obs, "gripper_touch_forces", None),
        metadata={"source": "rlbench"},
    )


def extract_sim_ground_truth(
    obs: Any,
    *,
    task_name: str,
    descriptions: Sequence[str] | None = None,
) -> SimGroundTruth:
    task_low_dim_state = getattr(obs, "task_low_dim_state", None)
    task_low_dim_array = None
    target_position = None
    if task_low_dim_state is not None:
        task_low_dim_array = np.asarray(task_low_dim_state, dtype=np.float32).reshape(-1)
        if task_name == "ReachTarget" and task_low_dim_array.shape[0] >= 3:
            target_position = task_low_dim_array[:3]
    misc = getattr(obs, "misc", None)
    metadata = {"source": "rlbench"}
    if isinstance(misc, dict) and "variation_index" in misc:
        metadata["variation_index"] = misc["variation_index"]
    return SimGroundTruth(
        task_name=task_name,
        target_position=target_position,
        task_low_dim_state=task_low_dim_array,
        descriptions=tuple(descriptions or ()),
        metadata=metadata,
    )


def _flatten_point_cloud(value: Any, *, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float32)
    if points.ndim < 2 or points.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [..., 3], got {points.shape}")
    return points.reshape(-1, 3)


def _flatten_rgb(value: Any, *, expected_rows: int, name: str) -> np.ndarray:
    rgb = np.asarray(value)
    if rgb.ndim < 2 or rgb.shape[-1] != 3:
        raise ValueError(f"{name}_rgb must have shape [..., 3], got {rgb.shape}")
    rgb = rgb.reshape(-1, 3)
    if rgb.shape[0] != expected_rows:
        raise ValueError(
            f"{name}_rgb row count must match point cloud rows: "
            f"{rgb.shape[0]} != {expected_rows}"
        )
    if np.issubdtype(rgb.dtype, np.floating):
        if rgb.size > 0 and np.nanmax(rgb) <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    elif rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def _flatten_instance_mask(value: Any, *, expected_rows: int, name: str) -> np.ndarray:
    mask = np.asarray(value)
    if mask.ndim >= 1 and mask.shape[-1] == 1:
        mask = np.squeeze(mask, axis=-1)
    if mask.ndim >= 3 and mask.shape[-1] == 3:
        mask = _rgb_handles_to_mask(mask)
    elif mask.ndim != 1 and mask.ndim != 2:
        raise ValueError(f"{name}_mask must have shape [N], [H, W], or [H, W, 3]")
    mask = mask.reshape(-1).astype(np.int64, copy=False)
    if mask.shape[0] != expected_rows:
        raise ValueError(
            f"{name}_mask row count must match point cloud rows: "
            f"{mask.shape[0]} != {expected_rows}"
        )
    return mask


def _rgb_handles_to_mask(rgb_coded_handles: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb_coded_handles)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint32)
    else:
        rgb = rgb.astype(np.uint32)
    return rgb[..., 0] + rgb[..., 1] * 256 + rgb[..., 2] * 256 * 256


def _ids_to_mask(instance_ids: np.ndarray | None, ids: Iterable[int] | None) -> np.ndarray | None:
    if instance_ids is None or ids is None:
        return None
    ids_array = np.asarray(list(ids), dtype=np.int64)
    if ids_array.size == 0:
        return None
    return np.isin(instance_ids, ids_array)
