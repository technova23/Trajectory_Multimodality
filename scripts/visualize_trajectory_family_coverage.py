from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TrajectoryFamilySpec:
    family_id: int
    name: str
    lateral_scale: float
    vertical_scale: float
    ratios: tuple[float, ...]
    min_curve_multiplier: float = 1.0
    lateral_jitter: float = 0.08
    vertical_jitter: float = 0.05


def main() -> int:
    args = parse_args()
    families = _trajectory_variant_specs(args.families)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for distance_cm in args.distances_cm:
        distance_m = float(distance_cm) / 100.0
        start = np.asarray([0.0, 0.0, args.z], dtype=np.float64)
        goal = np.asarray([distance_m, 0.0, args.z], dtype=np.float64)
        rng = np.random.default_rng(args.seed + int(round(distance_cm)))
        envelopes = {
            spec.family_id: _extreme_family_waypoints(
                start=start,
                goal=goal,
                spec=spec,
                xy_noise=args.waypoint_xy_noise,
                z_noise=args.waypoint_z_noise,
                lateral_z_offset=args.lateral_z_offset,
                vertical_lateral_offset=args.vertical_lateral_offset,
            )
            for spec in families
        }
        samples = (
            {
                spec.family_id: _sample_family_waypoints(
                    start=start,
                    goal=goal,
                    spec=spec,
                    samples=args.samples,
                    xy_noise=args.waypoint_xy_noise,
                    z_noise=args.waypoint_z_noise,
                    lateral_z_offset=args.lateral_z_offset,
                    vertical_lateral_offset=args.vertical_lateral_offset,
                    rng=rng,
                )
                for spec in families
            }
            if args.samples > 0
            else {}
        )
        prefix = args.output_dir / f"coverage_{int(round(distance_cm)):02d}cm"
        saved.extend(
            _plot_distance_set(
                start=start,
                goal=goal,
                distance_cm=float(distance_cm),
                families=families,
                envelopes=envelopes,
                samples=samples,
                tube_radius=args.tube_radius,
                prefix=prefix,
                formats=args.formats,
            )
        )

    for output in saved:
        print(f"saved: {output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize XY, XZ, and 3D waypoint coverage for multimodal reach "
            "trajectory families across several start-goal distances."
        )
    )
    parser.add_argument(
        "--distances-cm",
        type=float,
        nargs="+",
        default=[20, 30, 40, 50, 60, 70, 80],
        help="start-goal distances to plot, in centimeters",
    )
    parser.add_argument("--z", type=float, default=0.45, help="shared start-goal z height")
    parser.add_argument("--families", type=int, default=12)
    parser.add_argument(
        "--samples",
        type=int,
        default=250,
        help="random interior dots per family; shaded hulls use exact extrema",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--waypoint-xy-noise", type=float, default=0.04)
    parser.add_argument("--waypoint-z-noise", type=float, default=0.025)
    parser.add_argument("--lateral-z-offset", type=float, default=0.15)
    parser.add_argument("--vertical-lateral-offset", type=float, default=0.10)
    parser.add_argument(
        "--tube-radius",
        type=float,
        default=0.18,
        help="reference radius around start-goal axis to check coverage against",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/trajectory_family_coverage"),
    )
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf", "svg"])
    args = parser.parse_args()
    if not args.distances_cm or any(distance <= 0 for distance in args.distances_cm):
        raise ValueError("--distances-cm values must be positive")
    if args.families <= 0:
        raise ValueError("--families must be positive")
    if args.samples < 0:
        raise ValueError("--samples must be non-negative")
    if args.tube_radius <= 0:
        raise ValueError("--tube-radius must be positive")
    return args


def _plot_distance_set(
    *,
    start: np.ndarray,
    goal: np.ndarray,
    distance_cm: float,
    families: list[TrajectoryFamilySpec],
    envelopes: dict[int, np.ndarray],
    samples: dict[int, np.ndarray],
    tube_radius: float,
    prefix: Path,
    formats: Iterable[str],
) -> list[Path]:
    import matplotlib.pyplot as plt

    colors = plt.cm.tab20(np.linspace(0.0, 1.0, max(len(families), 1)))
    saved: list[Path] = []
    title_suffix = f"start-goal distance = {distance_cm:.0f} cm"

    fig_xy, ax_xy = plt.subplots(figsize=(8.5, 7.0), dpi=180)
    _plot_projection(
        ax_xy,
        start=start,
        goal=goal,
        families=families,
        envelopes=envelopes,
        samples=samples,
        colors=colors,
        projection="xy",
        tube_radius=tube_radius,
    )
    ax_xy.set_title(f"XY Family Coverage ({title_suffix})")
    saved.extend(_save_figure(fig_xy, prefix, "xy", formats))
    plt.close(fig_xy)

    fig_xz, ax_xz = plt.subplots(figsize=(8.5, 7.0), dpi=180)
    _plot_projection(
        ax_xz,
        start=start,
        goal=goal,
        families=families,
        envelopes=envelopes,
        samples=samples,
        colors=colors,
        projection="xz",
        tube_radius=tube_radius,
    )
    ax_xz.set_title(f"XZ Family Coverage ({title_suffix})")
    saved.extend(_save_figure(fig_xz, prefix, "xz", formats))
    plt.close(fig_xz)

    fig_3d = plt.figure(figsize=(9.2, 7.6), dpi=180)
    ax_3d = fig_3d.add_subplot(111, projection="3d")
    _plot_xyz(
        ax_3d,
        start=start,
        goal=goal,
        families=families,
        envelopes=envelopes,
        samples=samples,
        colors=colors,
        tube_radius=tube_radius,
    )
    ax_3d.set_title(f"XYZ Family Coverage ({title_suffix})")
    saved.extend(_save_figure(fig_3d, prefix, "xyz_3d", formats))
    plt.close(fig_3d)
    return saved


def _save_figure(fig: object, prefix: Path, view_name: str, formats: Iterable[str]) -> list[Path]:
    outputs = []
    for fmt in formats:
        output = prefix.with_name(f"{prefix.name}_{view_name}.{fmt}")
        fig.tight_layout()
        fig.savefig(output)
        outputs.append(output)
    return outputs


def _plot_projection(
    ax: object,
    *,
    start: np.ndarray,
    goal: np.ndarray,
    families: list[TrajectoryFamilySpec],
    envelopes: dict[int, np.ndarray],
    samples: dict[int, np.ndarray],
    colors: np.ndarray,
    projection: str,
    tube_radius: float,
) -> None:
    if projection == "xy":
        columns = [0, 1]
        start_2d = start[[0, 1]]
        goal_2d = goal[[0, 1]]
        xlabel = "x along start-goal axis (m)"
        ylabel = "y / left-right lateral coverage (m)"
    elif projection == "xz":
        columns = [0, 2]
        start_2d = start[[0, 2]]
        goal_2d = goal[[0, 2]]
        xlabel = "x along start-goal axis (m)"
        ylabel = "z / vertical coverage (m)"
    else:
        raise ValueError(f"unsupported projection {projection!r}")

    _draw_reference_band(ax, start, goal, radius=tube_radius, projection=projection)
    for color, spec in zip(colors, families, strict=True):
        envelope = envelopes[spec.family_id][:, columns]
        sample_points = samples.get(spec.family_id)
        sample_2d = sample_points[:, columns] if sample_points is not None else None
        _scatter_and_hull(
            ax,
            envelope,
            sample_2d,
            color=color,
            label=f"{spec.family_id}: {spec.name}",
        )

    ax.plot([start_2d[0], goal_2d[0]], [start_2d[1], goal_2d[1]], "k--", linewidth=1.5)
    ax.scatter([start_2d[0]], [start_2d[1]], c="gold", s=90, edgecolors="black", zorder=5)
    ax.scatter([goal_2d[0]], [goal_2d[1]], c="limegreen", s=95, edgecolors="black", zorder=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axis("equal")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)


def _plot_xyz(
    ax: object,
    *,
    start: np.ndarray,
    goal: np.ndarray,
    families: list[TrajectoryFamilySpec],
    envelopes: dict[int, np.ndarray],
    samples: dict[int, np.ndarray],
    colors: np.ndarray,
    tube_radius: float,
) -> None:
    _draw_reference_cylinder(ax, start, goal, radius=tube_radius)
    for color, spec in zip(colors, families, strict=True):
        envelope = envelopes[spec.family_id]
        sample_points = samples.get(spec.family_id)
        if sample_points is not None:
            ax.scatter(
                sample_points[:, 0],
                sample_points[:, 1],
                sample_points[:, 2],
                s=2,
                color=color,
                alpha=0.05,
            )
        ax.scatter(
            envelope[:, 0],
            envelope[:, 1],
            envelope[:, 2],
            s=12,
            color=color,
            alpha=0.72,
            label=f"{spec.family_id}: {spec.name}",
        )

    ax.plot([start[0], goal[0]], [start[1], goal[1]], [start[2], goal[2]], "k--", linewidth=1.6)
    ax.scatter([start[0]], [start[1]], [start[2]], c="gold", s=90, edgecolors="black")
    ax.scatter([goal[0]], [goal[1]], [goal[2]], c="limegreen", s=95, edgecolors="black")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y / lateral (m)")
    ax.set_zlabel("z / vertical (m)")
    ax.view_init(elev=24, azim=-58)
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=7)
    _set_equal_3d_axes(ax, np.concatenate([*envelopes.values(), start[None], goal[None]], axis=0))


def _extreme_family_waypoints(
    *,
    start: np.ndarray,
    goal: np.ndarray,
    spec: TrajectoryFamilySpec,
    xy_noise: float,
    z_noise: float,
    lateral_z_offset: float,
    vertical_lateral_offset: float,
) -> np.ndarray:
    delta = goal - start
    distance = float(np.linalg.norm(delta))
    lateral_axis, vertical_axis = _trajectory_basis(start, goal)
    lateral_mag, vertical_mag, lateral_scale, vertical_scale = _family_effective_scales(
        spec,
        distance=distance,
    )
    center = (len(spec.ratios) - 1) / 2.0
    points: list[np.ndarray] = []
    for idx, ratio in enumerate(spec.ratios):
        envelope = 1.0 + 0.10 * (1.0 - abs(idx - center) / max(center, 1.0))
        for ratio_jitter in (-0.035, 0.035):
            path_ratio = float(np.clip(ratio + ratio_jitter, 0.14, 0.86))
            base_point = start + path_ratio * delta
            for lateral_noise in (-spec.lateral_jitter, spec.lateral_jitter):
                lateral_offset = (lateral_scale + lateral_noise) * lateral_mag * envelope
                for vertical_noise in (-spec.vertical_jitter, spec.vertical_jitter):
                    vertical_offset = (vertical_scale + vertical_noise) * vertical_mag * envelope
                    vertical_offset = _clip_vertical_offset(
                        vertical_offset,
                        vertical_mag=vertical_mag,
                        lateral_z_offset=lateral_z_offset,
                        vertical_lateral_offset=vertical_lateral_offset,
                    )
                    offset = lateral_offset * lateral_axis + vertical_offset * vertical_axis
                    for noise_x in (-xy_noise, xy_noise):
                        for noise_y in (-xy_noise, xy_noise):
                            for noise_z in (-z_noise, z_noise):
                                perturbation = np.asarray([noise_x, noise_y, noise_z], dtype=np.float64)
                                points.append(base_point + offset + perturbation)
    return np.asarray(points, dtype=np.float64)


def _sample_family_waypoints(
    *,
    start: np.ndarray,
    goal: np.ndarray,
    spec: TrajectoryFamilySpec,
    samples: int,
    xy_noise: float,
    z_noise: float,
    lateral_z_offset: float,
    vertical_lateral_offset: float,
    rng: np.random.Generator,
) -> np.ndarray:
    delta = goal - start
    distance = float(np.linalg.norm(delta))
    lateral_axis, vertical_axis = _trajectory_basis(start, goal)
    lateral_mag, vertical_mag, lateral_scale, vertical_scale = _family_effective_scales(
        spec,
        distance=distance,
    )
    points: list[np.ndarray] = []
    center = (len(spec.ratios) - 1) / 2.0
    for _ in range(samples):
        for idx, ratio in enumerate(spec.ratios):
            ratio_jitter = float(rng.uniform(-0.035, 0.035))
            path_ratio = float(np.clip(ratio + ratio_jitter, 0.14, 0.86))
            envelope = 1.0 + 0.10 * (1.0 - abs(idx - center) / max(center, 1.0))
            lateral_noise = float(rng.uniform(-spec.lateral_jitter, spec.lateral_jitter))
            vertical_noise = float(rng.uniform(-spec.vertical_jitter, spec.vertical_jitter))
            lateral_offset = (lateral_scale + lateral_noise) * lateral_mag * envelope
            vertical_offset = (vertical_scale + vertical_noise) * vertical_mag * envelope
            vertical_offset = _clip_vertical_offset(
                vertical_offset,
                vertical_mag=vertical_mag,
                lateral_z_offset=lateral_z_offset,
                vertical_lateral_offset=vertical_lateral_offset,
            )
            perturbation = np.asarray(
                [
                    rng.uniform(-xy_noise, xy_noise),
                    rng.uniform(-xy_noise, xy_noise),
                    rng.uniform(-z_noise, z_noise),
                ],
                dtype=np.float64,
            )
            base_point = start + path_ratio * delta
            offset = lateral_offset * lateral_axis + vertical_offset * vertical_axis
            points.append(base_point + offset + perturbation)
    return np.asarray(points, dtype=np.float64)


def _clip_vertical_offset(
    value: float,
    *,
    vertical_mag: float,
    lateral_z_offset: float,
    vertical_lateral_offset: float,
) -> float:
    return float(
        np.clip(
            value,
            -max(0.08, vertical_lateral_offset + 0.5 * lateral_z_offset),
            max(0.10, vertical_lateral_offset + 0.5 * lateral_z_offset, vertical_mag),
        )
    )


def _scatter_and_hull(
    ax: object,
    envelope_2d: np.ndarray,
    sample_2d: np.ndarray | None,
    *,
    color: np.ndarray,
    label: str,
) -> None:
    if sample_2d is not None:
        ax.scatter(sample_2d[:, 0], sample_2d[:, 1], s=2, color=color, alpha=0.08)
    ax.scatter(envelope_2d[:, 0], envelope_2d[:, 1], s=13, color=color, alpha=0.76)
    hull = _convex_hull(envelope_2d)
    if hull.shape[0] >= 3:
        ax.fill(hull[:, 0], hull[:, 1], color=color, alpha=0.18, label=label)


def _draw_reference_band(
    ax: object,
    start: np.ndarray,
    goal: np.ndarray,
    *,
    radius: float,
    projection: str,
) -> None:
    x_min = min(float(start[0]), float(goal[0]))
    x_max = max(float(start[0]), float(goal[0]))
    center = 0.0 if projection == "xy" else float(start[2])
    ax.fill_between(
        [x_min, x_max],
        [center - radius, center - radius],
        [center + radius, center + radius],
        color="black",
        alpha=0.045,
        label="reference tube",
    )


def _draw_reference_cylinder(ax: object, start: np.ndarray, goal: np.ndarray, *, radius: float) -> None:
    x_values = np.linspace(float(start[0]), float(goal[0]), 32)
    theta = np.linspace(0.0, 2.0 * np.pi, 32)
    xx, tt = np.meshgrid(x_values, theta)
    yy = radius * np.cos(tt)
    zz = float(start[2]) + radius * np.sin(tt)
    ax.plot_wireframe(xx, yy, zz, color="black", alpha=0.08, linewidth=0.4)


def _family_effective_scales(
    spec: TrajectoryFamilySpec,
    *,
    distance: float,
) -> tuple[float, float, float, float]:
    lateral_mag = max(0.12, min(0.25, 0.60 * distance))
    vertical_mag = max(0.06, min(0.20, 0.50 * distance))
    if spec.name == "shallow_direct":
        lateral_mag = max(0.02, min(0.06, 0.15 * distance))
        vertical_mag = max(0.01, min(0.04, 0.10 * distance))

    lateral_scale = float(np.clip(spec.lateral_scale, -1.0, 1.0))
    vertical_scale = float(np.clip(spec.vertical_scale, -0.85, 1.0))
    if spec.name in {"high_loop", "upper_arc"}:
        vertical_scale = min(vertical_scale, 0.90)
    elif spec.name in {"low_loop", "lower_arc"}:
        vertical_scale = max(vertical_scale, -0.70)
    return lateral_mag, vertical_mag, lateral_scale, vertical_scale


def _convex_hull(points: np.ndarray) -> np.ndarray:
    points = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if points.shape[0] <= 2:
        return points
    points = points[np.lexsort((points[:, 1], points[:, 0]))]

    def cross(origin: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float(
            (a[0] - origin[0]) * (b[1] - origin[1])
            - (a[1] - origin[1]) * (b[0] - origin[0])
        )

    lower: list[np.ndarray] = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _set_equal_3d_axes(ax: object, points: np.ndarray) -> None:
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    centers = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins) * 0.55)
    radius = max(radius, 0.1)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def _trajectory_basis(start: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta = np.asarray(goal - start, dtype=np.float64)
    horizontal = delta[:2]
    horizontal_norm = float(np.linalg.norm(horizontal))
    if horizontal_norm < 1e-6:
        lateral_axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        lateral_axis = np.asarray([-horizontal[1], horizontal[0], 0.0], dtype=np.float64)
        lateral_axis /= np.linalg.norm(lateral_axis[:2])
    vertical_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return lateral_axis, vertical_axis


def _trajectory_variant_specs(variants_per_reset: int) -> list[TrajectoryFamilySpec]:
    base_specs: list[TrajectoryFamilySpec] = [
        TrajectoryFamilySpec(0, "left_wide", -0.90, 0.00, (0.34, 0.68), 1.10),
        TrajectoryFamilySpec(1, "right_wide", 0.90, 0.00, (0.34, 0.68), 1.10),
        TrajectoryFamilySpec(2, "upper_left", -0.68, 0.90, (0.30, 0.62), 1.15),
        TrajectoryFamilySpec(3, "upper_right", 0.68, 0.90, (0.30, 0.62), 1.15),
        TrajectoryFamilySpec(4, "lower_left", -0.36, -0.34, (0.35, 0.70), 1.05),
        TrajectoryFamilySpec(5, "lower_right", 0.36, -0.34, (0.35, 0.70), 1.05),
        TrajectoryFamilySpec(6, "upper_arc", 0.00, 1.15, (0.50,), 0.90),
        TrajectoryFamilySpec(7, "lower_arc", 0.00, -0.48, (0.50,), 0.80),
        TrajectoryFamilySpec(8, "high_loop", -0.30, 0.95, (0.50,), 0.95),
        TrajectoryFamilySpec(9, "low_loop", 0.48, -0.52, (0.50,), 1.00),
        TrajectoryFamilySpec(
            10,
            "shallow_direct",
            0.16,
            0.08,
            (0.50,),
            min_curve_multiplier=0.0,
            lateral_jitter=0.025,
            vertical_jitter=0.015,
        ),
        TrajectoryFamilySpec(11, "extreme_detour", 0.75, 0.38, (0.30, 0.60), 1.00),
    ]
    return base_specs[:variants_per_reset]


if __name__ == "__main__":
    raise SystemExit(main())
