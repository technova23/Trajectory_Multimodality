from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pg3d.eval import (
    AvoidOverlayConfig,
    EpisodePath,
    candidate_feasibility_fraction,
    concatenate_rollouts,
    direct_path_avoid_region,
    episode_metric_row,
    min_constraint_clearance,
    path_satisfies_constraints,
    save_episode_constraints,
    summarize_metrics,
    validate_planning_horizons,
    wilson_interval,
)
from pg3d.world_model import ActionChunk, ImaginedRollout


def test_direct_path_avoid_region_and_json_persistence(tmp_path: Path) -> None:
    constraint = direct_path_avoid_region(
        start_tcp=[0.0, 0.0, 0.2],
        target_position=[0.4, 0.0, 0.2],
        config=AvoidOverlayConfig(radius=0.08),
    )

    np.testing.assert_allclose(constraint.region.center, [0.2, 0.0, 0.2])
    assert constraint.region.radius == pytest.approx(0.08)

    path = tmp_path / "constraints" / "episode_000.json"
    save_episode_constraints(path, [constraint])

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded[0]["type"] == "avoid_region"
    assert loaded[0]["region"]["type"] == "sphere"


def test_direct_path_avoid_region_clamps_radius_for_short_paths() -> None:
    constraint = direct_path_avoid_region(
        start_tcp=[0.0, 0.0, 0.0],
        target_position=[0.1, 0.0, 0.0],
        config=AvoidOverlayConfig(radius=0.08, min_radius=0.02),
    )

    assert constraint.region.radius == pytest.approx(0.045)


def test_wilson_interval_bounds_known_center() -> None:
    low, high = wilson_interval(5, 10)

    assert 0.23 < low < 0.24
    assert 0.76 < high < 0.77


def test_episode_metric_row_computes_clearance_and_combined_success() -> None:
    constraint = direct_path_avoid_region(
        start_tcp=[0.0, 0.0, 0.0],
        target_position=[1.0, 0.0, 0.0],
        config=AvoidOverlayConfig(radius=0.1),
    )
    path = EpisodePath()
    path.append(tcp_position=[0.0, 0.2, 0.0], q=[0.0, 0.0], target_distance=1.0)
    path.append(tcp_position=[0.5, 0.2, 0.0], q=[0.1, 0.0], target_distance=0.5)
    path.append(tcp_position=[1.0, 0.2, 0.0], q=[0.2, 0.0], target_distance=0.0)

    row = episode_metric_row(
        method="reranking",
        episode=0,
        seed=100,
        path=path,
        constraints=[constraint],
        reach_success=True,
        first_success_step=2,
        steps=2,
        replans=1,
        candidate_feasibility_fraction=0.5,
    )

    assert row["reach_success"] is True
    assert row["constraint_satisfied"] is True
    assert row["combined_success"] is True
    assert row["final_target_distance"] == pytest.approx(0.0)
    assert row["min_clearance"] == pytest.approx(0.1)
    assert row["candidate_feasibility_fraction"] == pytest.approx(0.5)


def test_constraint_satisfaction_fails_for_path_inside_sphere() -> None:
    constraint = direct_path_avoid_region(
        start_tcp=[0.0, 0.0, 0.0],
        target_position=[1.0, 0.0, 0.0],
        config=AvoidOverlayConfig(radius=0.1),
    )
    path = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32)

    assert min_constraint_clearance(path, [constraint]) < 0.0
    assert not path_satisfies_constraints(path, [constraint])


def test_validate_planning_horizons() -> None:
    validate_planning_horizons(planning_horizon_chunks=2, execution_horizon_chunks=1)

    with pytest.raises(ValueError, match="planning_horizon_chunks"):
        validate_planning_horizons(planning_horizon_chunks=0, execution_horizon_chunks=1)
    with pytest.raises(ValueError, match="<="):
        validate_planning_horizons(planning_horizon_chunks=1, execution_horizon_chunks=2)


def test_concatenate_rollouts_combines_multichunk_candidate() -> None:
    first = _rollout([[0.1] * 7, [0.2] * 7])
    second = _rollout([[0.3] * 7])

    combined = concatenate_rollouts([first, second], metadata={"candidate": 1})

    assert combined.action_chunk.horizon == 3
    assert combined.q.shape == (3, 9)
    assert combined.eef_path.shape == (3, 3)
    assert len(combined.scene_point_clouds) == 3
    assert combined.metadata["planning_horizon_chunks"] == 2
    assert combined.metadata["candidate"] == 1


def test_summarize_metrics_uses_stable_schema() -> None:
    rows = []
    for idx, success in enumerate([True, False]):
        path = EpisodePath()
        path.append(tcp_position=[0.0, 0.2, 0.0], q=[0.0, 0.0], target_distance=1.0)
        path.append(tcp_position=[1.0, 0.2, 0.0], q=[0.1, 0.0], target_distance=0.1)
        rows.append(
            episode_metric_row(
                method="base",
                episode=idx,
                seed=idx,
                path=path,
                constraints=[],
                reach_success=success,
                first_success_step=1 if success else None,
                steps=1,
                replans=1,
                candidate_feasibility_fraction=None,
            )
        )

    summary = summarize_metrics(rows)

    assert summary["base"]["episodes"] == 2
    assert summary["base"]["reach_success_rate"] == pytest.approx(0.5)
    assert "combined_success_wilson_low" in summary["base"]
    assert "final_target_distance_mean" in summary["base"]


def test_candidate_feasibility_fraction_validates_counts() -> None:
    assert candidate_feasibility_fraction(1, 4) == pytest.approx(0.25)
    assert candidate_feasibility_fraction(0, 0) is None
    with pytest.raises(ValueError):
        candidate_feasibility_fraction(2, 1)


def test_eval_helpers_import_without_heavy_runtime_deps() -> None:
    code = """
import importlib
import sys

importlib.import_module("pg3d.eval")
assert "mani_skill" not in sys.modules
assert "sapien" not in sys.modules
assert "gymnasium" not in sys.modules
assert "rerun" not in sys.modules
assert "wandb" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def _rollout(actions: list[list[float]]) -> ImaginedRollout:
    chunk = ActionChunk(
        actions=np.asarray(actions, dtype=np.float32),
        action_mode="abs_joint",
        dt=1.0,
    )
    horizon = chunk.horizon
    q = np.zeros((horizon, 9), dtype=np.float32)
    q[:, :7] = chunk.actions
    eef = np.stack(
        [
            np.asarray([float(idx), 0.0, 0.2], dtype=np.float32)
            for idx in range(horizon)
        ],
        axis=0,
    )
    return ImaginedRollout(
        q=q,
        eef_path=eef,
        robot_point_clouds=[np.zeros((1, 3), dtype=np.float32) for _ in range(horizon)],
        scene_point_clouds=[np.zeros((2, 3), dtype=np.float32) for _ in range(horizon)],
        robot_masks=[np.asarray([True, False], dtype=bool) for _ in range(horizon)],
        action_chunk=chunk,
    )
