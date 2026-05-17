from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from pg3d.composition import (
    CandidateDiagnostics,
    ControllerInput,
    ControllerResult,
    RejectionController,
    RerankingController,
    ScoreWeights,
)
from pg3d.composition.scoring import (
    consensus_deviations,
    goal_distance,
    primary_constraint_penalty,
    trajectory_smoothness,
)
from pg3d.constraints import AvoidRegion
from pg3d.envs.maniskill_adapter import (
    ManiSkillGhostPandaGeometryProvider,
    register_pg3d_reach_envs,
)
from pg3d.envs.maniskill_adapter.dataset import (
    PointCloudCropConfig,
    load_reach_metadata,
)
from pg3d.eval import (
    AvoidOverlayConfig,
    EpisodePath,
    candidate_feasibility_fraction,
    concatenate_rollouts,
    direct_path_avoid_region,
    episode_metric_row,
    save_episode_constraints,
    scene_context_for_constraints,
    summarize_metrics,
    validate_planning_horizons,
)
from pg3d.policies.dp3 import SimpleDP3
from pg3d.policies.dp3.checkpoint import (
    latest_reach_checkpoint,
    load_reach_policy_from_checkpoint,
)
from pg3d.utils.arrays import bool_any as _bool_any
from pg3d.utils.arrays import bool_info as _bool_info
from pg3d.utils.arrays import frame_to_numpy as _frame_to_numpy
from pg3d.utils.devices import select_device
from pg3d.utils.serialization import jsonable as _jsonable
from pg3d.world_model import ActionChunk, GeometricWorldModel, ImaginedRollout
from scripts.compare_world_model_rollout import (
    entry_to_world_model_observation,
    world_model_entry_from_rollout_step,
)
from scripts.rollout_dp3_reach_policy import (
    ActionMode,
    RolloutSpec,
    append_obs_window,
    crop_config_from_metadata,
    make_initial_obs_window,
    obs_window_to_torch,
    policy_action_to_sim_action,
    rollout_observation_entry,
    save_rerun_timeline,
    save_video,
    select_rollout_specs,
)

EvalMethod = Literal["base", "rejection", "reranking"]
Entry = dict[str, np.ndarray | bool | float]


@dataclass
class EvalDecisionSummary:
    """Compact per-replan diagnostic summary."""

    selected_chunk: ActionChunk
    result: ControllerResult | None
    candidate_feasible: int
    candidate_total: int
    selection_reason: str | None


class DP3ChunkPolicyAdapter:
    """Adapt `SimpleDP3.predict_action` to the P09 candidate-sampling protocol."""

    def __init__(
        self,
        policy: SimpleDP3,
        *,
        action_mode: ActionMode,
        device: torch.device,
        dt: float = 1.0,
    ) -> None:
        self.policy = policy
        self.action_mode = action_mode
        self.device = device
        self.dt = float(dt)

    def sample_action_chunks(
        self,
        policy_input: list[Entry],
        *,
        k: int,
        rng: np.random.Generator | None = None,
    ) -> list[ActionChunk]:
        """Sample `k` DP3 action chunks from one rolling observation window."""
        if k <= 0:
            raise ValueError("k must be positive")
        with torch.no_grad():
            batch = _repeat_obs_window_to_torch(policy_input, k=k, device=self.device)
            output = self.policy.predict_action(batch)
            actions = output["action"].detach().cpu().numpy()
        return [
            ActionChunk(
                actions=actions[idx].astype(np.float32, copy=True),
                action_mode=self.action_mode,
                dt=self.dt,
                metadata={"candidate_index": idx},
            )
            for idx in range(actions.shape[0])
        ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint, args.checkpoint_dir)
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
    metadata = load_reach_metadata(args.dataset)
    device = select_device(args.device)
    policy = load_reach_policy_from_checkpoint(
        checkpoint_path,
        device=device,
        prefer_ema=args.checkpoint_model == "ema",
    )
    action_mode = _action_mode(str(metadata.get("action_mode", "abs_joint")))
    crop_config = crop_config_from_metadata(metadata)
    goal_thresh = (
        float(args.goal_thresh)
        if args.goal_thresh is not None
        else float(dict(metadata.get("env_kwargs", {})).get("goal_thresh", 0.025))
    )
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
        raise RuntimeError("no constrained-reach episodes selected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run = _init_wandb(args, metadata=metadata, checkpoint_path=checkpoint_path)
    sim_env: Any | None = None
    ghost_env: Any | None = None
    rows: list[dict[str, Any]] = []
    metrics_path = args.output_dir / "metrics.jsonl"
    decisions_path = args.output_dir / "decisions.jsonl"
    rng = np.random.default_rng(args.seed)
    try:
        sim_env = gym.make(
            str(metadata["env_id"]),
            **_env_kwargs(metadata, render_mode="rgb_array" if args.video else None),
        )
        ghost_env = gym.make(str(metadata["env_id"]), **_env_kwargs(metadata, render_mode=None))
        adapter = DP3ChunkPolicyAdapter(
            policy,
            action_mode=action_mode,
            device=device,
        )
        with (
            metrics_path.open("w", encoding="utf-8") as metrics_file,
            decisions_path.open("w", encoding="utf-8") as decisions_file,
        ):
            for spec in specs:
                constraints = _episode_constraints(
                    sim_env,
                    spec=spec,
                    crop_config=crop_config,
                    args=args,
                )
                constraint_path = (
                    args.output_dir
                    / "constraints"
                    / f"episode_{spec.output_index:03d}.json"
                )
                save_episode_constraints(constraint_path, constraints)
                for method in args.methods:
                    row = run_eval_episode(
                        sim_env=sim_env,
                        ghost_env=ghost_env,
                        policy=policy,
                        adapter=adapter,
                        method=method,
                        spec=spec,
                        constraints=constraints,
                        action_mode=action_mode,
                        crop_config=crop_config,
                        goal_thresh=goal_thresh,
                        output_dir=args.output_dir,
                        max_steps=args.max_steps,
                        post_success_steps=args.post_success_steps,
                        planning_horizon_chunks=args.planning_horizon_chunks,
                        execution_horizon_chunks=args.execution_horizon_chunks,
                        k_schedule=tuple(args.k_schedule),
                        gripper_open=args.gripper_open,
                        match_current_robot_points=args.match_current_robot_points,
                        video=args.video,
                        rerun=args.rerun,
                        video_fps=args.video_fps,
                        decisions_file=decisions_file,
                        rng=rng,
                    )
                    rows.append(row)
                    metrics_file.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
                    metrics_file.flush()
                    print(
                        f"method={method} episode={spec.output_index} seed={spec.seed} "
                        f"combined={row['combined_success']} reach={row['reach_success']} "
                        f"constraint={row['constraint_satisfied']} "
                        f"final={_format_optional(row['final_target_distance'])} "
                        f"clearance={_format_optional(row['min_clearance'])}"
                    )
    except Exception as exc:
        print(f"Failed constrained reach eval: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if sim_env is not None:
            sim_env.close()
        if ghost_env is not None:
            ghost_env.close()

    summary = {
        "checkpoint": str(checkpoint_path),
        "dataset": str(args.dataset),
        "source": args.source,
        "methods": list(args.methods),
        "env_id": metadata["env_id"],
        "env_kwargs": _env_kwargs(metadata, render_mode="rgb_array" if args.video else None),
        "planning_horizon_chunks": args.planning_horizon_chunks,
        "execution_horizon_chunks": args.execution_horizon_chunks,
        "k_schedule": list(args.k_schedule),
        "episodes": rows,
        "by_method": summarize_metrics(rows),
        "code_only_baseline_note": (
            "Code-only waypoint planning is a strong reach baseline and is intentionally "
            "not implemented in this P10 scaffold; do not over-claim reach-only results."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if args.plots:
        _maybe_write_plots(args.output_dir, summary["by_method"])
    if run is not None:
        _log_wandb_summary(run, args=args, rows=rows, summary=summary)

    failures = sum(0 if row["combined_success"] else 1 for row in rows)
    return 0 if args.allow_failure or failures == 0 else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate base DP3, rejection, and reranking on constrained reach."
    )
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--checkpoint", type=Path, default=None)
    checkpoint_group.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-model", choices=["ema", "raw"], default="ema")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--source", choices=["dataset", "fresh"], default="fresh")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--episode-indices", type=int, nargs="+", default=None)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["base", "rejection", "reranking"],
        default=["base", "rejection", "reranking"],
    )
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--post-success-steps", type=int, default=8)
    parser.add_argument("--planning-horizon-chunks", type=int, default=1)
    parser.add_argument("--execution-horizon-chunks", type=int, default=1)
    parser.add_argument("--k-schedule", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--goal-thresh", type=float, default=None)
    parser.add_argument("--avoid-radius", type=float, default=0.08)
    parser.add_argument("--avoid-min-radius", type=float, default=0.025)
    parser.add_argument("--avoid-margin", type=float, default=0.0)
    parser.add_argument("--avoid-weight", type=float, default=1.0)
    parser.add_argument("--gripper-open", type=float, default=0.04)
    parser.add_argument(
        "--match-current-robot-points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cap ghost robot clouds to the current cropped robot-mask count.",
    )
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument(
        "--wandb-mode",
        choices=["disabled", "offline", "online"],
        default=os.environ.get("WANDB_MODE", "disabled"),
    )
    parser.add_argument("--wandb-project", default="pg3d")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-required", action="store_true")
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args(argv)
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.post_success_steps < 0:
        raise ValueError("--post-success-steps must be non-negative")
    validate_planning_horizons(
        planning_horizon_chunks=args.planning_horizon_chunks,
        execution_horizon_chunks=args.execution_horizon_chunks,
    )
    if not args.k_schedule or any(k <= 0 for k in args.k_schedule):
        raise ValueError("--k-schedule values must be positive")
    if args.avoid_radius <= 0.0 or args.avoid_min_radius <= 0.0:
        raise ValueError("avoid radii must be positive")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive")
    return args


def resolve_checkpoint_path(checkpoint: Path | None, checkpoint_dir: Path | None) -> Path:
    """Resolve an explicit checkpoint or the latest step-named checkpoint in a directory."""
    if checkpoint is not None:
        return checkpoint
    if checkpoint_dir is None:
        raise ValueError("checkpoint or checkpoint_dir is required")
    return latest_reach_checkpoint(checkpoint_dir)


def run_eval_episode(
    *,
    sim_env: Any,
    ghost_env: Any,
    policy: SimpleDP3,
    adapter: DP3ChunkPolicyAdapter,
    method: EvalMethod,
    spec: RolloutSpec,
    constraints: list[AvoidRegion],
    action_mode: ActionMode,
    crop_config: PointCloudCropConfig,
    goal_thresh: float,
    output_dir: Path,
    max_steps: int,
    post_success_steps: int,
    planning_horizon_chunks: int,
    execution_horizon_chunks: int,
    k_schedule: tuple[int, ...],
    gripper_open: float,
    match_current_robot_points: bool,
    video: bool,
    rerun: bool,
    video_fps: int,
    decisions_file: Any,
    rng: np.random.Generator,
) -> dict[str, Any]:
    sim_obs, sim_info = sim_env.reset(seed=spec.seed, options={"reconfigure": True})
    sim_entry = rollout_observation_entry(sim_obs, sim_info, env=sim_env, crop_config=crop_config)
    obs_window = make_initial_obs_window(sim_entry, n_obs_steps=int(policy.n_obs_steps))
    target = np.asarray(sim_entry["target_position"], dtype=np.float32).reshape(3)
    scene = scene_context_for_constraints(
        target_position=target,
        constraints=constraints,
        metadata={"method": method, "episode": spec.output_index, "seed": spec.seed},
    )
    path = EpisodePath()
    _append_path(path, sim_entry)
    timeline = [sim_entry.copy()]
    frames = [_frame_to_numpy(sim_env.render())] if video else []
    provider: ManiSkillGhostPandaGeometryProvider | None = None
    world_model: GeometricWorldModel | None = None
    if method != "base":
        provider = ManiSkillGhostPandaGeometryProvider(
            ghost_env,
            task_name=_env_task_name(sim_env),
            crop_bounds=crop_config.bounds,
        )
        provider.reset(seed=spec.seed, options={"reconfigure": True})
        world_model = GeometricWorldModel(provider)

    steps = 0
    replans = 0
    first_success_step: int | None = None
    observed_post_success_steps = 0
    candidate_feasible = 0
    candidate_total = 0
    fallback_count = 0
    terminated_or_truncated = False
    was_training = policy.training
    policy.eval()
    try:
        while steps < max_steps:
            if first_success_step is not None and observed_post_success_steps >= post_success_steps:
                break
            decision = _select_decision(
                method=method,
                adapter=adapter,
                world_model=world_model,
                provider=provider,
                current_entry=sim_entry,
                obs_window=obs_window,
                scene=scene,
                constraints=constraints,
                crop_config=crop_config,
                goal_thresh=goal_thresh,
                planning_horizon_chunks=planning_horizon_chunks,
                k_schedule=k_schedule,
                match_current_robot_points=match_current_robot_points,
                rng=rng,
            )
            replans += 1
            if decision.result is not None:
                candidate_feasible += decision.candidate_feasible
                candidate_total += decision.candidate_total
                if decision.selection_reason == "least_bad_fallback":
                    fallback_count += 1
            _write_decision(
                decisions_file,
                method=method,
                spec=spec,
                replan_index=replans - 1,
                step=steps,
                decision=decision,
            )
            steps_to_execute = min(
                decision.selected_chunk.horizon,
                int(policy.n_action_steps) * execution_horizon_chunks,
                max_steps - steps,
            )
            for policy_action in decision.selected_chunk.actions[:steps_to_execute]:
                sim_action = policy_action_to_sim_action(
                    policy_action,
                    np.asarray(sim_entry["agent_pos"], dtype=np.float32),
                    action_mode=action_mode,
                    sim_action_dim=int(np.prod(sim_env.action_space.shape)),
                    low=getattr(sim_env.action_space, "low", None),
                    high=getattr(sim_env.action_space, "high", None),
                    gripper_open=gripper_open,
                )
                sim_obs, _reward, terminated, truncated, sim_info = sim_env.step(sim_action)
                steps += 1
                sim_entry = rollout_observation_entry(
                    sim_obs,
                    sim_info,
                    env=sim_env,
                    crop_config=crop_config,
                )
                obs_window = append_obs_window(
                    obs_window,
                    sim_entry,
                    n_obs_steps=int(policy.n_obs_steps),
                )
                _append_path(path, sim_entry)
                timeline.append(sim_entry.copy())
                if video:
                    frames.append(_frame_to_numpy(sim_env.render()))
                success = _bool_info(sim_info, "success")
                if success and first_success_step is None:
                    first_success_step = steps
                elif first_success_step is not None:
                    observed_post_success_steps += 1
                terminated_or_truncated = _bool_any(terminated) or _bool_any(truncated)
                if terminated_or_truncated:
                    break
                if (
                    first_success_step is not None
                    and observed_post_success_steps >= post_success_steps
                ):
                    break
            if terminated_or_truncated:
                break
    finally:
        if was_training:
            policy.train()

    video_path = None
    if video:
        video_path = output_dir / "videos" / method / f"episode_{spec.output_index:03d}.mp4"
        save_video(video_path, frames, fps=video_fps)
    rerun_path = None
    if rerun:
        rerun_path = output_dir / "rerun" / method / f"episode_{spec.output_index:03d}.rrd"
        save_rerun_timeline(rerun_path, timeline)
    return episode_metric_row(
        method=method,
        episode=spec.output_index,
        seed=spec.seed,
        path=path,
        constraints=constraints,
        reach_success=first_success_step is not None,
        first_success_step=first_success_step,
        steps=steps,
        replans=replans,
        candidate_feasibility_fraction=candidate_feasibility_fraction(
            candidate_feasible,
            candidate_total,
        ),
        fallback_count=fallback_count,
        video=str(video_path) if video_path is not None else None,
        rerun=str(rerun_path) if rerun_path is not None else None,
    )


def _select_decision(
    *,
    method: EvalMethod,
    adapter: DP3ChunkPolicyAdapter,
    world_model: GeometricWorldModel | None,
    provider: ManiSkillGhostPandaGeometryProvider | None,
    current_entry: Entry,
    obs_window: list[Entry],
    scene: Any,
    constraints: list[AvoidRegion],
    crop_config: PointCloudCropConfig,
    goal_thresh: float,
    planning_horizon_chunks: int,
    k_schedule: tuple[int, ...],
    match_current_robot_points: bool,
    rng: np.random.Generator,
) -> EvalDecisionSummary:
    if method == "base":
        chunk = adapter.sample_action_chunks(obs_window, k=1, rng=rng)[0]
        return EvalDecisionSummary(
            selected_chunk=chunk,
            result=None,
            candidate_feasible=0,
            candidate_total=0,
            selection_reason=None,
        )
    if world_model is None or provider is None:
        raise RuntimeError("controller methods require a world model and ghost provider")
    if match_current_robot_points:
        provider.set_robot_point_budget_from_mask(
            np.asarray(current_entry["robot_mask"], dtype=bool),
            point_valid_mask=np.asarray(current_entry["point_valid_mask"], dtype=bool),
        )
    controller_input = ControllerInput(
        observation=entry_to_world_model_observation(current_entry),
        scene=scene,
        policy_input=obs_window,
    )
    if planning_horizon_chunks == 1:
        controller_cls = RejectionController if method == "rejection" else RerankingController
        result = controller_cls(
            policy=adapter,
            world_model=world_model,
            constraints=constraints,
            k_schedule=k_schedule,
        ).select(controller_input, rng=rng)
    else:
        result = _select_multichunk(
            method=method,
            adapter=adapter,
            world_model=world_model,
            current_entry=current_entry,
            obs_window=obs_window,
            scene=scene,
            constraints=constraints,
            crop_config=crop_config,
            goal_thresh=goal_thresh,
            planning_horizon_chunks=planning_horizon_chunks,
            k_schedule=k_schedule,
            rng=rng,
        )
    feasible = sum(1 for candidate in result.candidates if candidate.feasible)
    return EvalDecisionSummary(
        selected_chunk=result.action_chunk,
        result=result,
        candidate_feasible=feasible,
        candidate_total=len(result.candidates),
        selection_reason=result.selection_reason,
    )


def _select_multichunk(
    *,
    method: EvalMethod,
    adapter: DP3ChunkPolicyAdapter,
    world_model: GeometricWorldModel,
    current_entry: Entry,
    obs_window: list[Entry],
    scene: Any,
    constraints: list[AvoidRegion],
    crop_config: PointCloudCropConfig,
    goal_thresh: float,
    planning_horizon_chunks: int,
    k_schedule: tuple[int, ...],
    rng: np.random.Generator,
) -> ControllerResult:
    candidates: list[CandidateDiagnostics] = []
    attempted: list[int] = []
    for k in k_schedule:
        attempted.append(k)
        batch = _build_multichunk_candidates(
            adapter=adapter,
            world_model=world_model,
            current_entry=current_entry,
            obs_window=obs_window,
            scene=scene,
            constraints=constraints,
            crop_config=crop_config,
            goal_thresh=goal_thresh,
            planning_horizon_chunks=planning_horizon_chunks,
            attempted_k=k,
            start_index=len(candidates),
            rng=rng,
        )
        candidates.extend(batch)
        feasible = [candidate for candidate in candidates if candidate.feasible]
        if feasible:
            if method == "rejection":
                selected = feasible[0]
                return _controller_result(selected, candidates, attempted, "first_feasible")
            selected = min(feasible, key=lambda candidate: candidate.total_score)
            return _controller_result(selected, candidates, attempted, "best_feasible")
    if not candidates:
        raise RuntimeError("policy returned no candidate action chunks")
    selected = min(candidates, key=lambda candidate: candidate.total_score)
    return _controller_result(selected, candidates, attempted, "least_bad_fallback")


def _build_multichunk_candidates(
    *,
    adapter: DP3ChunkPolicyAdapter,
    world_model: GeometricWorldModel,
    current_entry: Entry,
    obs_window: list[Entry],
    scene: Any,
    constraints: list[AvoidRegion],
    crop_config: PointCloudCropConfig,
    goal_thresh: float,
    planning_horizon_chunks: int,
    attempted_k: int,
    start_index: int,
    rng: np.random.Generator,
) -> list[CandidateDiagnostics]:
    first_chunks = adapter.sample_action_chunks(obs_window, k=attempted_k, rng=rng)
    branch_rollouts: list[ImaginedRollout] = []
    for branch_idx, first_chunk in enumerate(first_chunks):
        branch_entry = _copy_entry(current_entry)
        branch_window = _copy_window(obs_window)
        rollouts: list[ImaginedRollout] = []
        next_chunk = first_chunk
        for chunk_idx in range(planning_horizon_chunks):
            if chunk_idx > 0:
                next_chunk = adapter.sample_action_chunks(branch_window, k=1, rng=rng)[0]
            rollout = world_model.imagine(
                entry_to_world_model_observation(branch_entry),
                next_chunk,
                metadata={"branch": branch_idx, "chunk_index": chunk_idx},
            )
            rollouts.append(rollout)
            for step_idx in range(rollout.action_chunk.horizon):
                branch_entry = world_model_entry_from_rollout_step(
                    rollout,
                    step_idx,
                    previous_entry=branch_entry,
                    crop_config=crop_config,
                    goal_thresh=goal_thresh,
                )
                branch_window = append_obs_window(
                    branch_window,
                    branch_entry,
                    n_obs_steps=int(adapter.policy.n_obs_steps),
                )
        branch_rollouts.append(
            concatenate_rollouts(
                rollouts,
                metadata={"candidate_index": start_index + branch_idx},
            )
        )

    chunks = [rollout.action_chunk for rollout in branch_rollouts]
    consensus = consensus_deviations(chunks)
    return [
        _candidate_diagnostics(
            index=start_index + idx,
            attempted_k=attempted_k,
            action_chunk=rollout.action_chunk,
            rollout=rollout,
            scene=scene,
            constraints=constraints,
            consensus_deviation=consensus[idx],
        )
        for idx, rollout in enumerate(branch_rollouts)
    ]


def _candidate_diagnostics(
    *,
    index: int,
    attempted_k: int,
    action_chunk: ActionChunk,
    rollout: ImaginedRollout,
    scene: Any,
    constraints: list[AvoidRegion],
    consensus_deviation: float,
) -> CandidateDiagnostics:
    constraint_costs: dict[str, float] = {}
    constraint_satisfied: dict[str, bool] = {}
    for constraint_idx, constraint in enumerate(constraints):
        label = f"{constraint_idx}:{constraint.name}"
        costs = constraint.cost(rollout, scene)
        for key, value in costs.items():
            constraint_costs[_unique_cost_key(constraint_costs, key)] = float(value)
        constraint_satisfied[label] = bool(constraint.satisfied(rollout, scene))
    feasible = all(constraint_satisfied.values()) if constraint_satisfied else True
    distance = goal_distance(rollout, scene.target_position)
    smoothness = trajectory_smoothness(rollout, order=2)
    penalty = primary_constraint_penalty(constraint_costs)
    weights = ScoreWeights()
    total_score = (
        weights.constraint * penalty
        + weights.goal_distance * (0.0 if distance is None else distance)
        + weights.smoothness * smoothness
        + weights.consensus * consensus_deviation
    )
    return CandidateDiagnostics(
        index=index,
        attempted_k=attempted_k,
        action_chunk=action_chunk,
        rollout=rollout,
        constraint_costs=constraint_costs,
        constraint_satisfied=constraint_satisfied,
        feasible=feasible,
        goal_distance=distance,
        constraint_penalty=penalty,
        smoothness=smoothness,
        consensus_deviation=consensus_deviation,
        policy_surrogate=None,
        total_score=float(total_score),
    )


def _controller_result(
    selected: CandidateDiagnostics,
    candidates: list[CandidateDiagnostics],
    attempted: list[int],
    reason: str,
) -> ControllerResult:
    selected.selection_reason = reason
    return ControllerResult(
        selected=selected,
        candidates=candidates,
        attempted_k_values=list(attempted),
        selection_reason=reason,
    )


def _write_decision(
    decisions_file: Any,
    *,
    method: EvalMethod,
    spec: RolloutSpec,
    replan_index: int,
    step: int,
    decision: EvalDecisionSummary,
) -> None:
    result = decision.result
    row = {
        "method": method,
        "episode": spec.output_index,
        "seed": spec.seed,
        "replan_index": replan_index,
        "step": step,
        "selection_reason": decision.selection_reason,
        "candidate_feasible": decision.candidate_feasible,
        "candidate_total": decision.candidate_total,
    }
    if result is not None:
        scores = [candidate.total_score for candidate in result.candidates]
        row.update(
            {
                "attempted_k_values": result.attempted_k_values,
                "selected_index": result.selected.index,
                "selected_score": result.selected.total_score,
                "selected_feasible": result.selected.feasible,
                "selected_goal_distance": result.selected.goal_distance,
                "selected_constraint_penalty": result.selected.constraint_penalty,
                "selected_smoothness": result.selected.smoothness,
                "selected_constraint_costs": result.selected.constraint_costs,
                "score_min": min(scores) if scores else None,
                "score_mean": float(np.mean(scores)) if scores else None,
            }
        )
    decisions_file.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
    decisions_file.flush()


def _episode_constraints(
    env: Any,
    *,
    spec: RolloutSpec,
    crop_config: PointCloudCropConfig,
    args: argparse.Namespace,
) -> list[AvoidRegion]:
    obs, info = env.reset(seed=spec.seed, options={"reconfigure": True})
    entry = rollout_observation_entry(obs, info, env=env, crop_config=crop_config)
    start_tcp = np.asarray(entry["tcp_pose"], dtype=np.float32).reshape(-1)[:3]
    target = np.asarray(entry["target_position"], dtype=np.float32).reshape(3)
    constraint = direct_path_avoid_region(
        start_tcp=start_tcp,
        target_position=target,
        config=AvoidOverlayConfig(
            radius=args.avoid_radius,
            min_radius=args.avoid_min_radius,
            margin=args.avoid_margin,
            weight=args.avoid_weight,
        ),
    )
    return [constraint]


def _repeat_obs_window_to_torch(
    window: list[Entry],
    *,
    k: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch = obs_window_to_torch(window, device=device)
    return {
        key: value.repeat((k, *([1] * (value.ndim - 1))))
        for key, value in batch.items()
    }


def _append_path(path: EpisodePath, entry: Entry) -> None:
    tcp = np.asarray(entry["tcp_pose"], dtype=np.float32).reshape(-1)[:3]
    path.append(
        tcp_position=tcp,
        q=np.asarray(entry["agent_pos"], dtype=np.float32),
        target_distance=float(np.asarray(entry["final_distance"], dtype=np.float32).reshape(-1)[0]),
    )


def _env_kwargs(metadata: dict[str, Any], *, render_mode: str | None) -> dict[str, Any]:
    env_kwargs = dict(metadata["env_kwargs"])
    env_kwargs["obs_mode"] = "pointcloud"
    env_kwargs["num_envs"] = 1
    if render_mode is None:
        env_kwargs.pop("render_mode", None)
    else:
        env_kwargs["render_mode"] = render_mode
    return env_kwargs


def _copy_entry(entry: Entry) -> Entry:
    return {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in entry.items()
    }


def _copy_window(window: list[Entry]) -> list[Entry]:
    return [_copy_entry(entry) for entry in window]


def _action_mode(value: str) -> ActionMode:
    if value not in {"abs_joint", "delta_joint"}:
        raise ValueError(f"unsupported action_mode {value!r}")
    return value  # type: ignore[return-value]


def _env_task_name(env: Any) -> str:
    unwrapped = getattr(env, "unwrapped", env)
    spec = getattr(unwrapped, "spec", None)
    return str(getattr(spec, "id", "unknown"))


def _unique_cost_key(costs: dict[str, float], key: str) -> str:
    if key not in costs:
        return key
    suffix = 1
    while f"{key}#{suffix}" in costs:
        suffix += 1
    return f"{key}#{suffix}"


def _init_wandb(
    args: argparse.Namespace,
    *,
    metadata: dict[str, Any],
    checkpoint_path: Path,
) -> Any | None:
    if args.wandb_mode == "disabled":
        return None
    try:
        import wandb

        return wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            mode=args.wandb_mode,
            config={
                "dataset": str(args.dataset),
                "checkpoint": str(checkpoint_path),
                "env_id": metadata.get("env_id"),
                "methods": list(args.methods),
                "planning_horizon_chunks": args.planning_horizon_chunks,
                "execution_horizon_chunks": args.execution_horizon_chunks,
                "k_schedule": list(args.k_schedule),
                "command": "scripts/eval_constrained_reach.py",
            },
        )
    except Exception as exc:
        if args.wandb_required:
            raise
        print(
            f"warning: W&B init failed, continuing without W&B: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return None


def _log_wandb_summary(
    run: Any,
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    try:
        import wandb

        metrics: dict[str, Any] = {}
        for method, method_summary in summary["by_method"].items():
            for key, value in method_summary.items():
                if isinstance(value, (int, float)) and value is not None:
                    metrics[f"eval/{method}/{key}"] = value
        columns = sorted({key for row in rows for key in row.keys()})
        table = wandb.Table(columns=columns)
        for row in rows:
            table.add_data(*[_jsonable(row.get(column)) for column in columns])
        metrics["eval/episodes"] = table
        if args.video:
            for row in rows:
                video = row.get("video")
                if video and Path(str(video)).exists():
                    metrics[f"eval_video/{row['method']}/episode_{int(row['episode']):03d}"] = (
                        wandb.Video(str(video), fps=args.video_fps, format="mp4")
                    )
        run.log(metrics)
    except Exception as exc:
        if args.wandb_required:
            raise
        print(
            f"warning: W&B summary logging failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )


def _maybe_write_plots(output_dir: Path, by_method: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(
            f"warning: matplotlib unavailable for plots: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return

    methods = list(by_method.keys())
    rates = [float(by_method[method].get("combined_success_rate", 0.0)) for method in methods]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(methods, rates, color=["#777777", "#2c7fb8", "#41ab5d"][: len(methods)])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Combined success rate")
    ax.set_title("Constrained reach eval")
    fig.tight_layout()
    path = output_dir / "summary_rates.png"
    fig.savefig(path)
    plt.close(fig)


def _format_optional(value: Any) -> str:
    if value is None:
        return "nan"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(numeric):
        return "nan"
    return f"{numeric:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
