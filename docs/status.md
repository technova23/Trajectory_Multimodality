# pg3d status

Last updated: 2026-05-17

## Current objective

Bootstrap a sim-only research codebase for programmatic geometric guidance of 3D diffusion policies. The first MVP is constrained reaching in ManiSkill/SAPIEN:

- base policy: DP3-style point-cloud diffusion policy,
- simulator: ManiSkill/SAPIEN, with built-in task smoke plus custom narrow/medium reach tasks,
- action representation: start with absolute joint target chunks; keep delta joint chunks as fallback,
- world model: kinematic robot-geometry point-cloud imagination from joint-action chunks,
- first constraint: `avoid_region` over the end-effector path,
- first composition operator: candidate rejection/reranking, not energy guidance.

## Current phase

Simulator migration to ManiSkill/SAPIEN is complete in the active code path. The repo has a
pg3d-native simulation-free DP3 slice under `pg3d/policies/dp3` with synthetic import, inference,
and training-step smoke tests. ManiSkill is tracked as a pinned optional uv extra, while base `pg3d`
imports stay simulator-free. A small non-rendering ManiSkill smoke script validates a built-in
`PickCube-v1` environment. The first observation adapter now targets Franka/Panda `PickCube-v1`
state and point-cloud observations, including segmentation-derived robot/object masks when a live
ManiSkill env context is available. P05 adds custom `PG3DReach-Narrow-v0` /
`PG3DReach-Medium-v0` tasks plus a smoke-scale Zarr dataset writer for DP3-compatible reach data.
P06 adds a simulation-free reach Zarr sequence loader for pg3d-native DP3, plus CPU smoke
training/eval scripts with optional W&B metrics and histogram logging. P06 now also has a
closed-loop policy rollout script that loads a trained reach checkpoint, runs it in live
`PG3DReach-*` ManiSkill environments, and writes MP4 videos, Rerun timelines, and JSON metrics for
dataset-seed or fresh-seed rollouts. The current detour adds post-success hold-pose data to the
reach dataset writer and upgrades the trainer with validation, cosine warmup, gradient clipping,
EMA checkpoint state, directory-based periodic checkpoints, best-effort W&B checkpoint rollout
videos, and richer diagnostics for stable non-trivial training runs.

## Immediate next steps

1. Scale the exercised `PG3DReach-Narrow-v0` 100-episode path to a 500-episode dataset with
   `hold_steps=8`, replay a fixed subset, and inspect MP4/Rerun artifacts.
2. Train the moderate 5090 DP3 recipe on the 500-episode dataset, inspecting W&B validation
   metrics plus dataset-seed/fresh-seed policy rollout videos from periodic checkpoints.
3. Start the kinematic point-cloud world model/FK compositor work using saved reach trajectories
   as visual sanity checks.

## Active risks

- DP3 upstream was designed around older Python/CUDA assumptions; pg3d now ports only the
  simulation-free model core and avoids upstream benchmark dependencies.
- ManiSkill v3 is a fast-moving stack; keep the adapter isolated and commands pinned in runbooks.
- Rendering and point-cloud observation modes may require Vulkan/driver setup beyond the
  non-rendering `obs_mode="state"` smoke.
- Optional Rerun visualization is pinned to `rerun-sdk==0.22.1` while pg3d remains on NumPy 1.x.
- Reach is useful for mechanism validation, but code-only planners may be strong; avoid over-claiming from reach-only results.
- The kinematic point-cloud world model is the novel project pivot and should be validated visually early.
- New clones and fresh virtualenvs must sync the `maniskill` optional extra before running
  ManiSkill smoke checks.

## Decisions already made

- Project/repo/package name: `pg3d` for now.
- Sim-only for this phase; real robot hardware code is out of scope.
- ManiSkill/SAPIEN is the primary simulator.
- RLBench/PyRep/CoppeliaSim are deprecated and removed from active dependencies/backends.
- ManiSkill should be installed as an optional uv dependency, not carried as a submodule.
- DP3 is the only base policy for P0; RISE is deferred.
- DP3 runtime code should live in `pg3d/policies/dp3`; `external/dp3` is a temporary reference
  submodule during migration.
- Start with reach, then move to pick-and-place, then place-into-container.
- Start with handwritten constraints; LLM-generated constraints are later.
- Start with reranking/rejection; energy guidance is later.
- Use W&B from day one, but keep offline/debug modes available.
- ManiSkill observations use typed pg3d dataclasses and keep policy-visible point clouds/agent state
  separate from simulator ground truth and eval/debug masks.
- Robot masks are first-class observation metadata for the world model.
- Franka/Panda is the first robot target for built-in ManiSkill smoke and observation adaptation.
- Reach dataset DP3 action labels are 7D Panda arm joint targets/deltas; full simulator actions are
  stored separately for replay.
- Reach dataset replay can now save MP4 videos and per-episode Rerun timeline artifacts.
- DP3 reach training consumes only point cloud, agent position, and action arrays; simulator
  ground-truth/debug arrays stay out of policy batches.
- Standalone DP3 policy rollout visualization is local-first: MP4, Rerun `.rrd`, `metrics.jsonl`,
  and `summary.json`. The trainer can also upload a small configurable set of checkpoint-time MP4
  rollout videos to W&B when W&B and ManiSkill rendering are available.
- Reach datasets should include one DP3 action chunk of post-success hold-pose data by default so
  terminal policy chunks learn to stay at the goal.
- Stable DP3 reach checkpoints should prefer EMA weights for eval/rollout when present.
- DP3 reach training checkpoints are now directory-based: periodic files use `step_XXXXXXXX.pt`
  and final files use `final_step_XXXXXXXX.pt`.

## Latest work log

See `docs/worklog/`.

- Simulator choice is recorded in `docs/adr/0002-maniskill-primary-simulator.md`.
- Observation schema and mask policy are recorded in
  `docs/adr/0008-observation-schema-and-masks.md`.
- Current canonical setup command:
  `uv sync --extra cu129 --extra maniskill --group dev --group notebooks`.
- Optional visualization setup command:
  `uv sync --extra cu129 --extra maniskill --extra viz --group dev --group notebooks`.
- Current validation: `uv lock --check`, `make smoke`, `make test`, `make lint`,
  `make gpu-check`, `make maniskill-check`, and the state/point-cloud/MP4/Rerun observation
  artifact scripts pass on the RTX 5090 workstation environment. P05 reach dataset smoke/replay
  visualization plus P06 DP3 reach training/eval/rollout smoke validation are recorded in the
  worklog. The hold-tail dataset/training stability pass has also been validated with pure tests, a
  5-demo hold dataset smoke, short CPU training/eval, offline W&B outside the sandbox, and one
  dataset/fresh live rollout smoke. The intermediate-checkpoint pass adds pure tests for
  step-named checkpoint paths, periodic/final checkpoint writing, mixed rollout-video seed
  selection, lazy training imports, and non-fatal checkpoint-rollout failures. It also validates
  a two-step checkpoint-directory smoke and an outside-sandbox offline W&B checkpoint-video smoke.
  A focused cleanup pass then consolidated duplicate JSON/array/device/checkpoint helpers without
  changing scientific behavior, refreshed the custom reach setup notes, improved `make clean` for
  nested `__pycache__` directories, and passed ruff, 48 pytest tests, smoke imports, and
  `git diff --check`.
