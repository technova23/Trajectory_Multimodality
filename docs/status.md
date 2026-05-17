# pg3d status

Last updated: 2026-05-17

## Current objective

Bootstrap a sim-only research codebase for programmatic geometric guidance of 3D diffusion policies. The first MVP is constrained reaching in ManiSkill/SAPIEN:

- base policy: DP3-style point-cloud diffusion policy,
- simulator: ManiSkill/SAPIEN, starting with built-in task smoke and then a narrow reach task if needed,
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

## Immediate next steps

1. Inspect generated `PG3DReach-Narrow-v0` smoke datasets with MP4/Rerun replay artifacts and
   verify point-cloud crop/mask quality.
2. Run P06: load the ManiSkill reach Zarr dataset into pg3d-native DP3 and start a smoke training
   step.
3. Use the saved reach trajectories to sanity-check future FK/world-model rollouts.

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
  artifact scripts pass on the RTX 5090 workstation environment. P05 reach dataset smoke and replay
  visualization validation is recorded in the worklog.
