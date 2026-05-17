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
ManiSkill env context is available.

## Immediate next steps

1. Run P05: extend the ManiSkill adapter into the reach/custom-task dataset writer.
2. Use the saved P04 point-cloud artifacts to sanity-check robot/cube masks before world-model work.
3. Extend the pg3d-native DP3 slice from synthetic smoke tests to a generic zarr dataset and
   one-step trainer smoke.

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
  artifact scripts pass on the RTX 5090 workstation environment.
