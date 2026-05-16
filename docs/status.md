# pg3d status

Last updated: 2026-05-16

## Current objective

Bootstrap a sim-only research codebase for programmatic geometric guidance of 3D diffusion policies. The first MVP is constrained reaching in RLBench:

- base policy: DP3-style point-cloud diffusion policy,
- simulator: RLBench `ReachTarget`, possibly narrowed/customized,
- action representation: start with absolute joint target chunks; keep delta joint chunks as fallback,
- world model: kinematic robot-geometry point-cloud imagination from joint-action chunks,
- first constraint: `avoid_region` over the end-effector path,
- first composition operator: candidate rejection/reranking, not energy guidance.

## Current phase

M1 RLBench ReachTarget smoke setup. The repo has a pg3d-native simulation-free DP3 slice under
`pg3d/policies/dp3` with synthetic import, inference, and training-step smoke tests. RLBench is now
tracked as an optional uv extra rather than a submodule, and the first `ReachTarget` smoke script
fails early with actionable setup errors when RLBench, PyRep, or CoppeliaSim are missing.

## Immediate next steps

1. Install CoppeliaSim 4.1.0 and run `uv sync --extra cu129 --extra rlbench --group dev`.
2. Run `uv run python scripts/rlbench_smoke_reach.py --headless true` on the workstation.
3. Build the RLBench observation/dataset adapter against the pg3d DP3 schema.
4. Extend the pg3d-native DP3 slice from synthetic smoke tests to a generic zarr dataset and
   one-step trainer smoke.

## Active risks

- DP3 upstream was designed around older Python/CUDA assumptions; pg3d now ports only the
  simulation-free model core and avoids upstream benchmark dependencies.
- RLBench/PyRep/CoppeliaSim installation may constrain Python version or require system-package fixes.
- Reach is useful for mechanism validation, but code-only planners may be strong; avoid over-claiming from reach-only results.
- The kinematic point-cloud world model is the novel project pivot and should be validated visually early.
- This Codex sandbox cannot see a CUDA device, but `make gpu-check` and the CUDA DP3 smoke pass
  from the user's local pg3d terminal on the RTX 5090 workstation.
- RLBench/PyRep/CoppeliaSim are not installed/configured in the current Codex environment, so the
  first ReachTarget smoke can only validate dependency reporting here until workstation setup runs.

## Decisions already made

- Project/repo/package name: `pg3d` for now.
- Sim-only for this phase; real robot hardware code is out of scope.
- RLBench is the primary simulator.
- RLBench should be installed as an optional uv dependency, not carried as a submodule.
- DP3 is the only base policy for P0; RISE is deferred.
- DP3 runtime code should live in `pg3d/policies/dp3`; `external/dp3` is a temporary reference
  submodule during migration.
- Start with reach, then move to pick-and-place, then place-into-container.
- Start with handwritten constraints; LLM-generated constraints are later.
- Start with reranking/rejection; energy guidance is later.
- Use W&B from day one, but keep offline/debug modes available.

## Latest work log

See `docs/worklog/`.
