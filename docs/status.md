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

P00 repository scaffold bootstrapped. The repo now has the planned lightweight package
namespace, import smoke tests, and Make targets for local checks. Simulator and DP3 logic remain
deferred. The configured workstation setup path is `uv sync --extra cu129 --group dev`.

## Immediate next steps

1. Create private mirrors of DP3 and any other dependency repos we expect to patch.
2. Add private DP3 mirror as `external/dp3` submodule.
3. Verify `uv sync --extra cu129 --group dev` on the RTX 5090 workstation.
4. Verify PyTorch CUDA availability with `make gpu-check`.
5. Install RLBench/PyRep/CoppeliaSim and launch `ReachTarget`.

## Active risks

- DP3 upstream was designed around older Python/CUDA assumptions; the fork may need dependency cleanup for Python 3.11 and CUDA 12.9.
- RLBench/PyRep/CoppeliaSim installation may constrain Python version or require system-package fixes.
- Reach is useful for mechanism validation, but code-only planners may be strong; avoid over-claiming from reach-only results.
- The kinematic point-cloud world model is the novel project pivot and should be validated visually early.
- This Codex sandbox cannot see a CUDA device, so `make gpu-check` must be rerun on the RTX 5090 workstation.

## Decisions already made

- Project/repo/package name: `pg3d` for now.
- Sim-only for this phase; real robot hardware code is out of scope.
- RLBench is the primary simulator.
- DP3 is the only base policy for P0; RISE is deferred.
- Start with reach, then move to pick-and-place, then place-into-container.
- Start with handwritten constraints; LLM-generated constraints are later.
- Start with reranking/rejection; energy guidance is later.
- Use W&B from day one, but keep offline/debug modes available.

## Latest work log

See `docs/worklog/`.
