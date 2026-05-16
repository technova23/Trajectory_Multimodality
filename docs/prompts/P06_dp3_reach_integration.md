# Prompt P06 — DP3 integration for RLBench ReachTarget

Goal:
Modify the private DP3 fork to train/evaluate on the pg3d ReachTarget dataset.

Context to read first:
- `AGENTS.md`
- `docs/milestones.md` M3
- `docs/adr/0003-dp3-p0-policy.md`
- `docs/adr/0004-action-representation.md`
- `docs/runbooks/dependency_mirroring.md`
- DP3 README/install/custom task docs in `external/dp3`.

Constraints:
- This task may edit `external/dp3` only if I explicitly started Codex from a branch intended for DP3 modifications.
- Avoid dependency changes that reinstall old torch/gym packages into the main pg3d env.
- Prefer simple DP3 configs first.
- Do not run long training; run data-loading/config smoke only unless explicitly asked.

Tasks:
1. Inspect DP3 env runner, dataset, and task config patterns.
2. Add the minimal dataset loader/config needed for ReachTarget action chunks.
3. Add a training config for Reach-Narrow.
4. Add an evaluation adapter/script stub if needed.
5. Document exact training/eval commands in `docs/runbooks/commands.md`.
6. Update `docs/status.md` and worklog.

Done when:
- DP3 can load the ReachTarget dataset and start a smoke training step.
- Any DP3 fork changes are committed separately in the submodule branch.
