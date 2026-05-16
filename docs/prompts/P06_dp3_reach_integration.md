# Prompt P06 — DP3 integration for RLBench ReachTarget

Goal:
Integrate the pg3d-native DP3 policy core with the pg3d ReachTarget dataset.

Context to read first:
- `AGENTS.md`
- `docs/milestones.md` M3
- `docs/adr/0003-dp3-p0-policy.md`
- `docs/adr/0004-action-representation.md`
- `docs/runbooks/dependency_mirroring.md`
- `pg3d/policies/dp3/`
- DP3 README/install/custom task docs in `external/dp3` as reference only.

Constraints:
- Do not use `external/dp3` as a runtime import path; it is reference material during migration.
- Do not edit `external/dp3` unless explicitly asked.
- Avoid dependency changes that reinstall old torch/gym packages into the main pg3d env.
- Prefer simple DP3 configs first.
- Do not run long training; run data-loading/config smoke only unless explicitly asked.

Tasks:
1. Inspect the pg3d-native DP3 policy core and relevant DP3 reference patterns.
2. Add the minimal generic dataset loader/config needed for ReachTarget action chunks.
3. Add a training config or script for Reach-Narrow.
4. Add an evaluation adapter/script stub if needed.
5. Document exact training/eval commands in `docs/runbooks/commands.md`.
6. Update `docs/status.md` and worklog.

Done when:
- DP3 can load the ReachTarget dataset and start a smoke training step.
- Runtime imports come from `pg3d.policies.dp3`.
