# Prompt P04 — RLBench observation adapter

Goal:
Implement the first `pg3d.envs.rlbench_adapter` observation adapter for ReachTarget.

Context to read first:
- `AGENTS.md`
- `docs/architecture/system_architecture.md`
- `docs/milestones.md` M1
- `docs/adr/0001-reach-first-mvp.md`
- Existing smoke script from P03.

Constraints:
- Policy-visible inputs must be separated from sim GT/eval-only fields.
- Keep RLBench imports lazy.
- Do not train anything.
- Use simple typed dataclasses/Pydantic models.

Tasks:
1. Create core data models: `RobotState`, `Observation`, `SimGroundTruth`.
2. Implement adapter functions that convert an RLBench observation into these objects.
3. Extract point cloud, optional RGB/features, robot state, robot mask if available, and target position as sim GT/eval context.
4. Add `scripts/rlbench_save_observation.py` that saves one adapted observation summary and optional plot/artifact.
5. Add tests for pure data-model/schema logic; simulator tests should skip if RLBench is unavailable.
6. Update docs/worklog.

Done when:
- One ReachTarget observation can be adapted and saved.
- Data shape conventions are documented.
