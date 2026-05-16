# Prompt P07 — Kinematic point-cloud world model v0

Goal:
Implement robot-only kinematic point-cloud imagination from joint-action chunks.

Context to read first:
- `AGENTS.md`
- `docs/milestones.md` M4
- `docs/architecture/system_architecture.md`
- `docs/adr/0005-world-model-first-class-module.md`
- `docs/adr/0004-action-representation.md`

Constraints:
- Start with pure Python/numpy/torch utilities where possible.
- Keep simulator-specific robot mesh loading behind an interface.
- Do not implement object attachment yet.
- Do not require RLBench import for pure unit tests.

Tasks:
1. Define `ActionChunk`, `ImaginedRollout`, and world-model interfaces if not already present.
2. Implement chunk interpretation for absolute and delta joint targets.
3. Implement FK interface; if real robot FK is not available yet, create a test double and TODO for RLBench/Panda FK integration.
4. Implement point-cloud compositor: delete current robot points via mask, insert future robot/link points.
5. Add synthetic tests for chunk integration and compositor behavior.
6. Add visualization script stub for imagined rollouts.
7. Update docs/status and worklog.

Done when:
- Synthetic world-model tests pass.
- The module API is ready to plug into RLBench/Panda FK once available.
