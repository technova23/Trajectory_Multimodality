# Prompt P05 — Reach demo generation and dataset writer

Goal:
Generate/replay RLBench ReachTarget demonstrations and write a dataset suitable for the DP3 fork.

Context to read first:
- `AGENTS.md`
- `docs/milestones.md` M2
- `docs/architecture/system_architecture.md`
- `docs/adr/0004-action-representation.md`
- DP3 data loading examples in `external/dp3` if available.

Constraints:
- Start with small smoke datasets: 3-5 demos.
- Do not run large generation jobs.
- Support absolute joint target chunks first and delta joint chunks as fallback.
- Save enough metadata for replay: seed, task variant, action mode, camera config, submodule commit hashes.

Tasks:
1. Inspect DP3 dataset expectations in the fork/submodule.
2. Propose the minimal dataset schema for RLBench reach.
3. Implement a writer for observation/action sequences.
4. Implement a replay sanity script.
5. Add shape/schema tests independent of RLBench where possible.
6. Update docs/status, runbooks, and worklog.

Done when:
- A smoke dataset can be generated or the exact missing simulator blocker is documented.
- The dataset schema is documented clearly enough for DP3 integration.
