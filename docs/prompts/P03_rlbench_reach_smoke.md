# Prompt P03 — RLBench ReachTarget smoke test

Goal:
Create the first RLBench smoke script for `ReachTarget`.

Context to read first:
- `AGENTS.md`
- `docs/status.md`
- `docs/milestones.md` M1
- `docs/runbooks/rlbench_setup.md`
- `docs/adr/0001-reach-first-mvp.md`
- `docs/adr/0002-rlbench-primary-simulator.md`

Constraints:
- Keep RLBench/PyRep imports lazy.
- Do not modify RLBench or PyRep unless explicitly requested.
- Script should fail with a clear message if CoppeliaSim/PyRep/RLBench are missing.
- Support a `--headless` flag.
- Do not implement full observation adapter yet.

Tasks:
1. Add `scripts/rlbench_smoke_reach.py` that imports RLBench lazily, launches `ReachTarget`, resets, prints observation keys/shapes, takes a small number of steps if safe, and shuts down cleanly.
2. Add a small utility for dependency/error reporting if useful.
3. Update `docs/runbooks/rlbench_setup.md` with exact commands and observed failure modes.
4. Update `docs/status.md` and add a worklog entry.

Done when:
- The script runs or produces actionable setup errors.
- No package import-time dependency on RLBench is introduced.
