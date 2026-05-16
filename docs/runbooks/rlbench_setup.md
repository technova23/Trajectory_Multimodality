# RLBench setup runbook

Status: placeholder until M1.

Known constraints:

- PyRep requires CoppeliaSim 4.1.
- PyRep communication is Linux-focused.
- The initial task is RLBench `ReachTarget`.

Fill this in with exact working commands once setup succeeds on the target workstation.

## Target commands to create during M1

```bash
uv run python scripts/rlbench_smoke_reach.py --headless false
uv run python scripts/rlbench_smoke_reach.py --headless true
uv run python scripts/rlbench_save_observation.py --task reach_target --out outputs/smoke/reach_obs
```

## Troubleshooting log

Append specific failures/fixes here rather than burying them in chat history.
