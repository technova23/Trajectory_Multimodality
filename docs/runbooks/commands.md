# Commands runbook

Update this file whenever setup, run, test, or eval commands change.

## Create environment

RTX 5090 / CUDA 12.9 path:

```bash
uv sync --extra cu129 --group dev
```

CPU/debug path:

```bash
uv sync --extra cpu --group dev
```

## Basic checks

```bash
make test
make lint
make smoke
make gpu-check
```

Equivalent direct commands:

```bash
uv run pytest
uv run ruff check .
uv run python scripts/smoke_imports.py
uv run python scripts/check_gpu.py
```

## Submodules

```bash
git submodule update --init --recursive
```

When cloning a new workstation:

```bash
git clone --recurse-submodules git@github.com:YOUR_ORG/pg3d.git
cd pg3d
uv sync --extra cu129 --group dev
```

## RLBench/PyRep smoke placeholder

These commands should be replaced once RLBench setup is implemented:

```bash
# Example only; update after implementing M1.
uv run python scripts/rlbench_smoke_reach.py --headless false
uv run python scripts/rlbench_save_observation.py --task reach_target --out outputs/smoke/reach_obs
```

## W&B

```bash
wandb login
# or for offline/debug:
export WANDB_MODE=offline
```

## Long-running training policy

Do not run long training jobs from Codex unless explicitly instructed. Codex should prepare commands/configs and run only smoke-scale jobs by default.
