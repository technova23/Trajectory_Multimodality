# Commands runbook

Update this file whenever setup, run, test, or eval commands change.

## Create environment

RTX 5090 / CUDA 12.9 path:

```bash
uv sync --extra cu129 --group dev
```

RTX 5090 / CUDA 12.9 path with ManiSkill:

```bash
uv sync --extra cu129 --extra maniskill --group dev --group notebooks
```

CPU/debug path:

```bash
uv sync --extra cpu --group dev
```

CPU/debug path with ManiSkill:

```bash
uv sync --extra cpu --extra maniskill --group dev
```

## Basic checks

```bash
make test
make lint
make smoke
make gpu-check
make maniskill-check
```

Equivalent direct commands:

```bash
uv run pytest
uv run ruff check .
uv run python scripts/smoke_imports.py
uv run python scripts/check_gpu.py
uv run python scripts/check_maniskill.py
```

## DP3 policy smoke

The pg3d-native DP3 slice is tested with synthetic point-cloud/state/action data:

```bash
uv run python scripts/smoke_dp3_policy.py --device cpu
uv run python scripts/smoke_dp3_policy.py --device cuda
```

Use the CPU smoke in sandbox/CI contexts. Use the CUDA smoke on the RTX 5090 workstation after
`make gpu-check` succeeds.

## Submodules

```bash
git submodule update --init --recursive
```

`external/dp3` is a private reference submodule. Runtime code should import
`pg3d.policies.dp3`, not `external/dp3`.

When cloning a new workstation:

```bash
git clone --recurse-submodules git@github.com:YOUR_ORG/pg3d.git
cd pg3d
uv sync --extra cu129 --group dev
```

## ManiSkill smoke

First install the optional extra as described in `docs/runbooks/maniskill_setup.md`, then run:

```bash
uv run python scripts/check_maniskill.py
make maniskill-check
```

The default smoke uses `PickCube-v1` with `obs_mode="state"` and no rendering.

## W&B

```bash
wandb login
# or for offline/debug:
export WANDB_MODE=offline
```

## Long-running training policy

Do not run long training jobs from Codex unless explicitly instructed. Codex should prepare commands/configs and run only smoke-scale jobs by default.
