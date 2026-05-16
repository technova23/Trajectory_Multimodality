# RLBench setup runbook

Status: first ReachTarget smoke path added for M1.

Known constraints:

- PyRep requires CoppeliaSim 4.1.
- PyRep communication is Linux-focused.
- The initial task is RLBench `ReachTarget`.
- RLBench is an optional uv extra, installed from upstream `stepjam/RLBench` `master`.
- PyRep is pulled by RLBench and requires `COPPELIASIM_ROOT` during installation/build.
- `pyproject.toml` provides uv dependency metadata for PyRep so lockfile resolution does not need
  to build PyRep before CoppeliaSim is installed.

## Install CoppeliaSim 4.1.0

Download and unpack CoppeliaSim 4.1.0 for Linux. Example local layout:

```bash
mkdir -p ~/opt
tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C ~/opt
```

Set the simulator environment before installing the RLBench extra:

```bash
export COPPELIASIM_ROOT=$HOME/opt/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

Persist those in your shell profile or `.envrc` once the path is confirmed.

## Install pg3d with RLBench

RTX 5090 / CUDA 12.9 path:

```bash
uv sync --extra cu129 --extra rlbench --group dev
```

CPU/debug path:

```bash
uv sync --extra cpu --extra rlbench --group dev
```

Keep the `rlbench` extra out of default docs/test environments. Base `pg3d` imports must not
require RLBench, PyRep, or CoppeliaSim.

## ReachTarget smoke

```bash
uv run python scripts/rlbench_smoke_reach.py --headless false
uv run python scripts/rlbench_smoke_reach.py --headless true
```

The smoke uses RLBench `ReachTarget`, resets once, prints observation field summaries, executes
one zero-action step by default, and shuts down CoppeliaSim. Use `--steps 0` to only launch/reset:

```bash
uv run python scripts/rlbench_smoke_reach.py --headless true --steps 0
```

Expected successful output includes:

- `launching RLBench ReachTarget smoke`,
- `reset ok`,
- `observation fields:`,
- `shutdown ok`.

## Troubleshooting log

Append specific failures/fixes here rather than burying them in chat history.

### Missing simulator/package setup

Observed in the current Codex environment before RLBench/CoppeliaSim installation:

- `COPPELIASIM_ROOT` is unset,
- `rlbench` is not installed,
- `pyrep` is not installed,
- `LD_LIBRARY_PATH` and `QT_QPA_PLATFORM_PLUGIN_PATH` are not configured for CoppeliaSim.

The smoke script reports these as actionable setup errors before importing RLBench.

### PyRep build fails during `uv sync --extra rlbench`

Check that `COPPELIASIM_ROOT` is exported and points to an unpacked CoppeliaSim 4.1.0 directory.
PyRep's setup checks this variable while building.

If this failure appears during lockfile resolution rather than installation, verify that the
`[[tool.uv.dependency-metadata]]` entry for PyRep is still present in `pyproject.toml`.
