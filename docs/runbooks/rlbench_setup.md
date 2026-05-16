# RLBench setup runbook

Status: first ReachTarget smoke and observation-save paths added for M1.

Known constraints:

- pg3d's M1 smoke/save path is validated on CoppeliaSim 4.9.0 rev6 on Ubuntu 22.04.
- PyRep upstream still targets older CoppeliaSim APIs; pg3d applies local compatibility shims for
  the observation-only paths.
- PyRep communication is Linux-focused.
- The initial task is RLBench `ReachTarget`.
- RLBench is an optional uv extra, installed from upstream `stepjam/RLBench` `master`.
- PyRep is pulled by RLBench and requires `COPPELIASIM_ROOT` during installation/build.
- `pyproject.toml` provides uv dependency metadata for PyRep so lockfile resolution does not need
  to build PyRep before CoppeliaSim is installed.

## Install CoppeliaSim

Use the workstation CoppeliaSim install. The current validated path is CoppeliaSim 4.9.0 rev6 on
Ubuntu 22.04:

```bash
export COPPELIASIM_ROOT=/home/krishna/code/CoppeliaSim_Edu_V4_9_0_rev6_Ubuntu22_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

Other CoppeliaSim versions may need a small compatibility extension. CoppeliaSim 4.1.0 remains the
upstream PyRep reference, but it is not required for the current pg3d observation smoke.

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

The pg3d `rlbench` extra also pins `gymnasium==1.0.0a2`. Upstream RLBench declares this under its
own `gym` extra, but current RLBench imports can require it from runtime paths used by the
ReachTarget scripts. It also installs `pyzmq` and `cbor2` for CoppeliaSim 4.9's Python add-on
launcher.

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

## ReachTarget adapted observation bundle

Save one adapted pg3d observation:

```bash
uv run python scripts/rlbench_save_observation.py --headless true \
  --output-dir artifacts/rlbench_observation
```

Save the same observation with a rotating point-cloud MP4:

```bash
uv run python scripts/rlbench_save_observation.py --headless true \
  --output-dir artifacts/rlbench_observation \
  --visualize true
```

The script enables RGB, depth, point clouds, masks, robot proprioception, and
`task_low_dim_state` for the selected cameras. First-frame joint forces and touch forces are left
disabled because CoppeliaSim may report no value yet; the adapter schema keeps those fields
optional. Outputs:

- `summary.json`: shape/dtype summary, mask availability, robot state, and ReachTarget sim GT.
- `observation.npz`: arrays for `point_cloud`, `agent_pos`, point features, robot mask, optional
  named object masks, robot state, and sim GT.
- `observation.mp4`: optional visual artifact from `--visualize true`.

By default the save path requires a derived `robot_mask`. Use `--allow-missing-robot-mask` only for
setup debugging, because the M4 world model depends on robot-point removal.
The policy/evaluation split for these fields is recorded in
`docs/adr/0008-observation-schema-and-masks.md`.

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

### `ModuleNotFoundError: No module named 'gymnasium'`

Observed after installing RLBench at commit `02720bba4c73fe02eb75df946b8791b806028a9d` and PyRep
at commit `8f420be8064b1970aae18a9cfbc978dfb15747ef`.

Fix: sync from the updated pg3d lockfile:

```bash
uv sync --extra cu129 --extra rlbench --group dev
```

The setup check now reports missing `gymnasium` before trying to import RLBench.

### CoppeliaSim 4.9 compatibility notes

Observed and fixed for CoppeliaSim 4.9.0 rev6:

- `python/pythonLauncher.py` needed `zmq` and `cbor2`; pg3d now installs those in the `rlbench`
  extra and points CoppeliaSim's named `python` parameter at the active virtualenv.
- `PyRep.launch(scene_file=...)` did not load RLBench `task_design.ttt`; pg3d launches an empty
  scene and then calls `simLoadScene`.
- PyRep's single-precision CFFI wrappers could corrupt memory on 4.9; pg3d routes the
  precision-sensitive calls to CoppeliaSim `*_D` symbols.
- RLBench waypoint feasibility validation calls dropped path/IK APIs; pg3d disables that validation
  only in observation-only smoke/save paths.

Live RLBench demo generation on 4.9 is still not supported by this adapter path.
The compatibility decision is recorded in `docs/adr/0007-coppeliasim-49-rlbench-compat.md`.
