# ManiSkill setup runbook

Status: primary simulator stack for pg3d.

Known constraints:

- ManiSkill/SAPIEN is optional so base imports and CPU-only tests stay simulator-free.
- `pyproject.toml` pins `mani_skill==3.0.1`; do not change simulator versions without updating
  the lockfile, this runbook, and the simulator ADR/status notes.
- The default smoke path uses `obs_mode="state"` and does not require rendering.
- Point-cloud/RGB-D/segmentation paths may require Vulkan and asset setup.
- The first built-in smoke task is `PickCube-v1`; a narrow pg3d reach task comes after the smoke
  path is stable.

## Install pg3d with ManiSkill

RTX 5090 / CUDA 12.9 path:

```bash
uv sync --extra cu129 --extra maniskill --group dev --group notebooks
```

CPU/debug path:

```bash
uv sync --extra cpu --extra maniskill --group dev
```

Keep the `maniskill` extra out of default docs/test environments. Base `pg3d` imports must not
require ManiSkill, SAPIEN, rendering, Vulkan, or a GPU.

## Assets

Set an asset directory if the default cache location is not desired:

```bash
export MS_ASSET_DIR=/path/to/maniskill_assets
```

For non-interactive smoke runs, avoid asset download prompts:

```bash
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
```

The basic `PickCube-v1` state smoke should not require large visual assets. Custom tasks or
visual-observation scripts may need additional assets later.

## Non-rendering smoke

```bash
uv run python scripts/check_maniskill.py
make maniskill-check
```

The script imports `gymnasium` and `mani_skill.envs`, creates `PickCube-v1` with
`obs_mode="state"` and `num_envs=1`, resets with `seed=0`, prints observation/action spaces, steps
one sampled action, and closes the environment.

## Optional rendering/point-cloud checks

Do not make rendering part of the default smoke path. Once the adapter needs visual observations,
add a separate script for `obs_mode="pointcloud"` or RGB-D/segmentation and document:

- Vulkan driver/runtime status,
- required assets,
- `MS_ASSET_DIR`,
- whether the script can run headless on the workstation.

## Troubleshooting log

Append specific failures/fixes here rather than burying them in chat history.

### Missing optional dependency

If `scripts/check_maniskill.py` reports `Failed to import ManiSkill/Gymnasium`, run:

```bash
uv sync --extra cu129 --extra maniskill --group dev --group notebooks
```

### Vulkan or rendering failure

The default state smoke should not require rendering. If a point-cloud or RGB-D script fails in
renderer setup, first confirm the NVIDIA driver and Vulkan runtime before changing Python or torch
versions.
