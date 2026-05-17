# pg3d

`pg3d` studies programmatic geometric guidance for 3D diffusion robot policies.
The current project is simulation-only and uses ManiSkill/SAPIEN as the primary
simulator stack.

The reach-first MVP is:

1. adapt a DP3-style point-cloud diffusion policy to ManiSkill reach data,
2. build a kinematic point-cloud world model from joint-action chunks,
3. score executable geometric constraints such as `avoid_region`,
4. use candidate rejection/reranking in receding horizon mode,
5. move to pick-and-place only after constrained reach works.

The full source-of-truth research plan is `docs/project_proposal.html`.

## Current Status

- Package name: `pg3d`.
- Python: 3.11.
- Dependency manager: `uv`.
- Workstation target: RTX 5090 with PyTorch CUDA 12.9.
- Simulator: ManiSkill/SAPIEN, installed through an optional `maniskill` extra.
- Base policy: pg3d-native DP3 policy core under `pg3d/policies/dp3`.
- Active simulator smoke: `scripts/check_maniskill.py` using `PickCube-v1` with
  `obs_mode="state"`.
- Active reach task/data path: custom `PG3DReach-Narrow-v0` /
  `PG3DReach-Medium-v0` tasks and a Zarr writer for smoke-scale DP3-compatible
  reach datasets.
- RLBench, PyRep, CoppeliaSim, and real-robot/xArm implementation work are not
  active backends in this repo.

## Setup

For the main workstation environment:

```bash
uv sync --extra cu129 --extra maniskill --group dev --group notebooks
```

For CPU-only docs/tests without the simulator:

```bash
uv sync --extra cpu --group dev
```

For CPU-only work with ManiSkill installed:

```bash
uv sync --extra cpu --extra maniskill --group dev
```

For optional Rerun visualization artifacts:

```bash
uv sync --extra cu129 --extra maniskill --extra viz --group dev --group notebooks
```

If this is a fresh clone, initialize submodules:

```bash
git submodule update --init --recursive
```

`external/dp3` is reference material during migration. Runtime imports should use
`pg3d.policies.dp3`, not `external/dp3`.

## Checks

```bash
make smoke
make test
make lint
make gpu-check
make maniskill-check
```

Equivalent direct commands:

```bash
uv run python scripts/smoke_imports.py
uv run pytest
uv run ruff check .
uv run python scripts/check_gpu.py
uv run python scripts/check_maniskill.py
```

The default ManiSkill check is non-rendering. Point-cloud/RGB-D/segmentation
checks should stay separate because they may require Vulkan and asset setup.

## Reach Dataset Smoke

Generate a small custom-reach dataset:

```bash
uv run python scripts/write_maniskill_reach_dataset.py \
  --num-demos 3 \
  --output artifacts/reach-dataset-smoke/pg3d-reach-smoke.zarr \
  --overwrite
```

Replay stored simulator actions and write videos plus Rerun timelines:

```bash
uv run python scripts/replay_maniskill_reach_dataset.py \
  --dataset artifacts/reach-dataset-smoke/pg3d-reach-smoke.zarr \
  --episodes 3 \
  --video-dir artifacts/reach-dataset-smoke/videos \
  --rerun-dir artifacts/reach-dataset-smoke/rerun
```

Open a replay artifact:

```bash
uv run rerun artifacts/reach-dataset-smoke/rerun/episode_000.rrd
```

Use the `step` timeline in the Rerun viewer and press play. The dataset writer
stores 7D Panda arm labels for DP3 and keeps full simulator actions separately
for replay.

## Docs

- `AGENTS.md`: durable agent instructions.
- `docs/project_proposal.html`: source-of-truth research proposal.
- `docs/status.md`: current state and next steps.
- `docs/milestones.md`: staged implementation plan.
- `docs/runbooks/commands.md`: canonical commands.
- `docs/runbooks/maniskill_setup.md`: ManiSkill setup notes.
- `docs/adr/`: durable design decisions.
- `docs/prompts/`: Codex milestone prompts.
