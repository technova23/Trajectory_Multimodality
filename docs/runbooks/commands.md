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

## ManiSkill observation artifact

```bash
uv run python scripts/save_maniskill_observation.py --obs-mode state_dict \
  --output-dir artifacts/maniskill_state_observation
uv run python scripts/save_maniskill_observation.py --obs-mode pointcloud \
  --output-dir artifacts/maniskill_pointcloud_observation
```

For optional Rerun export, first sync the `viz` extra:

```bash
uv sync --extra cu129 --extra maniskill --extra viz --group dev --group notebooks
```

The optional `viz` extra uses `rerun-sdk==0.22.1` while pg3d remains on NumPy 1.x.

## ManiSkill reach dataset

Generate a small reach dataset:

```bash
uv run python scripts/write_maniskill_reach_dataset.py \
  --num-demos 5 \
  --hold-steps 8 \
  --output artifacts/pg3d_reach_narrow.zarr \
  --overwrite
```

Replay the stored simulator actions:

```bash
uv run python scripts/replay_maniskill_reach_dataset.py \
  --dataset artifacts/pg3d_reach_narrow.zarr \
  --episodes 5
```

Replay with MP4 videos and per-episode Rerun timeline artifacts:

```bash
uv run python scripts/replay_maniskill_reach_dataset.py \
  --dataset artifacts/pg3d_reach_narrow.zarr \
  --episodes 5 \
  --video-dir artifacts/reach_replay/videos \
  --rerun-dir artifacts/reach_replay/rerun
```

Open a replay `.rrd` in the Rerun viewer:

```bash
uv run rerun artifacts/reach_replay/rerun/episode_000.rrd
```

Use the `step` timeline in the Rerun viewer and press play.

The dataset writer uses `PG3DReach-Narrow-v0`, `obs_mode="pointcloud"`, `pd_joint_pos`,
Panda arm-only 7D DP3 action labels, one extra action chunk of post-success hold-pose data, and a
fixed-size cropped point cloud by default.

Pilot before launching the 500-episode dataset:

```bash
uv run python scripts/write_maniskill_reach_dataset.py \
  --env-id PG3DReach-Narrow-v0 \
  --num-demos 100 \
  --max-attempts 150 \
  --hold-steps 8 \
  --num-points 512 \
  --output artifacts/reach-datasets/pg3d-reach-narrow-100.zarr \
  --overwrite
```

Scale to the first nominal 500-episode dataset after pilot replay inspection:

```bash
uv run python scripts/write_maniskill_reach_dataset.py \
  --env-id PG3DReach-Narrow-v0 \
  --num-demos 500 \
  --max-attempts 700 \
  --hold-steps 8 \
  --num-points 512 \
  --output artifacts/reach-datasets/pg3d-reach-narrow-500.zarr \
  --overwrite
```

## DP3 reach training smoke

Run a short dataset-loading and training smoke:

```bash
uv run python scripts/train_dp3_reach.py \
  --dataset artifacts/reach-dataset-smoke/pg3d-reach-smoke.zarr \
  --device cpu \
  --max-steps 1 \
  --batch-size 2 \
  --num-workers 0 \
  --val-ratio 0 \
  --checkpoint-dir artifacts/reach-dataset-smoke/checkpoints \
  --checkpoint-every 1 \
  --no-checkpoint-rollout-videos
```

Run dataset-only inference/eval against that checkpoint:

```bash
uv run python scripts/eval_dp3_reach_dataset.py \
  --dataset artifacts/reach-dataset-smoke/pg3d-reach-smoke.zarr \
  --checkpoint artifacts/reach-dataset-smoke/checkpoints/final_step_00000001.pt \
  --device cpu \
  --max-batches 1 \
  --batch-size 2
```

Enable W&B metric and histogram logging when the local W&B service can start:

```bash
uv run python scripts/train_dp3_reach.py \
  --dataset artifacts/reach-dataset-smoke/pg3d-reach-smoke.zarr \
  --device cpu \
  --max-steps 1 \
  --wandb-mode offline \
  --log-histograms \
  --checkpoint-dir artifacts/reach-dataset-smoke/checkpoints \
  --no-checkpoint-rollout-videos
```

In restricted sandboxes, W&B may fail to create its local cache/socket. The trainer logs a warning
and continues unless `--wandb-required` is set.

Moderate 5090 pilot training recipe:

```bash
uv run python scripts/train_dp3_reach.py \
  --dataset artifacts/reach-datasets/pg3d-reach-narrow-100.zarr \
  --device cuda \
  --max-steps 20000 \
  --batch-size 64 \
  --num-workers 4 \
  --val-ratio 0.1 \
  --val-every 500 \
  --lr 1e-4 \
  --warmup-steps 500 \
  --grad-clip-norm 1.0 \
  --use-ema \
  --wandb-mode online \
  --wandb-project pg3d \
  --wandb-name dp3-reach-narrow-100-stable \
  --checkpoint-dir artifacts/reach-datasets/dp3-reach-narrow-100-stable-checkpoints \
  --checkpoint-every 5000 \
  --checkpoint-rollout-count 5
```

The trainer defaults to `pad_after=n_action_steps-1`, cosine LR with warmup, AdamW
`betas=(0.95, 0.999)`, gradient clipping, EMA checkpoint state, and W&B validation/action-error
metrics. It writes periodic `step_XXXXXXXX.pt` checkpoints and a final
`final_step_XXXXXXXX.pt` checkpoint under `--checkpoint-dir`. When W&B is active,
it also attempts to log checkpoint-time rollout MP4s using three dataset seeds and two fresh seeds
by default. Use `--no-checkpoint-rollout-videos` to skip simulator/rendering rollouts during
training.

Run closed-loop policy rollouts in ManiSkill and save MP4/Rerun artifacts:

```bash
uv sync --extra cu129 --extra maniskill --extra viz --group dev --group notebooks
```

```bash
uv run python scripts/rollout_dp3_reach_policy.py \
  --dataset artifacts/reach-dataset-smoke/pg3d-reach-smoke.zarr \
  --checkpoint artifacts/reach-dataset-smoke/checkpoints/final_step_00000001.pt \
  --source dataset \
  --episodes 3 \
  --device cuda \
  --output-dir artifacts/reach-dataset-smoke/policy-rollouts-dataset
```

Evaluate fresh seeds from the same reach distribution:

```bash
uv run python scripts/rollout_dp3_reach_policy.py \
  --dataset artifacts/reach-dataset-smoke/pg3d-reach-smoke.zarr \
  --checkpoint artifacts/reach-dataset-smoke/checkpoints/final_step_00000001.pt \
  --source fresh \
  --episodes 3 \
  --seed-start 10000 \
  --device cuda \
  --output-dir artifacts/reach-dataset-smoke/policy-rollouts-fresh
```

The rollout script re-observes after each configurable `--replan-stride` chunk, uses EMA checkpoint
weights by default when present, records one post-success hold window by default, and always logs
the goal marker in the Rerun timeline.

## W&B

```bash
wandb login
# or for offline/debug:
export WANDB_MODE=offline
```

## Long-running training policy

Do not run long training jobs from Codex unless explicitly instructed. Codex should prepare commands/configs and run only smoke-scale jobs by default.
