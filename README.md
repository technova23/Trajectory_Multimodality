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

## Focused Guide: Coding Agents Steer Policy

This repo is currently centered on a reach-task pipeline:

```text
ManiSkill PG3DReach env
  -> multimodal demonstration writer
  -> Zarr point-cloud/action dataset
  -> pg3d-native DP3 policy training
  -> stochastic candidate rollout / constrained visualization
  -> deterministic geometry world-model tree
```

The important idea is that the policy is learned, but the world model used for
imagined rollout is not learned. It is geometry-based: static scene points stay
fixed, robot points are removed from the current point cloud, future robot
geometry is rendered from imagined joint states, and the future point cloud is
composed as static scene plus future robot cloud.

### Files We Touch Most

- `scripts/write_maniskill_reach_dataset.py`
  Generates multimodal reach demonstrations with ManiSkill motion planning and
  writes the DP3-compatible Zarr dataset.

- `pg3d/envs/maniskill_adapter/dataset.py`
  Defines the Zarr schema, point-cloud crop/pad logic, action labels, and
  metadata writing.

- `pg3d/envs/maniskill_adapter/types.py`
  Defines the clean observation boundary: point cloud, robot mask, robot state,
  TCP pose, and simulator ground truth.

- `pg3d/envs/maniskill_adapter/reach_env.py`
  Defines `PG3DReach-*` ManiSkill environments and goal sampling.

- `pg3d/policies/dp3/policy.py`
  Contains `SimpleDP3`, the pg3d-native DP3 policy core.

- `pg3d/policies/dp3/modules.py`
  Contains the PointNet-style encoder, diffusion U-Net modules, FiLM/cross
  attention conditioning, and EMA helper.

- `scripts/train_dp3_reach.py`
  Trains the pg3d-native DP3 policy from the reach Zarr dataset and writes
  `.pt` checkpoints.

- `scripts/rollout_dp3_reach_policy.py`
  Executes a trained pg3d-native checkpoint in ManiSkill and writes MP4, Rerun,
  JSON, and metrics artifacts.

- `scripts/visualize_constrained_candidates_rerun.py`
  Samples many natural stochastic DP3 rollouts, clusters/visualizes candidate
  TCP trajectories, and overlays an avoid sphere.

- `scripts/visualize_constrained_candidates_external_dp3_rerun.py`
  Wrapper for external 3D-Diffusion-Policy `.ckpt` checkpoints. It loads the
  upstream `TrainDP3Workspace` checkpoint and delegates to the local constrained
  candidate visualizer.

- `pg3d/world_model/`
  The deterministic geometry world model: action chunks, joint interpretation,
  static-scene compositing, and imagined rollout types.

- `pg3d/envs/maniskill_adapter/geometry.py`
  `ManiSkillGhostPandaGeometryProvider`, which uses a second ManiSkill env to
  render robot-segmented point clouds at imagined Panda joint states.

- `scripts/trajectory_tree_world_model.py`
  Builds a recursive DP3 candidate tree with the geometry world model. It does
  not train any new model and does not execute the sampled candidate actions in
  the real env.

### Dataset Timing

The reach dataset stores one row per ManiSkill control step. For the current
PG3DReach/Panda setup:

```text
control_freq = 20 Hz
control_timestep = 0.05 s
sim_freq = 100 Hz
sim_timestep = 0.01 s
```

So each Zarr row is one point-cloud frame and one action label every `0.05 s`.
Each `env.step(action)` advances five physics substeps. The row stores the
observation at the start of the control step and the action applied during the
following control step.

### Zarr Schema

The dataset writer saves arrays under `/data`:

```text
state             [T, 9]       Panda qpos / DP3 low-dimensional state
action            [T, 7]       DP3 arm action label
sim_action        [T, A]       full simulator action, usually arm + gripper
point_cloud       [T, N, 3]    cropped/padded XYZ point cloud
robot_mask        [T, N]       robot segmentation mask aligned to point_cloud
point_valid_mask  [T, N]       valid crop slots; false entries are padding
target_position   [T, 3]       goal position
tcp_pose          [T, 7]       end-effector pose
success           [T]          success flag
```

Episode boundaries live in `/meta/episode_ends`. Human-readable metadata lives
in `metadata.json`.

Action labels are:

```text
abs_joint:   action = sim_action[:7]
delta_joint: action = sim_action[:7] - state[:7]
```

### Trajectory Families

The dataset writer creates multimodal demonstrations by choosing waypoint
families. For start `s`, goal `g`, and `d = g - s`, each waypoint is formed as:

```text
w = s + r(g - s) + alpha * lateral_axis + beta * vertical_axis + noise
```

where:

```text
lateral_axis = normalized perpendicular to XY projection of (g - s)
vertical_axis = [0, 0, 1]  # global Z
```

Family names such as `left_wide`, `right_wide`, `upper_arc`, `lower_arc`,
`shallow_direct`, and `extreme_detour` control the signs and magnitudes of
`alpha` and `beta`. Family metadata is used for generation/diagnostics and is
stripped before training so DP3 must infer modes from observations.

### Training Defaults

The current training path uses:

```text
horizon = 16
n_obs_steps = 2
n_action_steps = 8
condition_type = cross_attention
prediction_type = epsilon
num_train_timesteps = 350
down_dims = [512, 1024, 2048]
diffusion_step_embed_dim = 256
loss = Huber(delta=1.0)
```

DP3 predicts a horizon of 16 actions but uses receding-horizon execution: by
default only the next 8 actions are applied before replanning.

### Common Commands

Generate a multimodal dataset:

```bash
UV_CACHE_DIR=/tmp/pg3d-uv-cache uv run python scripts/write_maniskill_reach_dataset.py \
  --env-id PG3DReach-BalancedWorkspace-v0 \
  --num-demos 4800 \
  --trajectory-variants-per-reset 12 \
  --num-points 1024 \
  --output dataset_generation/artifacts/debug_multimodal_test.zarr \
  --overwrite
```

Train pg3d-native DP3:

```bash
UV_CACHE_DIR=/tmp/pg3d-uv-cache uv run python scripts/train_dp3_reach.py \
  --dataset dataset_generation/artifacts/debug_multimodal_test.zarr \
  --device cuda \
  --max-steps 100000 \
  --batch-size 64 \
  --val-ratio 0.1 \
  --checkpoint-dir artifacts/checkpoints/dp3_reach_cross_attention_huber \
  --checkpoint-every 5000 \
  --condition-type cross_attention \
  --encoder-output-dim 128 \
  --diffusion-step-embed-dim 256 \
  --down-dims 512 1024 2048 \
  --num-train-timesteps 350 \
  --num-inference-steps 350 \
  --prediction-type epsilon
```

Visualize many stochastic constrained candidates from a pg3d-native checkpoint:

```bash
UV_CACHE_DIR=/tmp/pg3d-uv-cache uv run python scripts/visualize_constrained_candidates_rerun.py \
  --dataset dataset_generation/artifacts/debug_multimodal_test.zarr \
  --checkpoint artifacts/checkpoints/dp3_reach_cross_attention_huber/final_step_00010000.pt \
  --checkpoint-model ema \
  --device cuda \
  --episode-index 0 \
  --candidates 100 \
  --steps 80 \
  --avoid-radius 0.08 \
  --avoid-min-radius 0.08 \
  --output artifacts/constrained_candidates/candidates.rrd \
  --video artifacts/constrained_candidates/candidates.mp4
```

Visualize an external 3D-Diffusion-Policy `.ckpt` checkpoint:

```bash
UV_CACHE_DIR=/tmp/pg3d-uv-cache uv run python scripts/visualize_constrained_candidates_external_dp3_rerun.py \
  --external-dp3-repo /home/skills/gnrs/3D-Diffusion-Policy/3D-Diffusion-Policy \
  --dataset dataset_generation/artifacts/debug_multimodal_test.zarr \
  --checkpoint /path/to/external/checkpoints/epoch=0040-test_mean_score=-0.000.ckpt \
  --checkpoint-model ema \
  --device cuda \
  --episode-index 0 \
  --candidates 100 \
  --steps 80 \
  --avoid-radius 0.08 \
  --avoid-min-radius 0.08 \
  --output artifacts/constrained_candidates_external_dp3/candidates.rrd \
  --video artifacts/constrained_candidates_external_dp3/candidates.mp4
```

Build a geometry world-model trajectory tree:

```bash
UV_CACHE_DIR=/tmp/pg3d-uv-cache uv run python scripts/trajectory_tree_world_model.py \
  --dataset dataset_generation/artifacts/debug_multimodal_test.zarr \
  --checkpoint artifacts/checkpoints/dp3_reach_cross_attention_huber/final_step_00010000.pt \
  --checkpoint-model ema \
  --device cuda \
  --episode-index 0 \
  --root-candidates 32 \
  --branching-factor 16 \
  --tree-depth 3 \
  --replan-step 8 \
  --hz 16 \
  --output artifacts/trajectory_tree_world_model/trajectory_tree.rrd \
  --video artifacts/trajectory_tree_world_model/trajectory_tree.mp4
```

For a quick smoke test, use:

```bash
--root-candidates 2 --branching-factor 2 --tree-depth 2
```

The tree script also writes `trajectory_tree.json` next to the `.rrd`, including
node ids, parent-child links, action sequences, predicted joint trajectories,
and predicted end-effector trajectories.

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
  --hold-steps 8 \
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
for replay. By default it records one extra DP3 action chunk of hold-pose data
after first success so terminal chunks learn to stay at the goal.

## Reach Dataset Pilot

Before generating the 500-episode dataset, create and inspect a 50-100 episode
pilot:

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

Replay a few pilot episodes with videos and Rerun timelines:

```bash
uv run python scripts/replay_maniskill_reach_dataset.py \
  --dataset artifacts/reach-datasets/pg3d-reach-narrow-100.zarr \
  --episodes 5 \
  --video-dir artifacts/reach-datasets/pg3d-reach-narrow-100-replay/videos \
  --rerun-dir artifacts/reach-datasets/pg3d-reach-narrow-100-replay/rerun
```

Once the pilot looks right, generate the 500-episode dataset:

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

## DP3 Training Smoke

Run a one-step behavior-cloning smoke on the reach dataset:

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

Check dataset-only inference against the checkpoint:

```bash
uv run python scripts/eval_dp3_reach_dataset.py \
  --dataset artifacts/reach-dataset-smoke/pg3d-reach-smoke.zarr \
  --checkpoint artifacts/reach-dataset-smoke/checkpoints/final_step_00000001.pt \
  --device cpu \
  --max-batches 1
```

Train the first moderate 5090 pilot checkpoint:

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

The trainer writes `step_XXXXXXXX.pt` checkpoints at `--checkpoint-every` intervals
and always writes `final_step_XXXXXXXX.pt` in the checkpoint directory. When W&B
is active, checkpoint-time rollout MP4s are logged best-effort; use
`--no-checkpoint-rollout-videos` to skip simulator/rendering rollouts during
training.

## DP3 Policy Rollout Smoke

Roll out a trained reach checkpoint in a live ManiSkill environment and save
local MP4, Rerun, and JSON artifacts:

```bash
uv run python scripts/rollout_dp3_reach_policy.py \
  --dataset artifacts/reach-dataset-smoke/pg3d-reach-smoke.zarr \
  --checkpoint artifacts/reach-dataset-smoke/checkpoints/final_step_00000001.pt \
  --source dataset \
  --episodes 3 \
  --device cuda \
  --output-dir artifacts/reach-dataset-smoke/policy-rollouts-dataset
```

For fresh seeds from the same reach distribution:

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

The rollout script loads env configuration from the dataset metadata, uses
closed-loop action chunks, uses EMA checkpoints by default when present, keeps
rolling briefly after success for stability diagnostics, and writes
`summary.json`, `metrics.jsonl`, `episode_*.mp4`, and `episode_*.rrd`.

## World-Model Comparison

Compare checkpoint-predicted action chunks rolled out through the P07 world
model against the same chunks executed in ManiSkill:

```bash
uv run python scripts/compare_world_model_rollout.py \
  --dataset artifacts/reach-datasets/pg3d-reach-narrow-100.zarr \
  --checkpoint-dir artifacts/reach-datasets/dp3-reach-narrow-100-stable-checkpoints \
  --source dataset \
  --episodes 3 \
  --device cuda \
  --output-dir artifacts/reach-datasets/world-model-vs-sim \
  --rerun \
  --video \
  --allow-failure
```

Open the overlay:

```bash
uv run rerun artifacts/reach-datasets/world-model-vs-sim/episode_000_comparison.rrd
```

The comparison uses a second ManiSkill ghost env to render robot-segmented point
clouds at imagined Panda qpos states. Rerun overlays the world-model branch and
the live simulator branch with distinct robot-point colors, writing one
`episode_XXX_comparison.rrd` file per compared episode.

## Docs

- `AGENTS.md`: durable agent instructions.
- `docs/project_proposal.html`: source-of-truth research proposal.
- `docs/status.md`: current state and next steps.
- `docs/milestones.md`: staged implementation plan.
- `docs/runbooks/commands.md`: canonical commands.
- `docs/runbooks/maniskill_setup.md`: ManiSkill setup notes.
- `docs/adr/`: durable design decisions.
- `docs/prompts/`: Codex milestone prompts.
