# ADR 0008: Observation schema and mask policy

Date: 2026-05-16

## Status

Accepted

## Context

The ReachTarget MVP needs one RLBench observation to feed later DP3-style policy code, dataset
writers, and the kinematic world model. The same simulator observation also contains fields that
should not become default policy inputs: target positions, raw simulator object handles, and named
object masks.

The first pg3d adapter therefore needs a small schema that makes the policy/evaluation boundary
explicit while keeping enough simulator context to debug and evaluate geometry guidance. The world
model also needs a robot-point mask so it can remove observed robot points before composing imagined
robot geometry back into the scene.

## Decision

Use typed dataclasses for the initial observation boundary:

- `RobotState`
- `Observation`
- `SimGroundTruth`

`Observation.point_cloud` is the primary policy-visible input and follows `float32 [N, 3]` world XYZ
conventions. Optional RGB may be carried as `point_features["rgb"]` with `uint8 [N, 3]`; color use is
opt-in.

`RobotState.as_agent_pos()` returns joint positions only for the first ReachTarget/DP3 adapter. Other
proprioceptive fields remain available for logging and future adapters, but they are not implicitly
added to policy inputs.

Keep evaluation-only simulator context separate:

- `Observation.sim_gt` holds `SimGroundTruth`, including `target_position`.
- `point_features["instance_id"]` may hold raw RLBench object handle ids for debugging/artifacts, but
  is not a default policy input.
- `Observation.object_masks` may hold named masks such as `target` for evaluation/debugging, but
  named object masks are not default policy inputs.

Make `Observation.robot_mask` a first-class optional `bool [N]` mask. The real ReachTarget save smoke
requires it by default and exposes `--allow-missing-robot-mask` only for setup debugging. Future
dataset writing should preserve the mask separately from policy-visible arrays.

Force and gripper-touch-force fields stay optional in `RobotState`. CoppeliaSim may not have
first-frame values available, and the observation adapter should not require those fields.

## Consequences

The schema prevents accidental leakage of target position or simulator identity labels into policy
training while keeping those fields available for evaluation, visual checks, and artifact summaries.

The robot mask requirement gives the M4 world model a stable contract for robot-point removal without
requiring every simulator backend to expose richer semantic segmentation on day one.

Dataset writers and policy adapters must make an explicit choice about which fields become model
inputs. They should not flatten the whole `Observation` object into a training item.

## Alternatives considered

- Pass through raw RLBench observations. This was rejected because the policy/evaluation boundary
  would be implicit and easy to violate.
- Make target position policy-visible for reach. This was rejected because it would hide whether the
  point-cloud policy can solve the intended observation problem.
- Require all named object masks. This was rejected because simulator support may vary, while the
  robot mask is the minimum needed for the first world-model path.
