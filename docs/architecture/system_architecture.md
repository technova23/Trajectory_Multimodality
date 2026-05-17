# System architecture

## Package layout

```text
pg3d/
  envs/
    maniskill_adapter/      # simulator-specific wrappers and data collection
  policies/                 # policy interface and DP3 adapter
    dp3/                    # pg3d-native, simulation-free DP3 policy core
  world_model/              # kinematic point-cloud imagination
  constraints/              # executable geometric constraint objects
  composition/              # rejection, reranking, receding horizon, later guidance
  baselines/                # code-only and prior-style baselines
  eval/                     # metrics, experiment runners, confidence intervals
  viz/                      # point-cloud/trajectory/constraint visualization
  logging/                  # W&B and structured logs
  utils/
```

Keep simulator and policy dependencies lazy. Importing `pg3d` should not require ManiSkill,
SAPIEN, rendering/GPU simulator dependencies, or DP3. DP3 runtime imports should use
`pg3d.policies.dp3`; the private
`external/dp3` submodule is reference material during migration and should not be imported by
pg3d runtime code.

## DP3 policy slice

The pg3d-native DP3 slice keeps only the model and training primitives needed for the ManiSkill
reach MVP:

- point-cloud/state encoder,
- 1D diffusion action model,
- normalizer, mask generator, EMA/checkpoint utilities as needed,
- synthetic import/inference/loss smoke tests,
- future generic zarr dataset and trainer scripts.

It intentionally excludes upstream DP3 benchmark/simulation dependencies such as MuJoCo, Gym,
MetaWorld, DexArt, RRL, PyTorch3D, and task-generation scripts.

## Core objects

### Observation

Durable policy/evaluation boundary decisions for this schema are recorded in
`docs/adr/0008-observation-schema-and-masks.md`.

```python
@dataclass
class Observation:
    point_cloud: np.ndarray          # [N, 3], policy-visible
    point_features: dict[str, Any]   # optional aligned point features
    robot_mask: np.ndarray | None    # [N], optional but important for world model
    object_masks: dict[str, np.ndarray]
    robot_state: RobotState
    sim_gt: SimGroundTruth | None    # eval/debug only; not policy input
```

Current shape conventions:

- `point_cloud`: `float32 [N, 3]`, finite XYZ world points, DP3-visible.
- `point_features["rgb"]`: optional `uint8 [N, 3]`; DP3 color use is opt-in.
- `point_features["camera_index"]`: `int16 [N]`, camera provenance for debugging/artifacts.
- `point_features["segmentation"]` or `point_features["instance_id"]`: optional `int64 [N]`
  simulator segmentation ids; do not feed
  this to policies by default.
- `robot_mask`: optional `bool [N]` derived from simulator segmentation; required by the
  world model for robot-point removal.
- `object_masks`: optional named `bool [N]` masks for eval/debug, such as reach `target` and
  distractors. These are not policy inputs by default.
- `RobotState.as_agent_pos()`: joint positions only for the first DP3 reach adapter.
- `SimGroundTruth.target_position`: optional `float32 [3]` from ManiSkill task state/info
  state; eval/debug only.

P04 ManiSkill adapter conventions:

- Default conversion uses `obs_mode="state_dict"` so the adapter can read structured
  `agent.qpos`, `agent.qvel`, `extra.tcp_pose`, and `extra.goal_pos`.
- `obs_mode="pointcloud"` reads `pointcloud.xyzw[..., :3]` as world XYZ, `pointcloud.rgb` as
  optional RGB, and `pointcloud.segmentation` as raw simulator ids.
- Panda robot masks are derived from link `per_scene_id` values when the live ManiSkill env is
  passed as adapter context; raw observations alone are not enough to map ids to robot/object names.

### ActionChunk

```python
@dataclass
class ActionChunk:
    actions: np.ndarray              # [H, action_dim]
    action_mode: Literal["abs_joint", "delta_joint", "ee_pose"]
    dt: float
    metadata: dict[str, Any]
```

Default P0 action representation: absolute joint target chunks. Delta joint chunks are fallback. EE-pose chunks are deferred.

### ImaginedRollout

```python
@dataclass
class ImaginedRollout:
    q: np.ndarray                    # [H, dof]
    eef_path: np.ndarray             # [H, 3]
    robot_point_clouds: list[np.ndarray]
    scene_point_clouds: list[np.ndarray]
    robot_masks: list[np.ndarray]
    action_chunk: ActionChunk
    metadata: dict[str, Any]
```

### Constraint

```python
class Constraint(Protocol):
    def cost(self, rollout: ImaginedRollout, scene: SceneContext) -> dict[str, float]: ...
    def satisfied(self, rollout: ImaginedRollout, scene: SceneContext) -> bool: ...
```

Start with Python objects, but every constraint instance should be JSON-serializable for replay.

## P0 data flow

```text
ManiSkill observation
  -> ObservationAdapter
  -> DP3 policy samples K ActionChunks
  -> GeometricWorldModel imagines each chunk
  -> ConstraintProgram scores each imagined rollout
  -> RerankingController selects chunk
  -> ManiSkill executes first chunk / first action horizon
  -> repeat
```

## World model v0

The first world model is kinematic/geometric:

1. Read current robot joint state.
2. Integrate/interpret candidate joint-action chunk into future joint states.
3. Run FK for future end-effector path.
4. Sample robot link geometry at future joint states.
5. Remove current robot points from current point cloud using `robot_mask`.
6. Insert future robot points into the static scene cloud.
7. Return imagined future clouds and trajectories.

No learned dynamics. No contact dynamics. No object attachment until pick-and-place.

## Constraint v0

`AvoidRegion(target="eef")` over simple sphere/box regions.

Cost terms:

- clearance margin violation,
- final target distance for reach,
- smoothness,
- deviation from policy sample consensus.

## Logging

Every experiment should emit:

- config YAML/JSON,
- constraint instance JSON,
- per-episode metrics JSONL,
- W&B logs where available,
- qualitative videos/plots,
- git commit hashes for main repo and submodules.
