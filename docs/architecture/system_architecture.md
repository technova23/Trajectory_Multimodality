# System architecture

## Package layout

```text
pg3d/
  envs/
    rlbench_adapter/        # simulator-specific wrappers and data collection
  policies/                 # policy interface and DP3 adapter
  world_model/              # kinematic point-cloud imagination
  constraints/              # executable geometric constraint objects
  composition/              # rejection, reranking, receding horizon, later guidance
  baselines/                # code-only and prior-style baselines
  eval/                     # metrics, experiment runners, confidence intervals
  viz/                      # point-cloud/trajectory/constraint visualization
  logging/                  # W&B and structured logs
  utils/
```

Keep simulator and policy dependencies lazy. Importing `pg3d` should not require RLBench, PyRep, CoppeliaSim, or DP3.

## Core objects

### Observation

```python
@dataclass
class Observation:
    point_cloud: np.ndarray          # [N, 3], policy-visible
    point_features: dict[str, Any]   # optional RGB/masks/features
    robot_mask: np.ndarray | None    # [N], optional but important for world model
    object_masks: dict[str, np.ndarray]
    robot_state: RobotState
    sim_gt: SimGroundTruth | None    # eval/debug only; not policy input
```

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
RLBench observation
  -> ObservationAdapter
  -> DP3 policy samples K ActionChunks
  -> GeometricWorldModel imagines each chunk
  -> ConstraintProgram scores each imagined rollout
  -> RerankingController selects chunk
  -> RLBench executes first chunk / first action horizon
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
