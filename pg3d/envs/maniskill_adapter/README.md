# ManiSkill adapter notes

This package is the simulator-specific home for ManiSkill/SAPIEN integration.

Current scope is intentionally small:

- keep `pg3d` imports free of ManiSkill, SAPIEN, rendering, Vulkan, and GPU requirements;
- keep typed observation boundary objects available for downstream dataset, policy, and world-model work;
- validate the optional simulator dependency through `scripts/check_maniskill.py`;
- adapt `state_dict` and optional `pointcloud` observations for built-in Franka/Panda
  `PickCube-v1` smoke;
- register narrow custom reach tasks lazily for smoke-scale dataset writing;
- write DP3-compatible reach Zarr datasets while keeping training integration in later milestones.

The first task path is:

1. smoke a built-in ManiSkill task such as `PickCube-v1`;
2. generate smoke data from `PG3DReach-Narrow-v0`;
3. build constrained reach before moving to pick-and-place.
