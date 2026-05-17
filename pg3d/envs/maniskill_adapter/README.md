# ManiSkill adapter notes

This package is the simulator-specific home for ManiSkill/SAPIEN integration.

Current migration scope is intentionally small:

- keep `pg3d` imports free of ManiSkill, SAPIEN, rendering, Vulkan, and GPU requirements;
- keep typed observation boundary objects available for downstream dataset, policy, and world-model work;
- validate the optional simulator dependency through `scripts/check_maniskill.py`;
- defer a full ManiSkill observation/action adapter until the P04/P05 milestones.

The first task path is:

1. smoke a built-in ManiSkill task such as `PickCube-v1`;
2. implement a narrow `PG3DReach` custom task only if built-in tasks are not enough for reach data;
3. build constrained reach before moving to pick-and-place.
