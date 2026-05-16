from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np

from pg3d.envs.rlbench_adapter import setup as rlbench_setup

parse_bool = rlbench_setup.parse_bool
package_available = rlbench_setup.package_available


def check_setup() -> tuple[list[str], list[str]]:
    return rlbench_setup.check_setup(package_available_func=package_available)


def print_setup_report(errors: list[str], warnings: list[str]) -> None:
    rlbench_setup.print_setup_report(errors, warnings)


def load_rlbench() -> dict[str, Any]:
    return rlbench_setup.load_reach_target_runtime()


def value_summary(value: Any) -> str:
    if value is None:
        return "None"
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None:
        suffix = f", dtype={dtype}" if dtype is not None else ""
        return f"{type(value).__name__}(shape={tuple(shape)}{suffix})"
    if isinstance(value, dict):
        keys = ", ".join(str(key) for key in sorted(value.keys(), key=str))
        return f"dict(keys=[{keys}])"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}(len={len(value)})"
    if isinstance(value, (str, int, float, bool)):
        return repr(value)
    return type(value).__name__


def print_observation_summary(obs: Any) -> None:
    print("observation fields:")
    for name in sorted(dir(obs)):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obs, name)
        except Exception as exc:  # pragma: no cover - diagnostic path
            print(f"  {name}: <error reading field: {type(exc).__name__}: {exc}>")
            continue
        if callable(value):
            continue
        print(f"  {name}: {value_summary(value)}")


def build_smoke_observation_config(rlbench: dict[str, Any]) -> Any:
    obs_config = rlbench["ObservationConfig"]()
    obs_config.set_all_high_dim(False)
    obs_config.set_all_low_dim(True)
    obs_config.joint_forces = False
    obs_config.gripper_touch_forces = False
    return obs_config


def run_smoke(headless: bool, steps: int) -> int:
    errors, warnings = check_setup()
    print_setup_report(errors, warnings)
    if errors:
        return 2

    try:
        rlbench = load_rlbench()
    except Exception as exc:
        print(
            "Failed to import RLBench/PyRep after setup checks passed. "
            "This usually means CoppeliaSim shared libraries are not visible.",
            file=sys.stderr,
        )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    action_mode = rlbench["MoveArmThenGripper"](
        arm_action_mode=rlbench["JointVelocity"](),
        gripper_action_mode=rlbench["Discrete"](),
    )
    env = rlbench["Environment"](
        action_mode=action_mode,
        obs_config=build_smoke_observation_config(rlbench),
        headless=headless,
    )

    try:
        print(f"launching RLBench ReachTarget smoke; headless={headless}")
        env.launch()
        print(f"action_shape: {env.action_shape}")

        task = env.get_task(rlbench["ReachTarget"])
        rlbench_setup.disable_waypoint_validation_for_observation(task)
        descriptions, obs = task.reset()
        print("reset ok")
        print(f"descriptions: {descriptions}")
        print_observation_summary(obs)

        for step_idx in range(steps):
            action = np.zeros(env.action_shape, dtype=np.float32)
            obs, reward, terminate = task.step(action)
            print(f"step {step_idx + 1}: reward={reward}, terminate={terminate}")
            if terminate:
                break
        return 0
    except Exception as exc:
        print("RLBench ReachTarget smoke failed during launch/reset/step.", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "Check CoppeliaSim, COPPELIASIM_ROOT, LD_LIBRARY_PATH, "
            "QT_QPA_PLATFORM_PLUGIN_PATH, DISPLAY/headless configuration, and PyRep build logs.",
            file=sys.stderr,
        )
        return 1
    finally:
        try:
            env.shutdown()
            print("shutdown ok")
        except Exception as exc:  # pragma: no cover - cleanup diagnostic path
            print(f"shutdown failed: {type(exc).__name__}: {exc}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test RLBench ReachTarget launch/reset.")
    parser.add_argument(
        "--headless",
        nargs="?",
        const=True,
        default=True,
        type=parse_bool,
        help="Run CoppeliaSim headless. Accepts true/false; bare --headless means true.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Number of zero-action steps after reset. Use 0 to only launch/reset.",
    )
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be >= 0")
    return run_smoke(headless=args.headless, steps=args.steps)


if __name__ == "__main__":
    raise SystemExit(main())
