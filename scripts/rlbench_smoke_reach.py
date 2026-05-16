from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def parse_bool(value: str | bool | None) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def check_setup() -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    coppeliasim_root = os.environ.get("COPPELIASIM_ROOT")
    if coppeliasim_root is None:
        errors.append(
            "COPPELIASIM_ROOT is not set. Install CoppeliaSim 4.1.0 and export "
            "COPPELIASIM_ROOT=/path/to/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04."
        )
    else:
        root = Path(coppeliasim_root)
        if not root.exists():
            errors.append(f"COPPELIASIM_ROOT does not exist: {root}")
        elif not (root / "coppeliaSim.sh").exists():
            warnings.append(
                "COPPELIASIM_ROOT is set, but coppeliaSim.sh was not found there: "
                f"{root}"
            )

    if not package_available("rlbench"):
        errors.append(
            "Python package 'rlbench' is not installed. Run "
            "`uv sync --extra cu129 --extra rlbench --group dev` after setting "
            "COPPELIASIM_ROOT."
        )
    if not package_available("pyrep"):
        errors.append(
            "Python package 'pyrep' is not installed. It is pulled by the rlbench "
            "extra and requires COPPELIASIM_ROOT during installation."
        )

    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if coppeliasim_root and coppeliasim_root not in ld_library_path.split(":"):
        warnings.append(
            "LD_LIBRARY_PATH does not include COPPELIASIM_ROOT. PyRep/CoppeliaSim "
            "usually needs `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT`."
        )

    qt_plugin_path = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH")
    if coppeliasim_root and qt_plugin_path != coppeliasim_root:
        warnings.append(
            "QT_QPA_PLATFORM_PLUGIN_PATH does not equal COPPELIASIM_ROOT. RLBench docs "
            "usually set `export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT`."
        )

    return errors, warnings


def print_setup_report(errors: Sequence[str], warnings: Sequence[str]) -> None:
    if errors:
        print("RLBench smoke setup check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
    if warnings:
        print("RLBench smoke setup warnings:", file=sys.stderr)
        for warning in warnings:
            print(f"- {warning}", file=sys.stderr)
    if errors:
        print(
            "\nExpected setup shape:\n"
            "  export COPPELIASIM_ROOT=/path/to/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04\n"
            "  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT\n"
            "  export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT\n"
            "  uv sync --extra cu129 --extra rlbench --group dev\n",
            file=sys.stderr,
        )


def load_rlbench() -> dict[str, Any]:
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointVelocity
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment
    from rlbench.tasks import ReachTarget

    return {
        "MoveArmThenGripper": MoveArmThenGripper,
        "JointVelocity": JointVelocity,
        "Discrete": Discrete,
        "Environment": Environment,
        "ReachTarget": ReachTarget,
    }


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
    env = rlbench["Environment"](action_mode=action_mode, headless=headless)

    try:
        print(f"launching RLBench ReachTarget smoke; headless={headless}")
        env.launch()
        print(f"action_shape: {env.action_shape}")

        task = env.get_task(rlbench["ReachTarget"])
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
            "Check CoppeliaSim 4.1.0, COPPELIASIM_ROOT, LD_LIBRARY_PATH, "
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
