from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytest

from scripts import rlbench_smoke_reach


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("true", True),
        ("false", False),
        ("1", True),
        ("0", False),
        ("yes", True),
        ("no", False),
        (None, True),
        (True, True),
        (False, False),
    ],
)
def test_parse_bool(value: str | bool | None, expected: bool) -> None:
    assert rlbench_smoke_reach.parse_bool(value) is expected


def test_parse_bool_rejects_invalid_value() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        rlbench_smoke_reach.parse_bool("maybe")


def test_check_setup_reports_missing_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COPPELIASIM_ROOT", raising=False)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.delenv("QT_QPA_PLATFORM_PLUGIN_PATH", raising=False)
    monkeypatch.setattr(rlbench_smoke_reach, "package_available", lambda _: False)

    errors, warnings = rlbench_smoke_reach.check_setup()

    assert warnings == []
    assert any("COPPELIASIM_ROOT is not set" in error for error in errors)
    assert any("Python package 'rlbench' is not installed" in error for error in errors)
    assert any("Python package 'pyrep' is not installed" in error for error in errors)
    assert any("Python package 'gymnasium' is not installed" in error for error in errors)
    assert any("Python package 'pyzmq' is not installed" in error for error in errors)
    assert any("Python package 'cbor2' is not installed" in error for error in errors)


def test_check_setup_reports_library_path_warnings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("COPPELIASIM_ROOT", os.fspath(tmp_path))
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.delenv("QT_QPA_PLATFORM_PLUGIN_PATH", raising=False)
    monkeypatch.setattr(rlbench_smoke_reach, "package_available", lambda _: True)

    errors, warnings = rlbench_smoke_reach.check_setup()

    assert errors == []
    assert any("coppeliaSim.sh was not found" in warning for warning in warnings)
    assert any(
        "LD_LIBRARY_PATH does not include COPPELIASIM_ROOT" in warning for warning in warnings
    )
    assert any(
        "QT_QPA_PLATFORM_PLUGIN_PATH does not equal COPPELIASIM_ROOT" in warning
        for warning in warnings
    )
    assert any("does not look like a validated pg3d CoppeliaSim" in warning for warning in warnings)


def test_check_setup_accepts_coppeliasim_410_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
    root.mkdir()
    (root / "coppeliaSim.sh").touch()
    monkeypatch.setenv("COPPELIASIM_ROOT", os.fspath(root))
    monkeypatch.setenv("LD_LIBRARY_PATH", os.fspath(root))
    monkeypatch.setenv("QT_QPA_PLATFORM_PLUGIN_PATH", os.fspath(root))
    monkeypatch.setattr(rlbench_smoke_reach, "package_available", lambda _: True)

    errors, warnings = rlbench_smoke_reach.check_setup()

    assert errors == []
    assert not any(
        "does not look like a validated pg3d CoppeliaSim" in warning
        for warning in warnings
    )


def test_check_setup_accepts_validated_coppeliasim_490_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "CoppeliaSim_Edu_V4_9_0_rev6_Ubuntu22_04"
    root.mkdir()
    (root / "coppeliaSim.sh").touch()
    monkeypatch.setenv("COPPELIASIM_ROOT", os.fspath(root))
    monkeypatch.setenv("LD_LIBRARY_PATH", os.fspath(root))
    monkeypatch.setenv("QT_QPA_PLATFORM_PLUGIN_PATH", os.fspath(root))
    monkeypatch.setattr(rlbench_smoke_reach, "package_available", lambda _: True)

    errors, warnings = rlbench_smoke_reach.check_setup()

    assert errors == []
    assert not any(
        "does not look like a validated pg3d CoppeliaSim" in warning
        for warning in warnings
    )


def test_check_setup_warns_on_unvalidated_coppeliasim_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "CoppeliaSim_Edu_V5_0_0_Ubuntu22_04"
    root.mkdir()
    (root / "coppeliaSim.sh").touch()
    monkeypatch.setenv("COPPELIASIM_ROOT", os.fspath(root))
    monkeypatch.setenv("LD_LIBRARY_PATH", os.fspath(root))
    monkeypatch.setenv("QT_QPA_PLATFORM_PLUGIN_PATH", os.fspath(root))
    monkeypatch.setattr(rlbench_smoke_reach, "package_available", lambda _: True)

    errors, warnings = rlbench_smoke_reach.check_setup()

    assert errors == []
    assert any("does not look like a validated pg3d CoppeliaSim" in warning for warning in warnings)


def test_check_setup_reports_missing_gymnasium(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
    root.mkdir()
    (root / "coppeliaSim.sh").touch()
    monkeypatch.setenv("COPPELIASIM_ROOT", os.fspath(root))
    monkeypatch.setenv("LD_LIBRARY_PATH", os.fspath(root))
    monkeypatch.setenv("QT_QPA_PLATFORM_PLUGIN_PATH", os.fspath(root))
    monkeypatch.setattr(
        rlbench_smoke_reach,
        "package_available",
        lambda name: name != "gymnasium",
    )

    errors, warnings = rlbench_smoke_reach.check_setup()

    assert warnings == []
    assert len(errors) == 1
    assert "Python package 'gymnasium' is not installed" in errors[0]
