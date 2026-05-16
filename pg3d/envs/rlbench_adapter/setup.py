from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


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


def check_setup(
    package_available_func: Callable[[str], bool] = package_available,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    coppeliasim_root = os.environ.get("COPPELIASIM_ROOT")
    if coppeliasim_root is None:
        errors.append(
            "COPPELIASIM_ROOT is not set. Install CoppeliaSim and export "
            "COPPELIASIM_ROOT=/path/to/CoppeliaSim_Edu_V4_9_0_rev6_Ubuntu22_04."
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
        is_legacy_410 = "V4_1_0" in root.name or "4_1_0" in root.name
        is_validated_490 = "V4_9_0" in root.name or "4_9_0" in root.name
        if not is_legacy_410 and not is_validated_490:
            version_message = (
                "COPPELIASIM_ROOT does not look like a validated pg3d CoppeliaSim "
                "version. The current adapter path has been smoke-tested with "
                "CoppeliaSim 4.9.0 rev6 on Ubuntu 22.04, with legacy 4.1.0 kept as "
                "the upstream PyRep reference. Other versions may need compatibility "
                "updates. "
                f"Current path is {root}."
            )
            warnings.append(version_message)

    if not package_available_func("rlbench"):
        errors.append(
            "Python package 'rlbench' is not installed. Run "
            "`uv sync --extra cu129 --extra rlbench --group dev` after setting "
            "COPPELIASIM_ROOT."
        )
    if not package_available_func("pyrep"):
        errors.append(
            "Python package 'pyrep' is not installed. It is pulled by the rlbench "
            "extra and requires COPPELIASIM_ROOT during installation."
        )
    if not package_available_func("gymnasium"):
        errors.append(
            "Python package 'gymnasium' is not installed. RLBench's current upstream "
            "package imports gymnasium from runtime paths; run "
            "`uv sync --extra cu129 --extra rlbench --group dev` after updating the "
            "pg3d lockfile."
        )
    if not package_available_func("zmq"):
        errors.append(
            "Python package 'pyzmq' is not installed. CoppeliaSim 4.9 launches "
            "Python add-ons through a wrapper that imports zmq; run "
            "`uv sync --extra cu129 --extra rlbench --group dev`."
        )
    if not package_available_func("cbor2"):
        errors.append(
            "Python package 'cbor2' is not installed. CoppeliaSim 4.9 launches "
            "Python add-ons through a wrapper that imports cbor2; run "
            "`uv sync --extra cu129 --extra rlbench --group dev`."
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


def ensure_current_python_on_path() -> None:
    """Let CoppeliaSim's pythonLauncher.py resolve the active pg3d Python."""
    python_bin = str(Path(sys.executable).parent)
    path_parts = os.environ.get("PATH", "").split(":")
    if path_parts and path_parts[0] == python_bin:
        return
    os.environ["PATH"] = ":".join([python_bin, *[part for part in path_parts if part]])


def install_coppeliasim_python_executable_compat() -> None:
    """Point CoppeliaSim 4.9 Python add-ons at the active interpreter."""
    from cffi import FFI
    from pyrep import PyRep
    from pyrep.backend import sim

    original_run_ui_thread = PyRep._run_ui_thread
    if getattr(original_run_ui_thread, "_pg3d_python_executable_compat", False):
        return

    coppeliasim_root = os.environ.get("COPPELIASIM_ROOT")
    library_path = Path(coppeliasim_root) / "libcoppeliaSim.so" if coppeliasim_root else None
    compat_ffi = FFI()
    compat_ffi.cdef("int simSetNamedStringParam(const char* name, const char* value, int length);")
    try:
        compat_lib = compat_ffi.dlopen(str(library_path)) if library_path else None
    except OSError:
        compat_lib = None

    def set_python_executable() -> None:
        executable = sys.executable
        try:
            sim.simSetStringParameter(134, executable)
        except Exception:
            pass
        if compat_lib is None:
            return
        try:
            encoded = executable.encode("utf-8")
            compat_lib.simSetNamedStringParam(b"python", encoded, len(encoded))
        except Exception:
            pass

    def run_ui_thread_with_python(
        self: Any, scene_file: str, headless: bool, verbosity: Any
    ) -> None:
        set_python_executable()
        original_run_ui_thread(self, scene_file, headless, verbosity)

    run_ui_thread_with_python._pg3d_python_executable_compat = True  # type: ignore[attr-defined]
    PyRep._run_ui_thread = run_ui_thread_with_python


def install_coppeliasim_precision_compat() -> None:
    """Patch PyRep float wrappers for CoppeliaSim builds with double C APIs.

    Upstream PyRep's CFFI layer is generated from the CoppeliaSim 4.1-era
    single-precision ``sim.h``. CoppeliaSim 4.9 exposes double-precision
    variants for the same legacy APIs. Calling the 4.9 functions with PyRep's
    ``float*`` buffers corrupts memory, so we route the precision-sensitive
    wrappers through the ``*_D`` symbols when those symbols are available.
    """
    from cffi import FFI
    from pyrep.backend import sim

    if getattr(sim, "_pg3d_precision_compat", False):
        return

    coppeliasim_root = os.environ.get("COPPELIASIM_ROOT")
    if coppeliasim_root is None:
        return
    library_path = Path(coppeliasim_root) / "libcoppeliaSim.so"
    if not library_path.exists():
        return

    compat_ffi = FFI()
    compat_ffi.cdef(
        """
        int simGetObjectPosition_D(int objectHandle, int relativeToObjectHandle, double* position);
        int simSetObjectPosition_D(
            int objectHandle, int relativeToObjectHandle, const double* position);
        int simGetObjectOrientation_D(
            int objectHandle, int relativeToObjectHandle, double* eulerAngles);
        int simSetObjectOrientation_D(
            int objectHandle, int relativeToObjectHandle, const double* eulerAngles);
        int simGetObjectQuaternion_D(
            int objectHandle, int relativeToObjectHandle, double* quaternion);
        int simSetObjectQuaternion_D(
            int objectHandle, int relativeToObjectHandle, const double* quaternion);
        int simGetObjectMatrix_D(int objectHandle, int relativeToObjectHandle, double* matrix);
        int simSetObjectMatrix_D(
            int objectHandle, int relativeToObjectHandle, const double* matrix);
        int simGetObjectVelocity_D(
            int objectHandle, double* linearVelocity, double* angularVelocity);
        double simGetObjectSizeFactor_D(int objectHandle);
        int simGetObjectFloatParameter_D(int objectHandle, int parameterID, double* parameter);
        int simSetObjectFloatParameter_D(int objectHandle, int parameterID, double parameter);
        int simGetArrayParameter_D(int parameter, double* arrayOfValues);
        int simSetArrayParameter_D(int parameter, const double* arrayOfValues);
        int simGetFloatParameter_D(int parameter, double* floatState);
        int simSetFloatParameter_D(int parameter, double floatState);
        double simGetSimulationTime_D(void);
        double simGetSimulationTimeStep_D(void);

        int simGetJointPosition_D(int objectHandle, double* position);
        int simSetJointPosition_D(int objectHandle, double position);
        int simGetJointTargetVelocity_D(int objectHandle, double* targetVelocity);
        int simSetJointTargetVelocity_D(int objectHandle, double targetVelocity);
        int simGetJointTargetPosition_D(int objectHandle, double* targetPosition);
        int simSetJointTargetPosition_D(int objectHandle, double targetPosition);
        int simGetJointForce_D(int jointHandle, double* forceOrTorque);
        int simSetJointForce_D(int jointHandle, double forceOrTorque);
        int simGetJointMaxForce_D(int jointHandle, double* forceOrTorque);
        int simSetJointMaxForce_D(int jointHandle, double forceOrTorque);
        int simGetJointInterval_D(int objectHandle, bool* cyclic, double* interval);
        int simSetJointInterval_D(int objectHandle, bool cyclic, const double* interval);
        int simGetJointMatrix_D(int objectHandle, double* matrix);
        int simSetSphericalJointMatrix_D(int objectHandle, const double* matrix);

        int simReadForceSensor_D(int objectHandle, double* forceVector, double* torqueVector);
        int simReadProximitySensor_D(
            int sensorHandle, double* detectedPoint, int* detectedObjectHandle,
            double* normalVector);
        int simCheckProximitySensor_D(int sensorHandle, int entityHandle, double* detectedPoint);
        int simHandleVisionSensor_D(int sensorHandle, double** auxValues, int** auxValuesCount);
        int simReadVisionSensor_D(int sensorHandle, double** auxValues, int** auxValuesCount);
        void simReleaseBuffer(char* buffer);
        int simCreateDummy_D(double size, const float* reserved);
        int simCreateForceSensor_D(
            int options, const int* intParams, const double* floatParams, const double* reserved);
        int simCreateVisionSensor_D(
            int options, const int* intParams, const double* floatParams, const double* reserved);
        int simRotateAroundAxis_D(
            const double* matrixIn, const double* axis, const double* axisPos,
            double angle, double* matrixOut);
        int simInvertMatrix_D(double* matrix);
        int simMultiplyMatrices_D(
            const double* matrixIn1, const double* matrixIn2, double* matrixOut);
        int simGetEulerAnglesFromMatrix_D(const double* matrix, double* eulerAngles);
        int simCheckDistance_D(
            int entity1Handle, int entity2Handle, double threshold, double* distanceData);
        """
    )
    try:
        compat_lib = compat_ffi.dlopen(str(library_path))
        _ = compat_lib.simGetJointPosition_D
    except (AttributeError, OSError):
        return

    def check_return(ret: int | float) -> None:
        sim._check_return(ret)

    def check_object_parameter(ret: int) -> None:
        sim._check_set_object_parameter(ret)
        sim._check_return(ret)

    def double_array(values: Sequence[float], size: int | None = None) -> Any:
        vals = [float(value) for value in values]
        if size is not None and len(vals) != size:
            raise ValueError(f"expected {size} values, got {len(vals)}")
        return compat_ffi.new(f"double[{len(vals)}]", vals)

    def optional_double_array(values: Sequence[float] | None) -> Any:
        if values is None:
            return compat_ffi.NULL
        return double_array(values)

    def sim_get_object_position(object_handle: int, relative_to_object_handle: int) -> list[float]:
        position = compat_ffi.new("double[3]")
        ret = compat_lib.simGetObjectPosition_D(
            object_handle, relative_to_object_handle, position
        )
        check_return(ret)
        return list(position)

    def sim_set_object_position(
        object_handle: int, relative_to_object_handle: int, position: Sequence[float]
    ) -> None:
        ret = compat_lib.simSetObjectPosition_D(
            object_handle, relative_to_object_handle, double_array(position, 3)
        )
        check_return(ret)

    def sim_get_object_orientation(
        object_handle: int, relative_to_object_handle: int
    ) -> list[float]:
        euler_angles = compat_ffi.new("double[3]")
        ret = compat_lib.simGetObjectOrientation_D(
            object_handle, relative_to_object_handle, euler_angles
        )
        check_return(ret)
        return list(euler_angles)

    def sim_set_object_orientation(
        object_handle: int,
        relative_to_object_handle: int,
        euler_angles: Sequence[float],
    ) -> None:
        ret = compat_lib.simSetObjectOrientation_D(
            object_handle, relative_to_object_handle, double_array(euler_angles, 3)
        )
        check_return(ret)

    def sim_get_object_quaternion(
        object_handle: int, relative_to_object_handle: int
    ) -> list[float]:
        quaternion = compat_ffi.new("double[4]")
        ret = compat_lib.simGetObjectQuaternion_D(
            object_handle, relative_to_object_handle, quaternion
        )
        check_return(ret)
        return list(quaternion)

    def sim_set_object_quaternion(
        object_handle: int,
        relative_to_object_handle: int,
        quaternion: Sequence[float],
    ) -> None:
        ret = compat_lib.simSetObjectQuaternion_D(
            object_handle, relative_to_object_handle, double_array(quaternion, 4)
        )
        check_return(ret)

    def sim_get_object_matrix(object_handle: int, relative_to_object_handle: int) -> list[float]:
        matrix = compat_ffi.new("double[12]")
        ret = compat_lib.simGetObjectMatrix_D(object_handle, relative_to_object_handle, matrix)
        check_return(ret)
        return list(matrix)

    def sim_set_object_matrix(
        object_handle: int, relative_to_object_handle: int, matrix: Sequence[float]
    ) -> None:
        ret = compat_lib.simSetObjectMatrix_D(
            object_handle, relative_to_object_handle, double_array(matrix, 12)
        )
        check_return(ret)

    def sim_get_object_velocity(object_handle: int) -> tuple[list[float], list[float]]:
        linear_velocity = compat_ffi.new("double[3]")
        angular_velocity = compat_ffi.new("double[3]")
        ret = compat_lib.simGetObjectVelocity_D(object_handle, linear_velocity, angular_velocity)
        check_return(ret)
        return list(linear_velocity), list(angular_velocity)

    def sim_get_object_float_parameter(object_handle: int, parameter: int) -> float:
        value = compat_ffi.new("double *")
        ret = compat_lib.simGetObjectFloatParameter_D(object_handle, parameter, value)
        check_object_parameter(ret)
        return float(value[0])

    def sim_set_object_float_parameter(object_handle: int, parameter: int, value: float) -> None:
        ret = compat_lib.simSetObjectFloatParameter_D(object_handle, parameter, float(value))
        check_object_parameter(ret)

    def sim_get_array_parameter(parameter: int) -> list[float]:
        values = compat_ffi.new("double[3]")
        ret = compat_lib.simGetArrayParameter_D(parameter, values)
        check_return(ret)
        return list(values)

    def sim_set_array_parameter(parameter: int, values: Sequence[float]) -> None:
        ret = compat_lib.simSetArrayParameter_D(parameter, double_array(values, 3))
        check_return(ret)

    def sim_get_float_parameter(parameter: int) -> float:
        value = compat_ffi.new("double *")
        ret = compat_lib.simGetFloatParameter_D(parameter, value)
        check_return(ret)
        return float(value[0])

    def sim_set_float_parameter(parameter: int, value: float) -> None:
        ret = compat_lib.simSetFloatParameter_D(parameter, float(value))
        check_return(ret)

    def sim_get_joint_position(joint_handle: int) -> float:
        position = compat_ffi.new("double *")
        ret = compat_lib.simGetJointPosition_D(joint_handle, position)
        check_return(ret)
        return float(position[0])

    def sim_set_joint_position(joint_handle: int, position: float) -> None:
        ret = compat_lib.simSetJointPosition_D(joint_handle, float(position))
        check_return(ret)

    def sim_get_joint_target_velocity(joint_handle: int) -> float:
        velocity = compat_ffi.new("double *")
        ret = compat_lib.simGetJointTargetVelocity_D(joint_handle, velocity)
        check_return(ret)
        return float(velocity[0])

    def sim_set_joint_target_velocity(joint_handle: int, target_velocity: float) -> None:
        ret = compat_lib.simSetJointTargetVelocity_D(joint_handle, float(target_velocity))
        check_return(ret)

    def sim_get_joint_target_position(joint_handle: int) -> float:
        position = compat_ffi.new("double *")
        ret = compat_lib.simGetJointTargetPosition_D(joint_handle, position)
        check_return(ret)
        return float(position[0])

    def sim_set_joint_target_position(joint_handle: int, target_position: float) -> None:
        ret = compat_lib.simSetJointTargetPosition_D(joint_handle, float(target_position))
        check_return(ret)

    def sim_get_joint_force(joint_handle: int) -> float:
        force = compat_ffi.new("double *")
        ret = compat_lib.simGetJointForce_D(joint_handle, force)
        check_return(ret)
        if ret == 0:
            raise RuntimeError("No value available yet.")
        return float(force[0])

    def sim_set_joint_force(joint_handle: int, force: float) -> None:
        ret = compat_lib.simSetJointForce_D(joint_handle, float(force))
        check_return(ret)

    def sim_get_joint_max_force(joint_handle: int) -> float:
        force = compat_ffi.new("double *")
        ret = compat_lib.simGetJointMaxForce_D(joint_handle, force)
        check_return(ret)
        if ret == 0:
            raise RuntimeError("No value available yet.")
        return float(force[0])

    def sim_set_joint_max_force(joint_handle: int, force: float) -> None:
        ret = compat_lib.simSetJointMaxForce_D(joint_handle, float(force))
        check_return(ret)

    def sim_get_joint_interval(joint_handle: int) -> tuple[bool, list[float]]:
        cyclic = compat_ffi.new("bool *")
        interval = compat_ffi.new("double[2]")
        ret = compat_lib.simGetJointInterval_D(joint_handle, cyclic, interval)
        check_return(ret)
        return bool(cyclic[0]), list(interval)

    def sim_set_joint_interval(joint_handle: int, cyclic: bool, interval: Sequence[float]) -> None:
        ret = compat_lib.simSetJointInterval_D(joint_handle, cyclic, double_array(interval, 2))
        check_return(ret)

    def sim_get_joint_matrix(joint_handle: int) -> list[float]:
        matrix = compat_ffi.new("double[12]")
        ret = compat_lib.simGetJointMatrix_D(joint_handle, matrix)
        check_return(ret)
        return list(matrix)

    def sim_set_spherical_joint_matrix(joint_handle: int, matrix: Sequence[float]) -> None:
        ret = compat_lib.simSetSphericalJointMatrix_D(joint_handle, double_array(matrix, 12))
        check_return(ret)

    def sim_read_force_sensor(force_sensor_handle: int) -> tuple[int, list[float], list[float]]:
        force_vector = compat_ffi.new("double[3]")
        torque_vector = compat_ffi.new("double[3]")
        state = compat_lib.simReadForceSensor_D(force_sensor_handle, force_vector, torque_vector)
        check_return(state)
        return int(state), list(force_vector), list(torque_vector)

    def sim_read_proximity_sensor(sensor_handle: int) -> tuple[int, Any, list[float], list[float]]:
        detected_point = compat_ffi.new("double[3]")
        detected_object_handle = compat_ffi.new("int *")
        normal_vector = compat_ffi.new("double[3]")
        state = compat_lib.simReadProximitySensor_D(
            sensor_handle, detected_point, detected_object_handle, normal_vector
        )
        check_return(state)
        return int(state), detected_object_handle, list(detected_point), list(normal_vector)

    def sim_check_proximity_sensor(
        sensor_handle: int, entity_handle: int
    ) -> tuple[int, list[float]]:
        detected_point = compat_ffi.new("double[3]")
        state = compat_lib.simCheckProximitySensor_D(sensor_handle, entity_handle, detected_point)
        check_return(state)
        return int(state), list(detected_point)

    def sim_handle_vision_sensor(sensor_handle: int) -> tuple[int, list[float]]:
        aux_values = compat_ffi.new("double **")
        aux_values_count = compat_ffi.new("int **")
        ret = compat_lib.simHandleVisionSensor_D(sensor_handle, aux_values, aux_values_count)
        check_return(ret)
        if aux_values == compat_ffi.NULL or aux_values[0] == compat_ffi.NULL:
            return int(ret), []
        k1 = 0
        out_aux_values = []
        for i in range(aux_values_count[0][0]):
            k2 = k1 + aux_values_count[0][i + 1]
            out_aux_values.extend([x for x in aux_values[0][k1:k2]])
            k1 = k2
        compat_lib.simReleaseBuffer(compat_ffi.cast("char *", aux_values[0]))
        compat_lib.simReleaseBuffer(compat_ffi.cast("char *", aux_values_count[0]))
        return int(ret), out_aux_values

    def sim_read_vision_sensor(sensor_handle: int) -> tuple[int, list[list[float]]]:
        aux_values = compat_ffi.new("double **")
        aux_values_count = compat_ffi.new("int **")
        state = compat_lib.simReadVisionSensor_D(sensor_handle, aux_values, aux_values_count)
        if state != 0 or aux_values == compat_ffi.NULL or aux_values[0] == compat_ffi.NULL:
            return int(state), []
        offset = 0
        values = []
        for i in range(aux_values_count[0][0]):
            count = aux_values_count[0][i + 1]
            values.append([x for x in aux_values[0][offset : offset + count]])
            offset += count
        compat_lib.simReleaseBuffer(compat_ffi.cast("char *", aux_values[0]))
        compat_lib.simReleaseBuffer(compat_ffi.cast("char *", aux_values_count[0]))
        return int(state), values

    def sim_create_dummy(size: float, color: Sequence[float] | None) -> int:
        if color is not None:
            raise ValueError("CoppeliaSim 4.9 dummy color is not supported by pg3d compat")
        ret = compat_lib.simCreateDummy_D(float(size), compat_ffi.NULL)
        check_return(ret)
        return int(ret)

    def sim_create_force_sensor(
        options: int,
        intParams: Sequence[int],
        floatParams: Sequence[float],
        color: Sequence[float] | None,
    ) -> int:
        ret = compat_lib.simCreateForceSensor_D(
            options,
            compat_ffi.new("int[]", intParams),
            double_array(floatParams),
            optional_double_array(color),
        )
        check_return(ret)
        return int(ret)

    def sim_create_vision_sensor(
        options: int,
        intParams: Sequence[int],
        floatParams: Sequence[float],
        color: Sequence[float] | None,
    ) -> int:
        ret = compat_lib.simCreateVisionSensor_D(
            options,
            compat_ffi.new("int[]", intParams),
            double_array(floatParams),
            optional_double_array(color),
        )
        check_return(ret)
        return int(ret)

    def sim_rotate_around_axis(
        matrix: Sequence[float],
        axis: Sequence[float],
        axis_pos: Sequence[float],
        angle: float,
    ) -> list[float]:
        matrix_out = compat_ffi.new("double[12]")
        ret = compat_lib.simRotateAroundAxis_D(
            double_array(matrix, 12), double_array(axis, 3), double_array(axis_pos, 3),
            float(angle), matrix_out
        )
        check_return(ret)
        return list(matrix_out)

    def sim_invert_matrix(matrix: Sequence[float]) -> list[float]:
        c_matrix = double_array(matrix, 12)
        ret = compat_lib.simInvertMatrix_D(c_matrix)
        check_return(ret)
        return list(c_matrix)

    def sim_multiply_matrices(matrix1: Sequence[float], matrix2: Sequence[float]) -> list[float]:
        matrix_out = compat_ffi.new("double[12]")
        ret = compat_lib.simMultiplyMatrices_D(
            double_array(matrix1, 12), double_array(matrix2, 12), matrix_out
        )
        check_return(ret)
        return list(matrix_out)

    def sim_get_euler_angles_from_matrix(matrix: Sequence[float]) -> list[float]:
        euler_angles = compat_ffi.new("double[3]")
        ret = compat_lib.simGetEulerAnglesFromMatrix_D(double_array(matrix, 12), euler_angles)
        check_return(ret)
        return list(euler_angles)

    def sim_check_distance(
        entity1_handle: int, entity2_handle: int, threshold: float
    ) -> list[float]:
        distance_data = compat_ffi.new("double[7]")
        ret = compat_lib.simCheckDistance_D(
            entity1_handle, entity2_handle, float(threshold), distance_data
        )
        check_return(ret)
        return list(distance_data)

    sim.simGetObjectPosition = sim_get_object_position
    sim.simSetObjectPosition = sim_set_object_position
    sim.simGetObjectOrientation = sim_get_object_orientation
    sim.simSetObjectOrientation = sim_set_object_orientation
    sim.simGetObjectQuaternion = sim_get_object_quaternion
    sim.simSetObjectQuaternion = sim_set_object_quaternion
    sim.simGetObjectMatrix = sim_get_object_matrix
    sim.simSetObjectMatrix = sim_set_object_matrix
    sim.simGetObjectVelocity = sim_get_object_velocity
    sim.simGetObjectFloatParameter = sim_get_object_float_parameter
    sim.simSetObjectFloatParameter = sim_set_object_float_parameter
    sim.simGetObjectSizeFactor = lambda handle: float(compat_lib.simGetObjectSizeFactor_D(handle))
    sim.simGetArrayParameter = sim_get_array_parameter
    sim.simSetArrayParameter = sim_set_array_parameter
    sim.simGetFloatParameter = sim_get_float_parameter
    sim.simSetFloatParameter = sim_set_float_parameter
    sim.simGetSimulationTime = lambda: float(compat_lib.simGetSimulationTime_D())
    sim.simGetSimulationTimeStep = lambda: float(compat_lib.simGetSimulationTimeStep_D())

    sim.simGetJointPosition = sim_get_joint_position
    sim.simSetJointPosition = sim_set_joint_position
    sim.simGetJointTargetVelocity = sim_get_joint_target_velocity
    sim.simSetJointTargetVelocity = sim_set_joint_target_velocity
    sim.simGetJointTargetPosition = sim_get_joint_target_position
    sim.simSetJointTargetPosition = sim_set_joint_target_position
    sim.simGetJointForce = sim_get_joint_force
    sim.simSetJointForce = sim_set_joint_force
    sim.simGetJointMaxForce = sim_get_joint_max_force
    sim.simSetJointMaxForce = sim_set_joint_max_force
    sim.simGetJointInterval = sim_get_joint_interval
    sim.simSetJointInterval = sim_set_joint_interval
    sim.simGetJointMatrix = sim_get_joint_matrix
    sim.simSetSphericalJointMatrix = sim_set_spherical_joint_matrix

    sim.simReadForceSensor = sim_read_force_sensor
    sim.simReadProximitySensor = sim_read_proximity_sensor
    sim.simCheckProximitySensor = sim_check_proximity_sensor
    sim.simHandleVisionSensor = sim_handle_vision_sensor
    sim.simReadVisionSensor = sim_read_vision_sensor
    sim.simCreateDummy = sim_create_dummy
    sim.simCreateForceSensor = sim_create_force_sensor
    sim.simCreateVisionSensor = sim_create_vision_sensor
    sim.simRotateAroundAxis = sim_rotate_around_axis
    sim.simInvertMatrix = sim_invert_matrix
    sim.simMultiplyMatrices = sim_multiply_matrices
    sim.simGetEulerAnglesFromMatrix = sim_get_euler_angles_from_matrix
    sim.simCheckDistance = sim_check_distance
    sim._pg3d_precision_compat = True


def disable_waypoint_validation_for_observation(task_env: Any) -> None:
    """Skip demo waypoint feasibility checks for observation-only smoke paths."""
    task = getattr(task_env, "_task", None)
    if task is None:
        return
    task.validate = lambda: None


def print_setup_report(errors: Sequence[str], warnings: Sequence[str]) -> None:
    if errors:
        print("RLBench setup check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
    if warnings:
        print("RLBench setup warnings:", file=sys.stderr)
        for warning in warnings:
            print(f"- {warning}", file=sys.stderr)
    if errors:
        print(
            "\nExpected setup shape:\n"
            "  export COPPELIASIM_ROOT=/path/to/CoppeliaSim_Edu_V4_9_0_rev6_Ubuntu22_04\n"
            "  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT\n"
            "  export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT\n"
            "  uv sync --extra cu129 --extra rlbench --group dev\n",
            file=sys.stderr,
        )


def install_coppeliasim_scene_load_compat(
    expected_handles: Sequence[str] = ("Panda", "workspace"),
) -> None:
    """Patch PyRep scene launch for CoppeliaSim builds that ignore scene files.

    CoppeliaSim 4.9 can start through PyRep while leaving the default scene
    loaded, or crash later, when ``PyRep.launch(scene_file=...)`` receives
    RLBench's ``task_design.ttt``. Launching CoppeliaSim first and then calling
    ``simLoadScene`` works across the observed 4.9 path, so this wrapper uses
    that path for non-empty scene files.
    """
    from pyrep import PyRep
    from pyrep.backend import sim
    from pyrep.objects import Object

    original_launch = PyRep.launch
    if getattr(original_launch, "_pg3d_scene_load_compat", False):
        return

    def launch_with_scene_load_compat(
        self: Any, scene_file: str = "", *args: Any, **kwargs: Any
    ) -> None:
        abs_scene_file = os.path.abspath(scene_file) if scene_file else ""
        launch_scene_file = "" if abs_scene_file else scene_file
        original_launch(self, launch_scene_file, *args, **kwargs)
        if not abs_scene_file or not os.path.isfile(abs_scene_file):
            return
        if all(Object.exists(handle_name) for handle_name in expected_handles):
            return
        sim.simLoadScene(abs_scene_file)
        self.step()

    launch_with_scene_load_compat._pg3d_scene_load_compat = True  # type: ignore[attr-defined]
    PyRep.launch = launch_with_scene_load_compat


def load_reach_target_runtime() -> dict[str, Any]:
    """Import the RLBench/PyRep objects needed for ReachTarget scripts lazily."""
    ensure_current_python_on_path()

    from pyrep.const import ObjectType
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointVelocity
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment
    from rlbench.observation_config import ObservationConfig
    from rlbench.tasks import ReachTarget

    install_coppeliasim_python_executable_compat()
    install_coppeliasim_precision_compat()
    install_coppeliasim_scene_load_compat()

    return {
        "MoveArmThenGripper": MoveArmThenGripper,
        "JointVelocity": JointVelocity,
        "Discrete": Discrete,
        "Environment": Environment,
        "ObservationConfig": ObservationConfig,
        "ObjectType": ObjectType,
        "ReachTarget": ReachTarget,
    }
