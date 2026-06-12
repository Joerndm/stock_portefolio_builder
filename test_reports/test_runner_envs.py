"""Helpers for routing test categories to the correct Python environment."""

from __future__ import annotations

import os
import sys


TEST_REPORTS_DIR = os.path.dirname(__file__)
UNIT_TEST_DIR = os.path.join(TEST_REPORTS_DIR, 'unit')

ENV_OVERRIDE_VARIABLES = {
    'py310': 'SPB_TEST_PY310',
    'py312': 'SPB_TEST_PY312',
}

DEFAULT_PYTHON_EXECUTABLES = {
    'py310': os.path.join('C:\\Users\\joern\\anaconda3\\envs\\tf_gpu_py_3_10', 'python.exe'),
    'py312': os.path.join('C:\\Users\\joern\\anaconda3\\envs\\fetch_stock_data_py_3_12', 'python.exe'),
}

CATEGORY_ENVIRONMENTS = {
    'integration': 'py312',
    'e2e': 'py312',
    'performance': 'py312',
    'security': 'py312',
    'validation': 'py312',
}

PY310_UNIT_TEST_FILES = {
    'test_cache_contract_admin_units.py',
    'test_gpu_runtime_utils_units.py',
    'test_ml_builder_units.py',
}


def current_python_env(version_info=None):
    """Return the active interpreter bucket used by the test runner."""
    version_info = version_info or sys.version_info
    if version_info.major == 3 and version_info.minor == 10:
        return 'py310'
    if version_info.major == 3 and version_info.minor == 12:
        return 'py312'
    return None


def resolve_python_executable(
    target_env,
    *,
    current_env=None,
    current_executable=None,
    env_vars=None,
    path_exists=os.path.exists,
):
    """Resolve the Python executable for a target env, preferring the current interpreter when it matches."""
    current_env = current_env or current_python_env()
    current_executable = current_executable or sys.executable
    env_vars = env_vars or os.environ

    if target_env == current_env:
        return current_executable

    override_var = ENV_OVERRIDE_VARIABLES[target_env]
    override_path = env_vars.get(override_var)
    if override_path and path_exists(override_path):
        return override_path

    default_path = DEFAULT_PYTHON_EXECUTABLES[target_env]
    if path_exists(default_path):
        return default_path

    raise FileNotFoundError(
        f"Could not resolve Python executable for {target_env}. "
        f"Set {override_var} or install the expected environment at {default_path}."
    )


def discover_unit_test_files(unit_dir=UNIT_TEST_DIR, list_dir=os.listdir):
    """Return unit-test filenames discovered in the unit test directory."""
    return sorted(
        filename
        for filename in list_dir(unit_dir)
        if filename.startswith('test') and filename.endswith('_units.py')
    )


def plan_unit_test_files(unit_files):
    """Split unit test files between the TensorFlow and fetch/data interpreters."""
    py310_files = [filename for filename in unit_files if filename in PY310_UNIT_TEST_FILES]
    py312_files = [filename for filename in unit_files if filename not in PY310_UNIT_TEST_FILES]
    return {
        'py310': py310_files,
        'py312': py312_files,
    }


def default_unit_env_plan(unit_dir=UNIT_TEST_DIR, list_dir=os.listdir):
    """Discover unit tests and return the default environment routing plan."""
    return plan_unit_test_files(discover_unit_test_files(unit_dir=unit_dir, list_dir=list_dir))