"""Unit tests for environment routing in the comprehensive test runner."""

import os
import sys
import unittest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tests.test_runner_envs import (
    current_python_env,
    default_unit_env_plan,
    plan_unit_test_files,
    resolve_python_executable,
)


class TestCurrentPythonEnv(unittest.TestCase):
    def test_current_python_env_recognizes_py310(self):
        version_info = type('VersionInfo', (), {'major': 3, 'minor': 10})()
        self.assertEqual(current_python_env(version_info), 'py310')

    def test_current_python_env_recognizes_py312(self):
        version_info = type('VersionInfo', (), {'major': 3, 'minor': 12})()
        self.assertEqual(current_python_env(version_info), 'py312')

    def test_current_python_env_returns_none_for_other_versions(self):
        version_info = type('VersionInfo', (), {'major': 3, 'minor': 11})()
        self.assertIsNone(current_python_env(version_info))


class TestResolvePythonExecutable(unittest.TestCase):
    def test_resolve_python_executable_prefers_current_interpreter_when_env_matches(self):
        resolved = resolve_python_executable(
            'py310',
            current_env='py310',
            current_executable='current-python.exe',
            env_vars={},
            path_exists=lambda _path: False,
        )

        self.assertEqual(resolved, 'current-python.exe')

    def test_resolve_python_executable_uses_override_when_available(self):
        resolved = resolve_python_executable(
            'py312',
            current_env='py310',
            current_executable='current-python.exe',
            env_vars={'SPB_TEST_PY312': 'override-python.exe'},
            path_exists=lambda path: path == 'override-python.exe',
        )

        self.assertEqual(resolved, 'override-python.exe')


class TestUnitEnvPlan(unittest.TestCase):
    def test_plan_unit_test_files_routes_ml_files_to_py310(self):
        plan = plan_unit_test_files([
            'test_ml_builder_units.py',
            'test_cache_contract_admin_units.py',
            'test_stock_data_fetch_units.py',
            'test_db_interactions_units.py',
        ])

        self.assertEqual(
            plan['py310'],
            ['test_ml_builder_units.py', 'test_cache_contract_admin_units.py'],
        )
        self.assertEqual(
            plan['py312'],
            ['test_stock_data_fetch_units.py', 'test_db_interactions_units.py'],
        )

    def test_default_unit_env_plan_discovers_files_before_routing(self):
        fake_files = [
            'test_db_interactions_units.py',
            'test_ml_builder_units.py',
            'README.md',
            'test_cache_contract_admin_units.py',
        ]
        plan = default_unit_env_plan(
            unit_dir='ignored',
            list_dir=lambda _path: fake_files,
        )

        self.assertEqual(plan['py310'], ['test_cache_contract_admin_units.py', 'test_ml_builder_units.py'])
        self.assertEqual(plan['py312'], ['test_db_interactions_units.py'])


if __name__ == '__main__':
    unittest.main(verbosity=2)