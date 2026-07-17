"""Unit tests for comprehensive test runner helpers."""

import os
import sys
import types
import unittest
from unittest import mock


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from test_reports import comprehensive_test_runner


class TestResolveUnitTestModules(unittest.TestCase):
    def test_resolve_unit_test_modules_imports_only_requested_files(self):
        requested_files = [
            'test_cache_contract_admin_units.py',
            'test_ml_builder_units.py',
        ]

        with mock.patch.object(
            comprehensive_test_runner,
            '_import_unit_test_module',
            side_effect=lambda module_stem: f'module:{module_stem}',
        ) as import_module:
            resolved = comprehensive_test_runner._resolve_unit_test_modules(requested_files)

        self.assertEqual(
            resolved,
            ['module:test_cache_contract_admin_units', 'module:test_ml_builder_units'],
        )
        self.assertEqual(
            [call.args[0] for call in import_module.call_args_list],
            ['test_cache_contract_admin_units', 'test_ml_builder_units'],
        )


class TestRunUnitTests(unittest.TestCase):
    def test_run_unit_tests_uses_selected_unit_files(self):
        fake_module = types.ModuleType('fake_unit_module')
        fake_result = types.SimpleNamespace(testsRun=0, failures=[], errors=[])

        with mock.patch.object(
            comprehensive_test_runner,
            '_resolve_unit_test_modules',
            return_value=[fake_module],
        ) as resolve_modules, mock.patch(
            'test_reports.comprehensive_test_runner.unittest.TestLoader.loadTestsFromModule',
            return_value=unittest.TestSuite(),
        ) as load_tests, mock.patch(
            'test_reports.comprehensive_test_runner.unittest.TextTestRunner.run',
            return_value=fake_result,
        ) as run_tests:
            result = comprehensive_test_runner.run_unit_tests(
                verbose=False,
                unit_files=['test_cache_contract_admin_units.py'],
            )

        self.assertIs(result, fake_result)
        resolve_modules.assert_called_once_with(unit_files=['test_cache_contract_admin_units.py'])
        load_tests.assert_called_once_with(fake_module)
        run_tests.assert_called_once()


if __name__ == '__main__':
    unittest.main(verbosity=2)