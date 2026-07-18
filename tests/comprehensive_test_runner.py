"""
Comprehensive Master Test Suite for Stock Portfolio Builder

This module provides a unified test runner that executes all test categories:
- Unit Tests
- Integration Tests
- End-to-End Tests
- Performance Tests
- Security Tests
- Data Validation Tests

Usage:
    python comprehensive_test_runner.py [--category CATEGORY] [--verbose] [--report]

Categories:
    all          - Run all tests (default)
    unit         - Run only unit tests
    integration  - Run only integration tests
    e2e          - Run only end-to-end tests
    performance  - Run only performance tests
    security     - Run only security tests
    validation   - Run only data validation tests
"""

import unittest
import sys
import os
import argparse
import subprocess
import tempfile
from datetime import datetime
import json
import io
import importlib

# Force UTF-8 encoding globally to handle Unicode characters on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from tests.test_runner_envs import (
        CATEGORY_ENVIRONMENTS,
        TEST_REPORTS_DIR,
        default_unit_env_plan,
        resolve_python_executable,
    )
except ModuleNotFoundError:
    from test_runner_envs import (
        CATEGORY_ENVIRONMENTS,
        TEST_REPORTS_DIR,
        default_unit_env_plan,
        resolve_python_executable,
    )

# Import all test modules
try:
    from unit import test_ml_builder_units
    from unit import test_stock_data_fetch_units
    from unit import test_db_interactions_units
    from unit import test_additional_modules_units
    from unit import test_runtime_compat_units
except ImportError:
    print("Warning: Some unit test modules could not be imported")

try:
    from integration import test_pipelines_integration
except ImportError:
    print("Warning: Integration test module could not be imported")

try:
    from e2e import test_complete_workflows
except ImportError:
    print("Warning: E2E test module could not be imported")

try:
    from performance import test_performance_benchmarks
except ImportError:
    print("Warning: Performance test module could not be imported")

try:
    from security import test_security_validation
except ImportError:
    print("Warning: Security test module could not be imported")

try:
    from data_validation import test_data_quality
except ImportError:
    print("Warning: Data validation test module could not be imported")


def _import_unit_test_module(module_stem):
    last_error = None
    for module_name in (
        f'tests.unit.{module_stem}',
        f'unit.{module_stem}',
        module_stem,
    ):
        try:
            return importlib.import_module(module_name)
        except ImportError as error:
            last_error = error

    raise last_error


def _resolve_unit_test_modules(unit_files=None):
    if unit_files:
        return [
            _import_unit_test_module(os.path.splitext(os.path.basename(unit_file))[0])
            for unit_file in unit_files
        ]

    loaded_modules = []
    for module_name in (
        'test_ml_builder_units',
        'test_stock_data_fetch_units',
        'test_db_interactions_units',
        'test_additional_modules_units',
        'test_runtime_compat_units',
    ):
        module = globals().get(module_name)
        if module is not None:
            loaded_modules.append(module)

    return loaded_modules


class ComprehensiveTestResult:
    """Container for comprehensive test results"""
    
    def __init__(self):
        self.categories = {}
        self.start_time = None
        self.end_time = None
        self.total_tests = 0
        self.total_successes = 0
        self.total_failures = 0
        self.total_errors = 0
    
    def add_category_result(self, category, result):
        """Add results from a test category"""
        successes = result.testsRun - len(result.failures) - len(result.errors)
        self.add_category_stats(
            category,
            tests_run=result.testsRun,
            failures=len(result.failures),
            errors=len(result.errors),
        )

    def add_category_stats(self, category, tests_run, failures, errors):
        """Add summary stats from a test category or subprocess report."""
        successes = tests_run - failures - errors
        
        existing = self.categories.get(
            category,
            {
                'tests_run': 0,
                'successes': 0,
                'failures': 0,
                'errors': 0,
                'success_rate': 0,
            },
        )

        existing['tests_run'] += tests_run
        existing['successes'] += successes
        existing['failures'] += failures
        existing['errors'] += errors
        existing['success_rate'] = (
            existing['successes'] / existing['tests_run'] * 100
        ) if existing['tests_run'] > 0 else 0

        self.categories[category] = existing
        
        self.total_tests += tests_run
        self.total_successes += successes
        self.total_failures += failures
        self.total_errors += errors

    def absorb_report(self, report):
        """Merge a saved subprocess report into this result."""
        for category, stats in report.get('categories', {}).items():
            self.add_category_stats(
                category,
                tests_run=stats.get('tests_run', 0),
                failures=stats.get('failures', 0),
                errors=stats.get('errors', 0),
            )
    
    def print_summary(self):
        """Print comprehensive summary"""
        print("\n" + "="*80)
        print("COMPREHENSIVE TEST SUITE SUMMARY")
        print("="*80)
        
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
            print(f"Duration: {duration:.2f} seconds")
        
        print(f"\nTotal Tests Run: {self.total_tests}")
        print(f"Total Successes: {self.total_successes}")
        print(f"Total Failures: {self.total_failures}")
        print(f"Total Errors: {self.total_errors}")
        
        if self.total_tests > 0:
            success_rate = (self.total_successes / self.total_tests * 100)
            print(f"Overall Success Rate: {success_rate:.1f}%")
        
        print("\n" + "-"*80)
        print("RESULTS BY CATEGORY")
        print("-"*80)
        
        for category, stats in self.categories.items():
            print(f"\n{category.upper()}:")
            print(f"  Tests Run: {stats['tests_run']}")
            print(f"  Successes: {stats['successes']}")
            print(f"  Failures: {stats['failures']}")
            print(f"  Errors: {stats['errors']}")
            print(f"  Success Rate: {stats['success_rate']:.1f}%")
        
        print("\n" + "="*80)
        
        # Overall verdict
        if self.total_failures == 0 and self.total_errors == 0:
            print("✓ ALL TESTS PASSED!")
        elif self.total_failures + self.total_errors < 5:
            print("⚠ MOSTLY PASSING (minor issues)")
        else:
            print("✗ TESTS FAILED (review needed)")
        
        print("="*80)
    
    def to_dict(self):
        return {
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else 0,
            'summary': {
                'total_tests': self.total_tests,
                'total_successes': self.total_successes,
                'total_failures': self.total_failures,
                'total_errors': self.total_errors,
                'success_rate': (self.total_successes / self.total_tests * 100) if self.total_tests > 0 else 0
            },
            'categories': self.categories
        }

    def save_report(self, filename='test_report.json', report_path=None):
        """Save test results to JSON file"""
        report = self.to_dict()

        if report_path is None:
            report_path = os.path.join(os.path.dirname(__file__), filename)

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Test report saved to: {report_path}")


def run_unit_tests(verbose=False, unit_files=None):
    """Run all unit tests"""
    print("\n" + "="*80)
    print("RUNNING UNIT TESTS")
    print("="*80)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    try:
        for module in _resolve_unit_test_modules(unit_files=unit_files):
            suite.addTests(loader.loadTestsFromModule(module))
    except (ImportError, NameError) as e:
        print(f"Warning: Could not load some unit test modules: {e}")
    
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result


def run_integration_tests(verbose=False):
    """Run all integration tests"""
    print("\n" + "="*80)
    print("RUNNING INTEGRATION TESTS")
    print("="*80)
    
    try:
        from tests.integration import test_pipelines_integration

        verbosity = 2 if verbose else 1
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(test_pipelines_integration)
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
        return result
    except ImportError:
        print("Integration tests module not available")
        return None


def run_e2e_tests(verbose=False):
    """Run all end-to-end tests"""
    print("\n" + "="*80)
    print("RUNNING END-TO-END TESTS")
    print("="*80)
    
    try:
        from tests.e2e import test_complete_workflows

        verbosity = 2 if verbose else 1
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(test_complete_workflows)
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
        return result
    except ImportError:
        print("E2E tests module not available")
        return None


def run_performance_tests(verbose=False):
    """Run all performance tests"""
    print("\n" + "="*80)
    print("RUNNING PERFORMANCE TESTS")
    print("="*80)
    
    try:
        from tests.performance import test_performance_benchmarks

        verbosity = 2 if verbose else 1
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(test_performance_benchmarks)
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
        return result
    except ImportError:
        print("Performance tests module not available")
        return None


def run_security_tests(verbose=False):
    """Run all security tests"""
    print("\n" + "="*80)
    print("RUNNING SECURITY TESTS")
    print("="*80)
    
    try:
        from tests.security import test_security_validation

        verbosity = 2 if verbose else 1
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(test_security_validation)
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
        return result
    except ImportError:
        print("Security tests module not available")
        return None


def run_validation_tests(verbose=False):
    """Run all data validation tests"""
    print("\n" + "="*80)
    print("RUNNING DATA VALIDATION TESTS")
    print("="*80)
    
    try:
        from tests.data_validation import test_data_quality

        verbosity = 2 if verbose else 1
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(test_data_quality)
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
        return result
    except ImportError:
        print("Data validation tests module not available")
        return None


def _wrap_single_category_result(category, result, start_time=None, end_time=None):
    wrapped_result = ComprehensiveTestResult()
    wrapped_result.start_time = start_time
    wrapped_result.end_time = end_time
    if result is not None:
        wrapped_result.add_category_result(category, result)
    return wrapped_result


def _run_requested_category(category, verbose=False, unit_files=None):
    if category == 'unit':
        return run_unit_tests(verbose=verbose, unit_files=unit_files)
    if category == 'integration':
        return run_integration_tests(verbose=verbose)
    if category == 'e2e':
        return run_e2e_tests(verbose=verbose)
    if category == 'performance':
        return run_performance_tests(verbose=verbose)
    if category == 'security':
        return run_security_tests(verbose=verbose)
    if category == 'validation':
        return run_validation_tests(verbose=verbose)
    raise ValueError(f"Unsupported category: {category}")


def _run_env_routed_category(category, target_env, verbose=False, unit_files=None):
    python_executable = resolve_python_executable(target_env)
    report_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as handle:
            report_file = handle.name

        print(f"\n[ENV] Routing {category} tests to {target_env}")
        command = [
            python_executable,
            os.path.abspath(__file__),
            '--category', category,
            '--disable-env-routing',
            '--report-file', report_file,
        ]
        if verbose:
            command.append('--verbose')
        if unit_files:
            command.extend(['--unit-files', *unit_files])

        completed = subprocess.run(
            command,
            cwd=os.path.abspath(os.path.join(TEST_REPORTS_DIR, '..')),
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
        )

        if completed.stdout:
            print(completed.stdout, end='' if completed.stdout.endswith('\n') else '\n')
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end='' if completed.stderr.endswith('\n') else '\n')

        if not report_file or not os.path.exists(report_file):
            raise RuntimeError(f"Missing subprocess report for {category} in {target_env}")

        with open(report_file, 'r', encoding='utf-8') as handle:
            report = json.load(handle)

        routed_result = ComprehensiveTestResult()
        routed_result.absorb_report(report)
        return routed_result
    finally:
        if report_file and os.path.exists(report_file):
            os.remove(report_file)


def run_unit_tests_env_aware(verbose=False, save_report=False):
    """Run unit tests across the required Python 3.10 and 3.12 environments."""
    comprehensive_result = ComprehensiveTestResult()
    comprehensive_result.start_time = datetime.now()

    for target_env, unit_files in default_unit_env_plan().items():
        if not unit_files:
            continue
        routed_result = _run_env_routed_category(
            'unit',
            target_env,
            verbose=verbose,
            unit_files=unit_files,
        )
        comprehensive_result.absorb_report(routed_result.to_dict())

    comprehensive_result.end_time = datetime.now()
    comprehensive_result.print_summary()

    if save_report:
        comprehensive_result.save_report()

    return comprehensive_result


def run_all_tests_env_aware(verbose=False, save_report=False):
    """Run all test categories in their required Python environments."""
    comprehensive_result = ComprehensiveTestResult()
    comprehensive_result.start_time = datetime.now()

    for target_env, unit_files in default_unit_env_plan().items():
        if not unit_files:
            continue
        routed_result = _run_env_routed_category(
            'unit',
            target_env,
            verbose=verbose,
            unit_files=unit_files,
        )
        comprehensive_result.absorb_report(routed_result.to_dict())

    for category, target_env in CATEGORY_ENVIRONMENTS.items():
        routed_result = _run_env_routed_category(category, target_env, verbose=verbose)
        comprehensive_result.absorb_report(routed_result.to_dict())

    comprehensive_result.end_time = datetime.now()
    comprehensive_result.print_summary()

    if save_report:
        comprehensive_result.save_report()

    return comprehensive_result


def run_all_tests(verbose=False, save_report=False):
    """Run all test categories"""
    comprehensive_result = ComprehensiveTestResult()
    comprehensive_result.start_time = datetime.now()
    
    # Run each category
    test_categories = [
        ('unit', run_unit_tests),
        ('integration', run_integration_tests),
        ('e2e', run_e2e_tests),
        ('performance', run_performance_tests),
        ('security', run_security_tests),
        ('validation', run_validation_tests)
    ]
    
    for category_name, test_runner in test_categories:
        try:
            result = test_runner(verbose=verbose)
            if result:
                comprehensive_result.add_category_result(category_name, result)
        except Exception as e:
            print(f"\nError running {category_name} tests: {e}")
    
    comprehensive_result.end_time = datetime.now()
    comprehensive_result.print_summary()
    
    if save_report:
        comprehensive_result.save_report()
    
    return comprehensive_result


def main():
    """Main entry point for test runner"""
    parser = argparse.ArgumentParser(
        description='Comprehensive Test Suite for Stock Portfolio Builder'
    )
    
    parser.add_argument(
        '--category',
        choices=['all', 'unit', 'integration', 'e2e', 'performance', 'security', 'validation'],
        default='all',
        help='Test category to run (default: all)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--report', '-r',
        action='store_true',
        help='Save test report to JSON file'
    )

    parser.add_argument(
        '--report-file',
        default=None,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        '--disable-env-routing',
        action='store_true',
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        '--unit-files',
        nargs='*',
        default=None,
        help=argparse.SUPPRESS,
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("STOCK PORTFOLIO BUILDER - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Category: {args.category}")
    print("="*80)
    start_time = datetime.now()

    # Run requested tests
    if args.category == 'all':
        if args.disable_env_routing:
            result = run_all_tests(verbose=args.verbose, save_report=False)
        else:
            result = run_all_tests_env_aware(verbose=args.verbose, save_report=False)
    elif args.category == 'unit' and not args.disable_env_routing and not args.unit_files:
        result = run_unit_tests_env_aware(verbose=args.verbose, save_report=False)
    elif args.category in CATEGORY_ENVIRONMENTS and not args.disable_env_routing:
        result = _run_env_routed_category(args.category, CATEGORY_ENVIRONMENTS[args.category], verbose=args.verbose)
        result.start_time = start_time
        result.end_time = datetime.now()
        result.print_summary()
    else:
        raw_result = _run_requested_category(args.category, verbose=args.verbose, unit_files=args.unit_files)
        result = _wrap_single_category_result(
            args.category,
            raw_result,
            start_time=start_time,
            end_time=datetime.now(),
        )

    report_target = args.report_file
    if report_target or args.report:
        result.save_report(report_path=report_target)
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Return exit code based on results
    exit_code = 0 if (result.total_failures + result.total_errors) == 0 else 1
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
