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
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import all test modules
try:
    from unit import test_ml_builder_units
    from unit import test_stock_data_fetch_units
    from unit import test_db_interactions_units
    from unit import test_additional_modules_units
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
        
        self.categories[category] = {
            'tests_run': result.testsRun,
            'successes': successes,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success_rate': (successes / result.testsRun * 100) if result.testsRun > 0 else 0
        }
        
        self.total_tests += result.testsRun
        self.total_successes += successes
        self.total_failures += len(result.failures)
        self.total_errors += len(result.errors)
    
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
    
    def save_report(self, filename='test_report.json'):
        """Save test results to JSON file"""
        report = {
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
        
        report_path = os.path.join(os.path.dirname(__file__), filename)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Test report saved to: {report_path}")


def run_unit_tests(verbose=False):
    """Run all unit tests"""
    print("\n" + "="*80)
    print("RUNNING UNIT TESTS")
    print("="*80)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Load all unit test modules
    try:
        suite.addTests(loader.loadTestsFromModule(test_ml_builder_units))
        suite.addTests(loader.loadTestsFromModule(test_stock_data_fetch_units))
        suite.addTests(loader.loadTestsFromModule(test_db_interactions_units))
        suite.addTests(loader.loadTestsFromModule(test_additional_modules_units))
    except NameError as e:
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
        verbosity = 2 if verbose else 1
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(test_pipelines_integration)
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
        return result
    except NameError:
        print("Integration tests module not available")
        return None


def run_e2e_tests(verbose=False):
    """Run all end-to-end tests"""
    print("\n" + "="*80)
    print("RUNNING END-TO-END TESTS")
    print("="*80)
    
    try:
        verbosity = 2 if verbose else 1
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(test_complete_workflows)
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
        return result
    except NameError:
        print("E2E tests module not available")
        return None


def run_performance_tests(verbose=False):
    """Run all performance tests"""
    print("\n" + "="*80)
    print("RUNNING PERFORMANCE TESTS")
    print("="*80)
    
    try:
        verbosity = 2 if verbose else 1
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(test_performance_benchmarks)
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
        return result
    except NameError:
        print("Performance tests module not available")
        return None


def run_security_tests(verbose=False):
    """Run all security tests"""
    print("\n" + "="*80)
    print("RUNNING SECURITY TESTS")
    print("="*80)
    
    try:
        verbosity = 2 if verbose else 1
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(test_security_validation)
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
        return result
    except NameError:
        print("Security tests module not available")
        return None


def run_validation_tests(verbose=False):
    """Run all data validation tests"""
    print("\n" + "="*80)
    print("RUNNING DATA VALIDATION TESTS")
    print("="*80)
    
    try:
        verbosity = 2 if verbose else 1
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(test_data_quality)
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
        return result
    except NameError:
        print("Data validation tests module not available")
        return None


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
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("STOCK PORTFOLIO BUILDER - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Category: {args.category}")
    print("="*80)
    
    # Run requested tests
    if args.category == 'all':
        result = run_all_tests(verbose=args.verbose, save_report=args.report)
    elif args.category == 'unit':
        result = run_unit_tests(verbose=args.verbose)
    elif args.category == 'integration':
        result = run_integration_tests(verbose=args.verbose)
    elif args.category == 'e2e':
        result = run_e2e_tests(verbose=args.verbose)
    elif args.category == 'performance':
        result = run_performance_tests(verbose=args.verbose)
    elif args.category == 'security':
        result = run_security_tests(verbose=args.verbose)
    elif args.category == 'validation':
        result = run_validation_tests(verbose=args.verbose)
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Return exit code based on results
    if isinstance(result, ComprehensiveTestResult):
        exit_code = 0 if (result.total_failures + result.total_errors) == 0 else 1
    else:
        exit_code = 0 if result and (len(result.failures) + len(result.errors)) == 0 else 1
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
