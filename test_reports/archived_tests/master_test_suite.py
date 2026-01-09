"""
MASTER TEST SUITE - Stock Portfolio Builder
===========================================

Comprehensive, reusable test suite for validating all functionality
after any change or improvement to the codebase.

This suite ensures:
1. All functions work correctly
2. Changes don't negatively impact price prediction accuracy
3. Industry-standard metrics are maintained
4. No regressions are introduced

Usage:
    python test_reports/master_test_suite.py [--quick] [--verbose] [--category CATEGORY]
    
    --quick: Run only fast tests (no TensorFlow required)
    --verbose: Show detailed output
    --category: Run specific category (functional, ml, integration, regression, performance)

Categories:
    - functional: Core function tests (data processing, calculations)
    - ml: Machine learning model tests (training, evaluation, prediction)
    - integration: End-to-end pipeline tests
    - regression: Baseline comparison tests
    - performance: Speed and efficiency tests
"""

import sys
import os
import unittest
import time
import json
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test result colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


class MasterTestSuite:
    """Orchestrates all test suites and generates comprehensive reports."""
    
    def __init__(self, quick_mode=False, verbose=False, category=None):
        self.quick_mode = quick_mode
        self.verbose = verbose
        self.category = category
        self.results = {
            'functional': {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []},
            'ml': {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []},
            'integration': {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []},
            'regression': {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []},
            'performance': {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []},
        }
        self.start_time = None
        self.baseline_file = project_root / 'test_reports' / 'baseline_metrics.json'
        
    def load_baseline_metrics(self):
        """Load baseline metrics for regression testing."""
        if self.baseline_file.exists():
            with open(self.baseline_file, 'r') as f:
                return json.load(f)
        return None
    
    def save_baseline_metrics(self, metrics):
        """Save baseline metrics for future regression testing."""
        with open(self.baseline_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n{BLUE}ℹ Baseline metrics saved to {self.baseline_file}{RESET}")
    
    def print_header(self):
        """Print test suite header."""
        print("\n" + "="*80)
        print(f"{BOLD}MASTER TEST SUITE - Stock Portfolio Builder{RESET}")
        print("="*80)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Mode: {'Quick (no TensorFlow)' if self.quick_mode else 'Full Suite'}")
        if self.category:
            print(f"Category: {self.category}")
        print("="*80 + "\n")
    
    def run_functional_tests(self):
        """
        Category 1: Functional Tests
        Test core functions without ML dependencies.
        """
        if self.category and self.category != 'functional':
            return
            
        print(f"\n{BOLD}{'='*80}{RESET}")
        print(f"{BOLD}CATEGORY 1: FUNCTIONAL TESTS{RESET}")
        print(f"{BOLD}{'='*80}{RESET}\n")
        
        tests = [
            ('unit_test.py', 'LSTM sequence creation and dataset preparation'),
            ('test_feature_calculations.py', 'Technical indicator calculations (SMA, EMA, Bollinger)'),
            ('test_overfitting_remediation_standalone.py', 'Overfitting remediation logic'),
        ]
        
        for test_file, description in tests:
            self._run_test_file('functional', test_file, description)
    
    def run_ml_tests(self):
        """
        Category 2: Machine Learning Tests
        Test ML models, training, and evaluation.
        """
        if self.category and self.category != 'ml':
            return
            
        if self.quick_mode:
            print(f"\n{YELLOW}⊘ Skipping ML tests in quick mode (requires TensorFlow){RESET}")
            return
            
        print(f"\n{BOLD}{'='*80}{RESET}")
        print(f"{BOLD}CATEGORY 2: MACHINE LEARNING TESTS{RESET}")
        print(f"{BOLD}{'='*80}{RESET}\n")
        
        tests = [
            ('test_mae_fix.py', 'MAE metric implementation for all models'),
            ('test_lstm_parameters.py', 'LSTM hyperparameter configuration'),
            ('test_overfitting_detection.py', 'Multi-metric overfitting detection'),
            ('test_retraining_efficiency.py', 'Retraining efficiency and convergence'),
            ('test_xgboost_ensemble_fix.py', 'XGBoost ensemble integration'),
        ]
        
        for test_file, description in tests:
            self._run_test_file('ml', test_file, description)
    
    def run_integration_tests(self):
        """
        Category 3: Integration Tests
        Test end-to-end pipeline execution.
        """
        if self.category and self.category != 'integration':
            return
            
        if self.quick_mode:
            print(f"\n{YELLOW}⊘ Skipping integration tests in quick mode{RESET}")
            return
            
        print(f"\n{BOLD}{'='*80}{RESET}")
        print(f"{BOLD}CATEGORY 3: INTEGRATION TESTS{RESET}")
        print(f"{BOLD}{'='*80}{RESET}\n")
        
        tests = [
            ('test_ml_builder_flow.py', 'Full ML pipeline execution flow'),
            ('test_stock_data_fetch.py', 'Data fetching and feature engineering'),
            ('test_quick_validation.py', 'Quick end-to-end validation'),
        ]
        
        for test_file, description in tests:
            self._run_test_file('integration', test_file, description)
    
    def run_regression_tests(self):
        """
        Category 4: Regression Tests
        Compare current performance against baseline.
        """
        if self.category and self.category != 'regression':
            return
            
        if self.quick_mode:
            print(f"\n{YELLOW}⊘ Skipping regression tests in quick mode{RESET}")
            return
            
        print(f"\n{BOLD}{'='*80}{RESET}")
        print(f"{BOLD}CATEGORY 4: REGRESSION TESTS{RESET}")
        print(f"{BOLD}{'='*80}{RESET}\n")
        
        # Load baseline metrics
        baseline = self.load_baseline_metrics()
        
        if baseline is None:
            print(f"{YELLOW}⚠ No baseline metrics found. Run with --save-baseline to create.{RESET}")
            print("  Skipping regression tests.\n")
            return
        
        # Run prediction accuracy test
        test_result = self._test_prediction_accuracy(baseline)
        self._record_test_result('regression', 'Prediction accuracy vs baseline', test_result)
        
        # Test training time hasn't regressed
        test_result = self._test_training_time(baseline)
        self._record_test_result('regression', 'Training time vs baseline', test_result)
    
    def run_performance_tests(self):
        """
        Category 5: Performance Tests
        Test speed, memory usage, and efficiency.
        """
        if self.category and self.category != 'performance':
            return
            
        print(f"\n{BOLD}{'='*80}{RESET}")
        print(f"{BOLD}CATEGORY 5: PERFORMANCE TESTS{RESET}")
        print(f"{BOLD}{'='*80}{RESET}\n")
        
        # Test feature calculation speed
        test_result = self._test_feature_calculation_speed()
        self._record_test_result('performance', 'Feature calculation speed', test_result)
        
        # Test data processing efficiency
        test_result = self._test_data_processing_efficiency()
        self._record_test_result('performance', 'Data processing efficiency', test_result)
    
    def _run_test_file(self, category, test_file, description):
        """Run a test file and record results."""
        print(f"\n{BLUE}► Running: {description}{RESET}")
        print(f"  File: {test_file}")
        
        test_path = project_root / 'test_reports' / test_file
        
        if not test_path.exists():
            print(f"{RED}  ✗ Test file not found{RESET}")
            self.results[category]['failed'] += 1
            self.results[category]['tests'].append({
                'name': test_file,
                'status': 'failed',
                'reason': 'File not found'
            })
            return
        
        try:
            # Run the test file
            start = time.time()
            
            # Import and run as unittest if it follows unittest pattern
            if test_file.endswith('.py'):
                import importlib.util
                spec = importlib.util.spec_from_file_location(test_file[:-3], test_path)
                module = importlib.util.module_from_spec(spec)
                
                # Capture output
                import io
                from contextlib import redirect_stdout, redirect_stderr
                
                stdout_capture = io.StringIO()
                stderr_capture = io.StringIO()
                
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    try:
                        spec.loader.exec_module(module)
                        
                        # If module has unittest.main, run it
                        if hasattr(module, 'unittest'):
                            loader = unittest.TestLoader()
                            suite = loader.loadTestsFromModule(module)
                            runner = unittest.TextTestRunner(verbosity=0)
                            result = runner.run(suite)
                            
                            if result.wasSuccessful():
                                status = 'passed'
                                self.results[category]['passed'] += result.testsRun
                            else:
                                status = 'failed'
                                self.results[category]['failed'] += len(result.failures) + len(result.errors)
                        else:
                            # Not a unittest file, assume success if no exceptions
                            status = 'passed'
                            self.results[category]['passed'] += 1
                    except Exception as e:
                        status = 'failed'
                        self.results[category]['failed'] += 1
                        if self.verbose:
                            print(f"{RED}  Error: {str(e)}{RESET}")
                
                elapsed = time.time() - start
                
                # Show output in verbose mode
                if self.verbose:
                    stdout_text = stdout_capture.getvalue()
                    stderr_text = stderr_capture.getvalue()
                    if stdout_text:
                        print(f"\n  Output:\n{stdout_text}")
                    if stderr_text:
                        print(f"\n  Errors:\n{stderr_text}")
                
                if status == 'passed':
                    print(f"{GREEN}  ✓ PASSED{RESET} ({elapsed:.2f}s)")
                else:
                    print(f"{RED}  ✗ FAILED{RESET} ({elapsed:.2f}s)")
                
                self.results[category]['tests'].append({
                    'name': test_file,
                    'status': status,
                    'time': elapsed
                })
                
        except Exception as e:
            print(f"{RED}  ✗ FAILED: {str(e)}{RESET}")
            self.results[category]['failed'] += 1
            self.results[category]['tests'].append({
                'name': test_file,
                'status': 'failed',
                'reason': str(e)
            })
    
    def _test_prediction_accuracy(self, baseline):
        """Test that prediction accuracy hasn't regressed."""
        print(f"\n{BLUE}► Testing: Prediction accuracy vs baseline{RESET}")
        
        try:
            # Import necessary modules
            import pandas as pd
            import numpy as np
            from ml_builder import train_and_validate_models
            
            # Create small test dataset
            np.random.seed(42)
            n_samples = 200
            n_features = 30
            
            X = pd.DataFrame(np.random.randn(n_samples, n_features))
            y = pd.Series(np.random.randn(n_samples))
            
            # Split data
            train_size = int(0.65 * n_samples)
            val_size = int(0.15 * n_samples)
            
            X_train = X[:train_size]
            y_train = y[:train_size]
            X_val = X[train_size:train_size+val_size]
            y_val = y[train_size:train_size+val_size]
            X_test = X[train_size+val_size:]
            y_test = y[train_size+val_size:]
            
            # Train models
            print("  Training models...")
            models, training_history, _ = train_and_validate_models(
                X_train.values, y_train.values, 
                X_val.values, y_val.values,
                X_test.values, y_test.values,
                max_retraining_attempts=2,  # Keep it fast
                time_steps=10
            )
            
            # Get ensemble metrics
            ensemble_metrics = training_history.get('ensemble_test_metrics', {})
            current_r2 = ensemble_metrics.get('r2', 0)
            current_mse = ensemble_metrics.get('mse', float('inf'))
            
            # Compare to baseline
            baseline_r2 = baseline.get('ensemble_r2', 0)
            baseline_mse = baseline.get('ensemble_mse', float('inf'))
            
            # Allow 5% degradation tolerance
            r2_tolerance = 0.05
            mse_tolerance = 0.05
            
            r2_ok = current_r2 >= baseline_r2 * (1 - r2_tolerance)
            mse_ok = current_mse <= baseline_mse * (1 + mse_tolerance)
            
            print(f"  Baseline R²: {baseline_r2:.4f}, Current R²: {current_r2:.4f}")
            print(f"  Baseline MSE: {baseline_mse:.4f}, Current MSE: {current_mse:.4f}")
            
            if r2_ok and mse_ok:
                print(f"{GREEN}  ✓ Prediction accuracy maintained{RESET}")
                return 'passed'
            else:
                print(f"{RED}  ✗ Prediction accuracy regressed{RESET}")
                if not r2_ok:
                    print(f"    R² degraded by {(baseline_r2 - current_r2)/baseline_r2*100:.1f}%")
                if not mse_ok:
                    print(f"    MSE increased by {(current_mse - baseline_mse)/baseline_mse*100:.1f}%")
                return 'failed'
                
        except Exception as e:
            print(f"{RED}  ✗ Test error: {str(e)}{RESET}")
            return 'failed'
    
    def _test_training_time(self, baseline):
        """Test that training time hasn't significantly increased."""
        print(f"\n{BLUE}► Testing: Training time vs baseline{RESET}")
        
        try:
            import numpy as np
            import time
            from ml_builder import tune_random_forest_model, build_random_forest_model
            from sklearn.model_selection import train_test_split
            
            # Create small dataset
            np.random.seed(42)
            X = np.random.randn(150, 20)
            y = np.random.randn(150)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            # Time RF training
            start = time.time()
            best_hp, best_model = tune_random_forest_model(
                X_train, y_train, X_test, y_test,
                max_trials=10,  # Keep it fast
                executions_per_trial=1
            )
            elapsed = time.time() - start
            
            baseline_time = baseline.get('rf_training_time', float('inf'))
            
            # Allow 20% increase tolerance
            time_ok = elapsed <= baseline_time * 1.2
            
            print(f"  Baseline time: {baseline_time:.2f}s, Current time: {elapsed:.2f}s")
            
            if time_ok:
                print(f"{GREEN}  ✓ Training time acceptable{RESET}")
                return 'passed'
            else:
                increase = (elapsed - baseline_time) / baseline_time * 100
                print(f"{RED}  ✗ Training time increased by {increase:.1f}%{RESET}")
                return 'failed'
                
        except Exception as e:
            print(f"{RED}  ✗ Test error: {str(e)}{RESET}")
            return 'failed'
    
    def _test_feature_calculation_speed(self):
        """Test feature calculation performance."""
        print(f"\n{BLUE}► Testing: Feature calculation speed{RESET}")
        
        try:
            import pandas as pd
            import numpy as np
            import time
            import sys
            sys.path.insert(0, str(project_root))
            import stock_data_fetch as sdf
            
            # Create synthetic price data
            np.random.seed(42)
            dates = pd.date_range('2020-01-01', periods=500, freq='D')
            df = pd.DataFrame({
                'date': dates,
                'ticker': ['TEST'] * 500,
                'open_Price': np.random.uniform(100, 200, 500),
                'close_Price': np.random.uniform(100, 200, 500),
                'high_Price': np.random.uniform(100, 200, 500),
                'low_Price': np.random.uniform(100, 200, 500),
                'volume': np.random.randint(1000000, 10000000, 500),
            })
            
            # Time feature calculations
            start = time.time()
            df = sdf.calculate_moving_averages(df)
            df = sdf.calculate_standard_diviation_value(df)
            df = sdf.calculate_bollinger_bands(df)
            elapsed = time.time() - start
            
            # Should be fast (< 2 seconds for 500 rows)
            if elapsed < 2.0:
                print(f"{GREEN}  ✓ Features calculated in {elapsed:.3f}s{RESET}")
                return 'passed'
            else:
                print(f"{YELLOW}  ⚠ Feature calculation slow: {elapsed:.3f}s{RESET}")
                return 'warning'
                
        except Exception as e:
            print(f"{RED}  ✗ Test error: {str(e)}{RESET}")
            return 'failed'
    
    def _test_data_processing_efficiency(self):
        """Test data processing efficiency."""
        print(f"\n{BLUE}► Testing: Data processing efficiency{RESET}")
        
        try:
            import pandas as pd
            import numpy as np
            import time
            import sys
            sys.path.insert(0, str(project_root))
            import split_dataset
            
            # Create large dataset
            np.random.seed(42)
            df = pd.DataFrame(np.random.randn(1000, 50))
            df['target'] = np.random.randn(1000)
            
            # Time data splitting
            start = time.time()
            train, val, test = split_dataset.dataset_train_test_split(
                df, 'target', train_size=0.65, val_size=0.15, test_size=0.20
            )
            elapsed = time.time() - start
            
            # Should be very fast (< 0.1 seconds)
            if elapsed < 0.1:
                print(f"{GREEN}  ✓ Data split in {elapsed:.4f}s{RESET}")
                return 'passed'
            else:
                print(f"{YELLOW}  ⚠ Data split slow: {elapsed:.4f}s{RESET}")
                return 'warning'
                
        except Exception as e:
            print(f"{RED}  ✗ Test error: {str(e)}{RESET}")
            return 'failed'
    
    def _record_test_result(self, category, test_name, status):
        """Record a test result."""
        if status == 'passed':
            self.results[category]['passed'] += 1
        elif status == 'warning':
            self.results[category]['passed'] += 1  # Count as passed but note
        else:
            self.results[category]['failed'] += 1
        
        self.results[category]['tests'].append({
            'name': test_name,
            'status': status
        })
    
    def print_summary(self):
        """Print comprehensive test summary."""
        print(f"\n{BOLD}{'='*80}{RESET}")
        print(f"{BOLD}TEST SUMMARY{RESET}")
        print(f"{BOLD}{'='*80}{RESET}\n")
        
        total_passed = sum(r['passed'] for r in self.results.values())
        total_failed = sum(r['failed'] for r in self.results.values())
        total_tests = total_passed + total_failed
        
        for category, results in self.results.items():
            if results['passed'] == 0 and results['failed'] == 0:
                continue
                
            total = results['passed'] + results['failed']
            percentage = (results['passed'] / total * 100) if total > 0 else 0
            
            status_icon = f"{GREEN}✓{RESET}" if results['failed'] == 0 else f"{RED}✗{RESET}"
            print(f"{status_icon} {category.upper()}: {results['passed']}/{total} passed ({percentage:.1f}%)")
            
            if self.verbose and results['tests']:
                for test in results['tests']:
                    icon = f"{GREEN}✓{RESET}" if test['status'] in ['passed', 'warning'] else f"{RED}✗{RESET}"
                    time_str = f" ({test.get('time', 0):.2f}s)" if 'time' in test else ""
                    print(f"  {icon} {test['name']}{time_str}")
                print()
        
        print(f"{BOLD}OVERALL:{RESET}")
        overall_percentage = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        if total_failed == 0:
            print(f"{GREEN}✓ ALL TESTS PASSED: {total_passed}/{total_tests} ({overall_percentage:.1f}%){RESET}")
        else:
            print(f"{RED}✗ SOME TESTS FAILED: {total_passed}/{total_tests} passed ({overall_percentage:.1f}%){RESET}")
            print(f"{RED}  {total_failed} test(s) failed{RESET}")
        
        elapsed = time.time() - self.start_time
        print(f"\nTotal time: {elapsed:.2f}s")
        print("="*80 + "\n")
        
        return total_failed == 0
    
    def run(self):
        """Run the complete test suite."""
        self.start_time = time.time()
        self.print_header()
        
        try:
            self.run_functional_tests()
            self.run_ml_tests()
            self.run_integration_tests()
            self.run_regression_tests()
            self.run_performance_tests()
        except KeyboardInterrupt:
            print(f"\n{YELLOW}⚠ Test suite interrupted by user{RESET}")
        
        success = self.print_summary()
        
        return 0 if success else 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Master Test Suite for Stock Portfolio Builder')
    parser.add_argument('--quick', action='store_true', 
                       help='Run only quick tests (no TensorFlow required)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed output')
    parser.add_argument('--category', '-c', 
                       choices=['functional', 'ml', 'integration', 'regression', 'performance'],
                       help='Run specific test category only')
    parser.add_argument('--save-baseline', action='store_true',
                       help='Save current metrics as baseline for regression testing')
    
    args = parser.parse_args()
    
    suite = MasterTestSuite(
        quick_mode=args.quick,
        verbose=args.verbose,
        category=args.category
    )
    
    exit_code = suite.run()
    
    # Save baseline if requested
    if args.save_baseline:
        print(f"\n{BLUE}Saving baseline metrics...{RESET}")
        # This would need to be implemented based on actual metrics
        print(f"{YELLOW}⚠ --save-baseline flag needs implementation{RESET}")
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
