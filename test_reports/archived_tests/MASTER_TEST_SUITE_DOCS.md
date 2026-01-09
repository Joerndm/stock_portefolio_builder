# Master Test Suite Documentation

## Overview

The Master Test Suite is a comprehensive, reusable testing framework designed to validate all functionality in the Stock Portfolio Builder after any change or improvement. It ensures that modifications don't negatively impact stock price prediction accuracy or introduce regressions.

## Quick Start

### Run Quick Tests (< 1 minute, no ML dependencies)
```bash
python test_reports/quick_test_runner.py
```

### Run Full Test Suite
```bash
python test_reports/master_test_suite.py
```

### Run Specific Category
```bash
python test_reports/master_test_suite.py --category functional
python test_reports/master_test_suite.py --category ml
python test_reports/master_test_suite.py --category integration
```

### Quick Mode (Skip ML Tests)
```bash
python test_reports/master_test_suite.py --quick
```

### Verbose Output
```bash
python test_reports/master_test_suite.py --verbose
```

## Test Categories

### 1. Functional Tests ⚡ (Fast, no ML dependencies)
Tests core functions without machine learning dependencies.

**What's Tested:**
- ✅ LSTM sequence creation and dataset preparation
- ✅ Technical indicator calculations (SMA, EMA, Bollinger Bands, etc.)
- ✅ Overfitting remediation logic (early stopping, search space modification)
- ✅ Data splitting and preprocessing
- ✅ Hyperparameter comparison logic

**Run Time:** ~10-30 seconds

**Files:**
- `unit_test.py` - LSTM sequence creation
- `test_feature_calculations.py` - Technical indicators
- `test_overfitting_remediation_standalone.py` - Overfitting logic

### 2. Machine Learning Tests 🤖 (Requires TensorFlow)
Tests ML models, training, and evaluation.

**What's Tested:**
- ✅ MAE metric implementation for all models (LSTM, RF, XGBoost)
- ✅ LSTM hyperparameter configuration
- ✅ Multi-metric overfitting detection (MSE, R², MAE, consistency)
- ✅ Retraining efficiency and convergence
- ✅ XGBoost ensemble integration
- ✅ Early stopping mechanisms
- ✅ Search space constraints

**Run Time:** ~2-5 minutes

**Files:**
- `test_mae_fix.py` - MAE implementation
- `test_lstm_parameters.py` - LSTM configuration
- `test_overfitting_detection.py` - Overfitting detection
- `test_retraining_efficiency.py` - Retraining logic
- `test_xgboost_ensemble_fix.py` - Ensemble prediction

### 3. Integration Tests 🔗 (Requires full environment)
Tests end-to-end pipeline execution.

**What's Tested:**
- ✅ Full ML pipeline execution flow (database → preprocessing → training → prediction)
- ✅ Data fetching and feature engineering pipeline
- ✅ Ensemble prediction methodology (LSTM + RF + XGBoost)
- ✅ Monte Carlo simulation integration
- ✅ Database interactions
- ✅ Graph generation and output

**Run Time:** ~3-10 minutes

**Files:**
- `test_ml_builder_flow.py` - Complete pipeline
- `test_stock_data_fetch.py` - Data fetching
- `test_quick_validation.py` - End-to-end validation

### 4. Regression Tests 📊 (Compares to baseline)
Ensures changes don't degrade performance.

**What's Tested:**
- ✅ Prediction accuracy vs baseline (R², MSE, MAE)
- ✅ Training time hasn't increased significantly
- ✅ Model performance maintained across iterations
- ✅ No accuracy degradation after changes

**Baseline Metrics Tracked:**
- Ensemble R² (target: maintain within 5%)
- Ensemble MSE (target: maintain within 5%)
- Ensemble MAE (target: maintain within 5%)
- Training time (target: maintain within 20%)

**Run Time:** ~2-4 minutes

**Setup:**
1. First run generates baseline: `python test_reports/master_test_suite.py --save-baseline`
2. Subsequent runs compare against baseline

### 5. Performance Tests ⚡ (Speed and efficiency)
Tests computational efficiency.

**What's Tested:**
- ✅ Feature calculation speed (target: < 2s for 500 rows)
- ✅ Data processing efficiency (target: < 0.1s for 1000 rows)
- ✅ Model training time benchmarks
- ✅ Memory usage (planned)

**Run Time:** ~30-60 seconds

## Industry Standard Metrics

The test suite validates all industry-standard metrics are present and calculated correctly:

### Prediction Accuracy Metrics
- **MSE (Mean Squared Error)** - Measures average squared prediction error
- **R² (R-squared)** - Coefficient of determination (0-1, higher is better)
- **MAE (Mean Absolute Error)** - Average absolute prediction error
- **RMSE (Root Mean Squared Error)** - Square root of MSE, in original units

### Model Evaluation Metrics
- **Training metrics** - Performance on training data
- **Validation metrics** - Performance on validation data (overfitting check)
- **Test metrics** - Final performance on unseen data

### Ensemble Metrics
- **Weighted average** of LSTM (40%), Random Forest (30%), XGBoost (30%)
- **All three metrics** (MSE, R², MAE) calculated for ensemble

### Overfitting Detection
- **Multi-metric score** combining MSE (35%), R² (25%), MAE (30%), Consistency (10%)
- **Threshold-based detection** (score > 0.15 indicates overfitting)
- **Automatic remediation** with early stopping and search space modification

## Test Suite Architecture

```
test_reports/
├── master_test_suite.py           # Main orchestrator
├── quick_test_runner.py            # Fast validation (< 1 min)
├── test_config.yaml                # Test configuration
├── baseline_metrics_template.json  # Baseline metrics structure
│
├── Functional Tests/
│   ├── unit_test.py
│   ├── test_feature_calculations.py
│   └── test_overfitting_remediation_standalone.py
│
├── ML Tests/
│   ├── test_mae_fix.py
│   ├── test_lstm_parameters.py
│   ├── test_overfitting_detection.py
│   ├── test_retraining_efficiency.py
│   └── test_xgboost_ensemble_fix.py
│
├── Integration Tests/
│   ├── test_ml_builder_flow.py
│   ├── test_stock_data_fetch.py
│   └── test_quick_validation.py
│
└── Documentation/
    └── MASTER_TEST_SUITE_DOCS.md (this file)
```

## Workflow for Changes

### Before Making Changes
1. Run quick tests to establish working state:
   ```bash
   python test_reports/quick_test_runner.py
   ```

2. Run full suite to establish baseline (first time only):
   ```bash
   python test_reports/master_test_suite.py --save-baseline
   ```

### After Making Changes
1. Run quick tests for immediate feedback:
   ```bash
   python test_reports/quick_test_runner.py
   ```

2. If quick tests pass, run relevant category:
   ```bash
   # If you changed ML logic
   python test_reports/master_test_suite.py --category ml
   
   # If you changed data processing
   python test_reports/master_test_suite.py --category functional
   
   # If you changed the pipeline
   python test_reports/master_test_suite.py --category integration
   ```

3. Run full suite before committing:
   ```bash
   python test_reports/master_test_suite.py --verbose
   ```

4. Check regression tests to ensure no performance degradation:
   ```bash
   python test_reports/master_test_suite.py --category regression
   ```

### If Tests Fail
1. **Read the error output** - The verbose mode (`--verbose`) shows detailed information
2. **Check which category failed** - Narrows down the problem area
3. **Run individual test file** - For detailed debugging:
   ```bash
   python test_reports/test_overfitting_detection.py
   ```
4. **Fix the issue** and re-run tests
5. **Update baseline if intentional change** - If the change was meant to alter performance:
   ```bash
   python test_reports/master_test_suite.py --save-baseline
   ```

## Adding New Tests

### 1. Add Test to Existing Category
Edit the appropriate test file in `test_reports/` and add your test case.

### 2. Create New Test File
```python
"""
Test [Feature Name]
==================
Description of what this tests.
"""

import unittest

class TestNewFeature(unittest.TestCase):
    
    def test_something(self):
        """Test that something works."""
        # Your test code
        assert True
        
if __name__ == '__main__':
    unittest.main()
```

### 3. Register in Master Suite
Edit `master_test_suite.py` and add your test file to the appropriate category:

```python
def run_functional_tests(self):
    tests = [
        # ... existing tests ...
        ('test_new_feature.py', 'Description of new feature test'),
    ]
```

## Baseline Metrics

### Creating Baseline
Run the full suite with `--save-baseline` flag:
```bash
python test_reports/master_test_suite.py --save-baseline
```

This creates `baseline_metrics.json` with:
- Ensemble prediction metrics (R², MSE, MAE)
- Individual model metrics
- Training time benchmarks
- Performance benchmarks

### Updating Baseline
Update baseline after intentional changes that improve or modify performance:
```bash
# Make your improvements
# Run tests to verify improvement
python test_reports/master_test_suite.py --verbose

# If tests show improvement, update baseline
python test_reports/master_test_suite.py --save-baseline
```

### Baseline Structure
```json
{
  "ensemble_metrics": {
    "r2": 0.75,
    "mse": 0.15,
    "mae": 0.30
  },
  "performance_metrics": {
    "rf_training_time": 30.0,
    "feature_calculation_time": 1.5
  }
}
```

## Continuous Integration (CI/CD)

### Quick CI Check (< 1 minute)
```bash
# In CI pipeline
python test_reports/quick_test_runner.py
```

### Full CI Check (5-10 minutes)
```bash
# In CI pipeline
python test_reports/master_test_suite.py --quick
```

### Nightly Full Suite
```bash
# Nightly build
python test_reports/master_test_suite.py --verbose
```

## Interpreting Results

### All Tests Pass ✅
```
✓ FUNCTIONAL: 3/3 passed (100.0%)
✓ ML: 5/5 passed (100.0%)
✓ INTEGRATION: 3/3 passed (100.0%)
✓ REGRESSION: 2/2 passed (100.0%)
✓ PERFORMANCE: 2/2 passed (100.0%)

OVERALL:
✓ ALL TESTS PASSED: 15/15 (100.0%)
```
**Action:** Changes are safe to commit.

### Some Tests Fail ❌
```
✓ FUNCTIONAL: 3/3 passed (100.0%)
✗ ML: 4/5 passed (80.0%)
✓ INTEGRATION: 3/3 passed (100.0%)
✗ REGRESSION: 1/2 passed (50.0%)
✓ PERFORMANCE: 2/2 passed (100.0%)

OVERALL:
✗ SOME TESTS FAILED: 13/15 passed (86.7%)
  2 test(s) failed
```
**Action:** 
1. Check which specific tests failed (use `--verbose`)
2. Fix the issues
3. Re-run tests
4. If regression test failed due to intentional change, update baseline

### Performance Warning ⚠️
```
⚠ Feature calculation slow: 2.5s
```
**Action:** Check if performance degradation is acceptable or needs optimization.

## Best Practices

### 1. Run Tests Frequently
- After every significant change
- Before committing code
- After merging branches

### 2. Use Quick Tests During Development
```bash
# Fast iteration during development
python test_reports/quick_test_runner.py
```

### 3. Run Full Suite Before Release
```bash
# Complete validation before release
python test_reports/master_test_suite.py --verbose
```

### 4. Keep Baseline Updated
- Update baseline after verified improvements
- Document why baseline changed
- Keep baseline in version control

### 5. Fix Failing Tests Immediately
- Don't commit with failing tests
- Investigate failures thoroughly
- Update tests if behavior intentionally changed

## Troubleshooting

### "ModuleNotFoundError: No module named 'tensorflow'"
**Solution:** Run in quick mode or install TensorFlow:
```bash
python test_reports/master_test_suite.py --quick
# OR
conda install tensorflow
```

### "No baseline metrics found"
**Solution:** Create baseline:
```bash
python test_reports/master_test_suite.py --save-baseline
```

### Tests are very slow
**Solution:** Use quick mode or specific category:
```bash
python test_reports/master_test_suite.py --quick
python test_reports/master_test_suite.py --category functional
```

### Regression tests always fail
**Solution:** Check if baseline is outdated:
```bash
# View current baseline
cat test_reports/baseline_metrics.json

# Update if needed
python test_reports/master_test_suite.py --save-baseline
```

## Summary

The Master Test Suite provides:
- ✅ **Comprehensive coverage** - All critical functionality tested
- ✅ **Fast feedback** - Quick tests run in < 1 minute
- ✅ **Regression prevention** - Baseline comparison ensures no degradation
- ✅ **Industry standards** - All standard metrics validated
- ✅ **Easy integration** - Simple command-line interface
- ✅ **Flexible execution** - Run all tests or specific categories

**Use it every time you make changes to ensure code quality and prevent regressions!**
