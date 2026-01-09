# Archived Tests

This folder contains legacy test files that were created between December 2025 and January 2026. All functionality from these tests has been migrated to the new comprehensive test suite.

## Why These Tests Were Archived

In January 2026, a comprehensive test suite was implemented with:
- Industry-standard organization (unit/integration/e2e/performance/security/validation)
- 305+ tests with proper mocking and isolation
- Unified test runner with category filtering
- Better code reuse and maintainability
- Enhanced documentation

All functionality from these archived tests is now covered by the new suite with improved structure and coverage.

## Archived Files (27 total)

### Test Runners (2 files)
- `master_test_suite.py` - Old master runner → **Replaced by:** comprehensive_test_runner.py
- `quick_test_runner.py` - Old quick runner → **Replaced by:** comprehensive_test_runner.py --category unit

### Overfitting Tests (5 files)
- `test_overfitting_detection.py` → **Replaced by:** unit/test_ml_builder_units.py (TestDetectOverfitting)
- `test_overfitting_standalone.py` → **Replaced by:** unit/test_ml_builder_units.py
- `test_overfitting_remediation_standalone.py` → **Replaced by:** unit/test_ml_builder_units.py
- `test_overfitting_remediation.py` → **Replaced by:** integration/test_pipelines_integration.py
- `test_overfitting_improvements.py` → **Replaced by:** unit/test_ml_builder_units.py

### LSTM Tests (4 files)
- `test_lstm_parameters.py` → **Replaced by:** unit/test_ml_builder_units.py (TestBuildLSTMModel)
- `test_lstm_retraining_analysis.py` → **Replaced by:** performance/test_performance_benchmarks.py
- `test_lstm_input_robustness.py` → **Replaced by:** unit/test_ml_builder_units.py
- `test_lstm_scaling_fix.py` → **Replaced by:** data_validation/test_data_quality.py

### Model Tests (4 files)
- `test_mae_fix.py` → **Replaced by:** unit/test_ml_builder_units.py (all model tests)
- `test_xgboost_ensemble_fix.py` → **Replaced by:** unit/test_ml_builder_units.py
- `test_model_instantiation.py` → **Replaced by:** unit/test_ml_builder_units.py
- `test_model_comparison.py` → **Replaced by:** integration/test_pipelines_integration.py

### Feature & Data Tests (3 files)
- `test_feature_calculations.py` → **Replaced by:** unit/test_stock_data_fetch_units.py
- `test_feature_coverage.py` → **Replaced by:** data_validation/test_data_quality.py
- `test_stock_data_fetch.py` → **Replaced by:** unit/test_stock_data_fetch_units.py

### Integration Tests (4 files)
- `test_ml_builder_flow.py` → **Replaced by:** integration/test_pipelines_integration.py
- `test_quick_validation.py` → **Replaced by:** e2e/test_complete_workflows.py
- `test_single_ticker.py` → **Replaced by:** e2e/test_complete_workflows.py
- `unit_test.py` → **Replaced by:** unit/test_ml_builder_units.py (TestCreateSequences)

### Data Type & Export Tests (4 files)
- `test_data_type_consistency.py` → **Replaced by:** data_validation/test_data_quality.py
- `test_dtype_fix_validation.py` → **Replaced by:** data_validation/test_data_quality.py
- `test_dtype_issue.py` → **Replaced by:** data_validation/test_data_quality.py
- `test_column_mapping.py` → **Replaced by:** unit/test_db_interactions_units.py

### Database & Export Tests (4 files)
- `test_db_components.py` → **Replaced by:** unit/test_db_interactions_units.py
- `test_export_fix.py` → **Replaced by:** unit/test_db_interactions_units.py
- `test_actual_export.py` → **Replaced by:** unit/test_db_interactions_units.py
- `test_dropna_fix.py` → **Replaced by:** data_validation/test_data_quality.py

### Optimization Tests (2 files)
- `test_retraining_efficiency.py` → **Replaced by:** performance/test_performance_benchmarks.py
- `test_optimal_trial_configuration.py` → **Replaced by:** performance/test_performance_benchmarks.py

### Improvement & Validation Tests (2 files)
- `test_improvements.py` → **Replaced by:** unit tests across all modules
- `test_improvement_comparison.py` → **Replaced by:** integration/test_pipelines_integration.py

### Miscellaneous Tests (2 files)
- `test_real_stock_data.py` → **Replaced by:** e2e/test_complete_workflows.py
- `demo_model_comparison.py` → **Replaced by:** integration/test_pipelines_integration.py

### Documentation (2 files)
- `MASTER_TEST_SUITE_DOCS.md` → **Replaced by:** COMPREHENSIVE_TEST_SUITE_README.md
- `TEST_SUITE_IMPLEMENTATION_COMPLETE.md` → **Replaced by:** COMPREHENSIVE_TEST_SUITE_README.md

## Using the New Test Suite

Instead of running these archived tests, use the comprehensive test suite:

```bash
# Run all tests
python test_reports/comprehensive_test_runner.py

# Run specific category
python test_reports/comprehensive_test_runner.py --category unit
python test_reports/comprehensive_test_runner.py --category integration
python test_reports/comprehensive_test_runner.py --category e2e

# Generate report
python test_reports/comprehensive_test_runner.py --report
```

See [COMPREHENSIVE_TEST_SUITE_README.md](../COMPREHENSIVE_TEST_SUITE_README.md) for complete documentation.

## When to Reference Archived Tests

These archived tests may be useful for:
- Understanding historical implementation decisions
- Reviewing specific test cases that were migrated
- Debugging legacy issues
- Learning how tests evolved over time

**Note:** Do not use these archived tests in CI/CD or for validation. Always use the new comprehensive test suite.

---

**Archived Date:** January 9, 2026  
**Total Archived Files:** 27 test files + 2 documentation files  
**Replacement Suite:** Comprehensive Test Suite (305+ tests)
