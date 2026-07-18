# Data Type Consistency Test

**Created:** December 20, 2025  
**Purpose:** Prevent sklearn feature name warnings throughout the ML pipeline

## Overview

This test validates that data types remain consistent throughout the pipeline to prevent the following sklearn warnings:

```
UserWarning: X has feature names, but RandomForestRegressor was fitted without feature names
UserWarning: X does not have valid feature names, but MinMaxScaler was fitted with feature names
```

## Test Coverage

### 1. MinMaxScaler with DataFrames ✅
- **Validates:** Scaler fits and transforms DataFrames without warnings
- **Ensures:** Feature names are preserved through scaling

### 2. MinMaxScaler DataFrame → Numpy Warning ✅
- **Validates:** Warning is correctly raised when mixing data types
- **Purpose:** Confirms sklearn warning system is working

### 3. RandomForest with Numpy Arrays ✅
- **Validates:** RF fits and predicts with numpy arrays without warnings
- **Ensures:** Consistent numpy usage for tree-based models

### 4. RandomForest Numpy → DataFrame Warning ✅
- **Validates:** Warning is correctly raised when mixing data types
- **Purpose:** Confirms sklearn warning system is working

### 5. Full Pipeline Consistency ✅
- **Validates:** Complete pipeline maintains type consistency:
  1. Create DataFrame with features
  2. Fit scaler with DataFrame
  3. Transform with DataFrame (no warnings)
  4. Convert to numpy for RF/XGB
  5. Fit RF with numpy (no warnings)
  6. Predict with numpy (no warnings)
  7. Test prediction pipeline (no warnings)

### 6. Dimension Reduction Consistency ✅
- **Validates:** `feature_selection_rf()` maintains type consistency
- **Ensures:** No warnings when selecting features with RandomForest

## Pipeline Rules

### ✅ CORRECT Usage

**Scalers (MinMaxScaler):**
```python
# Fit with DataFrame
scaler = MinMaxScaler()
scaler.set_output(transform="pandas")
scaler.fit(dataframe)

# Transform with DataFrame
scaled_df = scaler.transform(dataframe)

# Convert to numpy AFTER scaling
X = scaled_df.values
```

**RF/XGB Models:**
```python
# Fit with numpy arrays
model.fit(X_train.values, y_train)

# Predict with numpy arrays
predictions = model.predict(X_test.values)
```

### ❌ INCORRECT Usage

**Don't do this:**
```python
# BAD: Fit scaler with DataFrame, transform with numpy
scaler.fit(dataframe)
scaled = scaler.transform(dataframe.values)  # ⚠️ Warning!

# BAD: Fit RF with numpy, predict with DataFrame
model.fit(X.values, y)
predictions = model.predict(X_df)  # ⚠️ Warning!
```

## Files Modified to Fix Warnings

### ml_builder.py
**Lines 1995-1997:** Convert to numpy for RF validation predictions
```python
# Convert to numpy to avoid feature name warnings
rf_train_pred_full = rf_model.predict(x_train_df.values)
rf_val_pred_full = rf_model.predict(x_val_df.values)
rf_test_pred_full = rf_model.predict(x_test_df.values)
```

**Lines 2236, 2238:** Convert to numpy for RF/XGB historical predictions
```python
# Random Forest prediction (already unscaled)
# Convert DataFrame to numpy to avoid feature name warning
forecast_rf = rf_model.predict(scaled_x_input_rf_df.values)[0]

# XGBoost prediction (already unscaled)
if xgb_model is not None:
    forecast_xgb = xgb_model.predict(scaled_x_input_rf_df.values)[0]
```

**Lines 2669, 2671:** Convert to numpy for RF/XGB future predictions
```python
# Random Forest prediction (already unscaled)
# Convert DataFrame to numpy to avoid feature name warning
forecast_rf = rf_model.predict(x_input_rf_df.values)[0]

# XGBoost prediction (already unscaled)
if xgb_model is not None:
    forecast_xgb = xgb_model.predict(x_input_rf_df.values)[0]
```

**Lines 2639-2644:** Keep DataFrame for scaling, convert to numpy after
```python
# Convert to NumPy array and scale
# Keep as DataFrame for scaling to avoid feature name warning
scaled_x_lstm_df = scaler_x.transform(x_lstm_df)
# Convert to numpy array after scaling
scaled_x_lstm_array = scaled_x_lstm_df.values if hasattr(scaled_x_lstm_df, 'values') else np.array(scaled_x_lstm_df)
scaled_x_input_lstm = scaled_x_lstm_array.reshape(1, time_steps, scaled_x_lstm_array.shape[1])
```

## Running the Test

```bash
# Run standalone
python test_reports/test_data_type_consistency.py

# Run as part of quick test suite
python test_reports/quick_test_runner.py
```

## Expected Output

```
================================================================================
DATA TYPE CONSISTENCY TEST SUITE
================================================================================
Validates data type flow to prevent sklearn warnings
================================================================================

TEST 1: MinMaxScaler with DataFrames
   ✓ PASS: Scaler works correctly with DataFrames

TEST 2: MinMaxScaler DataFrame → Numpy (should warn)
   ✓ PASS: Warning correctly raised when mixing DataFrame fit + numpy transform

TEST 3: RandomForest with Numpy Arrays
   ✓ PASS: RandomForest works correctly with numpy arrays

TEST 4: RandomForest Numpy → DataFrame (should warn)
   ✓ PASS: Warning correctly raised when mixing numpy fit + DataFrame predict

TEST 5: Full Pipeline Data Type Consistency
   ✓ PASS: Full pipeline maintains data type consistency

TEST 6: Dimension Reduction Type Consistency
   ✓ PASS: feature_selection_rf maintains type consistency

================================================================================
TEST SUMMARY
================================================================================
Total: 6 tests, 6 passed, 0 failed
Time: 0.49s

✅ ALL TESTS PASSED!
Data type consistency validated throughout pipeline
```

## Benefits

1. **No more warnings** - Clean output during training and prediction
2. **Consistent behavior** - Predictable data type flow throughout pipeline
3. **Future-proof** - Catches type mismatches early before they cause warnings
4. **Documentation** - Clear examples of correct usage patterns

## Integration with CI/CD

This test is included in `quick_test_runner.py` as **Test #7** and runs in < 1 second. It's perfect for:
- Pre-commit validation
- CI/CD pipeline checks
- Post-refactoring verification
- Onboarding documentation

## Maintenance

If you add new sklearn models or scalers:
1. Add test case to `test_data_type_consistency.py`
2. Follow the pattern: fit and predict/transform with same data type
3. Run test to verify no warnings

## Related Issues

- Fixed sklearn warnings reported on December 20, 2025
- Prevents regression of feature name warnings
- Validates fixes in ml_builder.py lines 1995-1997, 2236, 2238, 2639-2644, 2669, 2671
