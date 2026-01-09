"""
Diagnostic Test for DataFrame dtype Issue in XGBoost Prediction

Error: DataFrame columns are 'object' type instead of numeric types.
This test reproduces the issue, identifies the root cause, and validates the fix.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("="*80)
print("DIAGNOSTIC TEST - DataFrame dtype Issue")
print("="*80)

# Test 1: Reproduce the issue
print("\n📋 TEST 1: Reproduce the dtype issue")
print("-"*80)

# Simulate what might be happening in the code
print("\nScenario A: Creating DataFrame from dict with mixed types")
data_mixed = {
    'close_Price': ['100.5', '101.2', '99.8'],  # Strings instead of floats
    'volume': [1000000, 1100000, 950000],
}
df_mixed = pd.DataFrame(data_mixed)
print(f"DataFrame dtypes:\n{df_mixed.dtypes}")
print(f"close_Price is object: {df_mixed['close_Price'].dtype == 'object'}")

print("\nScenario B: Reading from database (simulated)")
# When data comes from database, sometimes it's read as object
df_from_db = pd.DataFrame({
    'close_Price': pd.Series(['100.5', '101.2', '99.8'], dtype='object'),
    'volume': pd.Series([1000000, 1100000, 950000], dtype='int64')
})
print(f"DataFrame dtypes:\n{df_from_db.dtypes}")

print("\nScenario C: After feature selection with object dtype")
# This might happen if feature_selection returns object dtype
selected_features = ['close_Price', 'sma_5', 'sma_20']
df_features = pd.DataFrame(
    np.array([['100.5', '99.8', '101.0'],
              ['101.2', '100.5', '101.5'],
              ['99.8', '101.0', '100.8']]),
    columns=selected_features
)
print(f"DataFrame dtypes:\n{df_features.dtypes}")
print(f"All columns are object: {all(df_features.dtypes == 'object')}")

# Test 2: Check where the issue originates
print("\n\n📋 TEST 2: Identify root cause in ml_builder.py")
print("-"*80)

try:
    # Read the relevant section of ml_builder.py
    with open(project_root / 'ml_builder.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for feature selection that might cause dtype issues
    issues_found = []
    
    # Check 1: Feature selection using .values without dtype preservation
    if '.values' in content and 'feature_selection' in content:
        issues_found.append("⚠️  Uses .values which can lose dtype information")
    
    # Check 2: DataFrame creation without explicit dtype
    if 'pd.DataFrame(' in content and 'x_input_rf' in content:
        issues_found.append("⚠️  DataFrame creation might not preserve dtypes")
    
    # Check 3: Indexing that returns object dtype
    if '[selected_features]' in content:
        issues_found.append("⚠️  Feature indexing might return object dtype")
    
    if issues_found:
        print("\nPotential issues found in ml_builder.py:")
        for issue in issues_found:
            print(f"  {issue}")
    else:
        print("\n✓ No obvious dtype issues in code patterns")
        
except Exception as e:
    print(f"❌ Error reading ml_builder.py: {e}")

# Test 3: Solutions
print("\n\n📋 TEST 3: Test dtype conversion solutions")
print("-"*80)

print("\nSolution A: Convert to numeric when creating DataFrame")
df_solution_a = pd.DataFrame(data_mixed)
df_solution_a = df_solution_a.apply(pd.to_numeric, errors='coerce')
print(f"After pd.to_numeric:\n{df_solution_a.dtypes}")
print(f"✓ All numeric: {all(df_solution_a.dtypes != 'object')}")

print("\nSolution B: Explicit dtype during DataFrame creation")
df_solution_b = pd.DataFrame(
    np.array([['100.5', '99.8', '101.0'],
              ['101.2', '100.5', '101.5'],
              ['99.8', '101.0', '100.8']]).astype(float),
    columns=['close_Price', 'sma_5', 'sma_20']
)
print(f"With .astype(float):\n{df_solution_b.dtypes}")
print(f"✓ All numeric: {all(df_solution_b.dtypes != 'object')}")

print("\nSolution C: Use select_dtypes and convert")
df_solution_c = pd.DataFrame(data_mixed)
# Convert only object columns to numeric
object_cols = df_solution_c.select_dtypes(include=['object']).columns
for col in object_cols:
    df_solution_c[col] = pd.to_numeric(df_solution_c[col], errors='coerce')
print(f"Converting object columns:\n{df_solution_c.dtypes}")
print(f"✓ All numeric: {all(df_solution_c.dtypes != 'object')}")

# Test 4: Validate XGBoost compatibility
print("\n\n📋 TEST 4: Validate XGBoost compatibility")
print("-"*80)

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    
    # Create test data
    X = pd.DataFrame(np.random.randn(100, 5), columns=['f1', 'f2', 'f3', 'f4', 'f5'])
    y = np.random.randn(100)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train a simple model
    model = xgb.XGBRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Test with correct dtypes
    print("\nTest A: Correct dtypes (float64)")
    X_test_correct = X_test.copy()
    print(f"  Dtypes: {X_test_correct.dtypes.unique()}")
    try:
        pred = model.predict(X_test_correct)
        print(f"  ✓ Prediction successful: {len(pred)} predictions")
    except Exception as e:
        print(f"  ❌ Prediction failed: {e}")
    
    # Test with object dtypes (should fail)
    print("\nTest B: Object dtypes (should fail)")
    X_test_object = X_test.copy().astype('object')
    print(f"  Dtypes: {X_test_object.dtypes.unique()}")
    try:
        pred = model.predict(X_test_object)
        print(f"  ⚠️  Prediction succeeded (unexpected): {len(pred)} predictions")
    except ValueError as e:
        print(f"  ✓ Expected failure: DataFrame.dtypes must be int, float, bool or category")
    
    # Test with converted dtypes (should work)
    print("\nTest C: Converted from object to float")
    X_test_converted = X_test_object.apply(pd.to_numeric, errors='coerce')
    print(f"  Dtypes: {X_test_converted.dtypes.unique()}")
    try:
        pred = model.predict(X_test_converted)
        print(f"  ✓ Prediction successful after conversion: {len(pred)} predictions")
    except Exception as e:
        print(f"  ❌ Prediction failed: {e}")
    
except ImportError:
    print("⚠️  XGBoost not available, skipping validation")
except Exception as e:
    print(f"❌ Test error: {e}")

# Test 5: Find exact location in ml_builder.py
print("\n\n📋 TEST 5: Locate problematic code in ml_builder.py")
print("-"*80)

try:
    with open(project_root / 'ml_builder.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find the problematic line (around line 2466)
    target_line = 2466 - 1  # 0-indexed
    context_start = max(0, target_line - 10)
    context_end = min(len(lines), target_line + 10)
    
    print(f"\nCode around line 2466 (where error occurs):")
    print("-"*80)
    for i in range(context_start, context_end):
        marker = ">>> " if i == target_line else "    "
        print(f"{marker}{i+1:4d}: {lines[i].rstrip()}")
    
    # Look for x_input_rf_df creation
    print("\n\nSearching for x_input_rf_df creation...")
    for i, line in enumerate(lines):
        if 'x_input_rf_df' in line and 'pd.DataFrame' in line:
            print(f"Found at line {i+1}:")
            start = max(0, i-3)
            end = min(len(lines), i+4)
            for j in range(start, end):
                marker = ">>> " if j == i else "    "
                print(f"{marker}{j+1:4d}: {lines[j].rstrip()}")
            break
    
except Exception as e:
    print(f"❌ Error: {e}")

# Summary
print("\n\n" + "="*80)
print("DIAGNOSTIC SUMMARY")
print("="*80)

print("\n🔍 Root Cause:")
print("  - DataFrame columns are created with 'object' dtype instead of numeric")
print("  - XGBoost requires int/float/bool/category dtypes")
print("  - Likely caused by DataFrame creation from arrays without dtype specification")

print("\n💡 Recommended Solution:")
print("  1. Add explicit dtype conversion when creating x_input_rf_df")
print("  2. Use pd.to_numeric() to convert object columns to numeric")
print("  3. Or use .astype(float) when creating DataFrame from numpy array")

print("\n📝 Implementation:")
print("  Add after x_input_rf_df creation:")
print("  ```python")
print("  # Convert all columns to numeric (fix dtype issue)")
print("  x_input_rf_df = x_input_rf_df.apply(pd.to_numeric, errors='coerce')")
print("  ```")

print("\n✅ Validation:")
print("  - Test with quick_test_runner.py after fix")
print("  - Run integration tests to ensure no regressions")

print("\n" + "="*80)
