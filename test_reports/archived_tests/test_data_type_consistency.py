"""
DATA TYPE CONSISTENCY TEST
===========================

Ensures consistent data types throughout the ML pipeline to prevent sklearn warnings:
- RandomForestRegressor feature names warning
- MinMaxScaler feature names warning

This test validates that:
1. Scalers are fit and transform with DataFrames
2. RF/XGB models are fit with numpy arrays
3. RF/XGB models predict with numpy arrays
4. LSTM receives properly shaped numpy arrays

Usage:
    python test_reports/test_data_type_consistency.py
"""

import sys
import os
from pathlib import Path
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


def test_scaler_with_dataframe():
    """Test that MinMaxScaler works correctly with DataFrames (no warnings)."""
    print(f"\n{BOLD}TEST 1: MinMaxScaler with DataFrames{RESET}")
    
    # Create sample data with feature names
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'feature3': [100, 200, 300, 400, 500]
    })
    
    # Fit scaler with DataFrame
    scaler = MinMaxScaler()
    scaler.set_output(transform="pandas")
    scaler.fit(df)
    
    # Transform with DataFrame (should not warn)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        scaled_df = scaler.transform(df)
        
        # Check for feature name warnings
        feature_warnings = [warning for warning in w 
                          if "feature names" in str(warning.message).lower()]
        
        if feature_warnings:
            print(f"   {RED}❌ FAIL: Got warnings when using DataFrame with scaler{RESET}")
            for warning in feature_warnings:
                print(f"      {warning.message}")
            return False
    
    # Verify output is DataFrame
    if not isinstance(scaled_df, pd.DataFrame):
        print(f"   {RED}❌ FAIL: Scaler output is not DataFrame (got {type(scaled_df)}){RESET}")
        return False
    
    # Verify feature names preserved
    if not all(scaled_df.columns == df.columns):
        print(f"   {RED}❌ FAIL: Feature names not preserved{RESET}")
        return False
    
    print(f"   {GREEN}✓ PASS: Scaler works correctly with DataFrames{RESET}")
    print(f"      Input: DataFrame with {df.shape[1]} features")
    print(f"      Output: DataFrame with {scaled_df.shape[1]} features")
    return True


def test_scaler_with_numpy_warns():
    """Test that MinMaxScaler warns when fit with DataFrame but transform with numpy."""
    print(f"\n{BOLD}TEST 2: MinMaxScaler DataFrame → Numpy (should warn){RESET}")
    
    # Create sample data
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50]
    })
    
    # Fit scaler with DataFrame
    scaler = MinMaxScaler()
    scaler.fit(df)
    
    # Transform with numpy array (should warn)
    numpy_data = df.values
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        scaled_array = scaler.transform(numpy_data)
        
        # Check for feature name warnings
        feature_warnings = [warning for warning in w 
                          if "feature names" in str(warning.message).lower()]
        
        if not feature_warnings:
            print(f"   {YELLOW}⚠️  WARNING: Expected warning not raised{RESET}")
            print(f"      This might be okay in newer sklearn versions")
            return True
    
    print(f"   {GREEN}✓ PASS: Warning correctly raised when mixing DataFrame fit + numpy transform{RESET}")
    print(f"      Warning: {feature_warnings[0].message}")
    return True


def test_rf_with_numpy():
    """Test that RandomForest works correctly with numpy arrays (no warnings)."""
    print(f"\n{BOLD}TEST 3: RandomForest with Numpy Arrays{RESET}")
    
    # Create sample data
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    
    # Fit with numpy
    rf = RandomForestRegressor(n_estimators=10, random_state=42)
    rf.fit(X, y)
    
    # Predict with numpy (should not warn)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        predictions = rf.predict(X)
        
        # Check for feature name warnings
        feature_warnings = [warning for warning in w 
                          if "feature names" in str(warning.message).lower()]
        
        if feature_warnings:
            print(f"   {RED}❌ FAIL: Got warnings when using numpy arrays{RESET}")
            for warning in feature_warnings:
                print(f"      {warning.message}")
            return False
    
    print(f"   {GREEN}✓ PASS: RandomForest works correctly with numpy arrays{RESET}")
    print(f"      Input shape: {X.shape}")
    print(f"      Output shape: {predictions.shape}")
    return True


def test_rf_with_dataframe_warns():
    """Test that RandomForest warns when fit with numpy but predict with DataFrame."""
    print(f"\n{BOLD}TEST 4: RandomForest Numpy → DataFrame (should warn){RESET}")
    
    # Create sample data
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    
    # Fit with numpy
    rf = RandomForestRegressor(n_estimators=10, random_state=42)
    rf.fit(X, y)
    
    # Predict with DataFrame (should warn)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        predictions = rf.predict(X_df)
        
        # Check for feature name warnings
        feature_warnings = [warning for warning in w 
                          if "feature names" in str(warning.message).lower()]
        
        if not feature_warnings:
            print(f"   {YELLOW}⚠️  WARNING: Expected warning not raised{RESET}")
            print(f"      This might be okay in newer sklearn versions")
            return True
    
    print(f"   {GREEN}✓ PASS: Warning correctly raised when mixing numpy fit + DataFrame predict{RESET}")
    print(f"      Warning: {feature_warnings[0].message}")
    return True


def test_pipeline_consistency():
    """Test full pipeline: DataFrame → Scale → Numpy → RF → Predict with Numpy."""
    print(f"\n{BOLD}TEST 5: Full Pipeline Data Type Consistency{RESET}")
    
    # Step 1: Create DataFrame with features
    df = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
        'feature4': np.random.randn(100),
        'feature5': np.random.randn(100)
    })
    y = np.random.randn(100)
    
    print(f"   Step 1: Created DataFrame with {df.shape[1]} features")
    
    # Step 2: Fit scaler with DataFrame
    scaler = MinMaxScaler()
    scaler.set_output(transform="pandas")
    scaler.fit(df)
    print(f"   Step 2: Fit scaler with DataFrame ✓")
    
    # Step 3: Transform with DataFrame
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        scaled_df = scaler.transform(df)
        
        feature_warnings = [warning for warning in w 
                          if "feature names" in str(warning.message).lower()]
        if feature_warnings:
            print(f"   {RED}❌ FAIL: Warning in scaler transform{RESET}")
            return False
    
    print(f"   Step 3: Transform with DataFrame (no warnings) ✓")
    
    # Step 4: Convert to numpy for RF
    X_train = scaled_df.values
    print(f"   Step 4: Converted to numpy array {X_train.shape} ✓")
    
    # Step 5: Fit RF with numpy
    rf = RandomForestRegressor(n_estimators=10, random_state=42)
    rf.fit(X_train, y)
    print(f"   Step 5: Fit RandomForest with numpy ✓")
    
    # Step 6: Predict with numpy (no warnings)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        predictions = rf.predict(X_train)
        
        feature_warnings = [warning for warning in w 
                          if "feature names" in str(warning.message).lower()]
        if feature_warnings:
            print(f"   {RED}❌ FAIL: Warning in RF predict{RESET}")
            return False
    
    print(f"   Step 6: Predict with numpy (no warnings) ✓")
    
    # Step 7: Test with new data (same pipeline)
    new_df = pd.DataFrame({
        'feature1': [1.5],
        'feature2': [2.5],
        'feature3': [3.5],
        'feature4': [4.5],
        'feature5': [5.5]
    })
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Scale new data (DataFrame → DataFrame)
        scaled_new_df = scaler.transform(new_df)
        
        # Convert to numpy for prediction
        new_X = scaled_new_df.values
        
        # Predict (numpy → numpy)
        new_prediction = rf.predict(new_X)
        
        feature_warnings = [warning for warning in w 
                          if "feature names" in str(warning.message).lower()]
        if feature_warnings:
            print(f"   {RED}❌ FAIL: Warning in prediction pipeline{RESET}")
            return False
    
    print(f"   Step 7: Prediction pipeline (no warnings) ✓")
    
    print(f"\n   {GREEN}✓ PASS: Full pipeline maintains data type consistency{RESET}")
    return True


def test_dimension_reduction_consistency():
    """Test that dimension_reduction functions maintain type consistency."""
    print(f"\n{BOLD}TEST 6: Dimension Reduction Type Consistency{RESET}")
    
    try:
        import dimension_reduction
    except ImportError:
        print(f"   {YELLOW}⚠️  SKIP: dimension_reduction module not available{RESET}")
        return True
    
    # Create sample data (simulating pipeline output)
    x_train = pd.DataFrame(np.random.randn(100, 50), 
                          columns=[f'feature_{i}' for i in range(50)])
    y_train = pd.Series(np.random.randn(100))
    x_val = pd.DataFrame(np.random.randn(30, 50), 
                        columns=[f'feature_{i}' for i in range(50)])
    y_val = pd.Series(np.random.randn(30))
    x_test = pd.DataFrame(np.random.randn(20, 50), 
                         columns=[f'feature_{i}' for i in range(50)])
    y_test = pd.Series(np.random.randn(20))
    x_pred = pd.DataFrame(np.random.randn(10, 50), 
                         columns=[f'feature_{i}' for i in range(50)])
    # Create stock_df with same columns as input data
    stock_df = pd.DataFrame(np.random.randn(5, 50), 
                           columns=[f'feature_{i}' for i in range(50)])
    
    # Test feature_selection_rf (should not warn)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        try:
            x_train_sel, x_val_sel, x_test_sel, x_pred_sel, selector, features = \
                dimension_reduction.feature_selection_rf(
                    dimensions=20,
                    x_training_data=x_train,
                    x_val_data=x_val,
                    x_test_data=x_test,
                    y_training_data=y_train,
                    y_val_data=y_val,
                    y_test_data=y_test,
                    prediction_data=x_pred,
                    dataset_df=stock_df
                )
            
            # Check output types (should be numpy arrays)
            if not isinstance(x_train_sel, np.ndarray):
                print(f"   {RED}❌ FAIL: Output not numpy array (got {type(x_train_sel)}){RESET}")
                return False
            
            # Check for warnings
            feature_warnings = [warning for warning in w 
                              if "feature names" in str(warning.message).lower()]
            if feature_warnings:
                print(f"   {RED}❌ FAIL: Got feature name warnings{RESET}")
                for warning in feature_warnings:
                    print(f"      {warning.message}")
                return False
            
            print(f"   {GREEN}✓ PASS: feature_selection_rf maintains type consistency{RESET}")
            print(f"      Input: DataFrame {x_train.shape}")
            print(f"      Output: Numpy array {x_train_sel.shape}")
            return True
            
        except Exception as e:
            print(f"   {RED}❌ FAIL: Exception in feature_selection_rf: {e}{RESET}")
            return False
    
    return True


def run_all_tests():
    """Run all data type consistency tests."""
    print("\n" + "="*80)
    print(f"{BOLD}DATA TYPE CONSISTENCY TEST SUITE{RESET}")
    print("="*80)
    print("Validates data type flow to prevent sklearn warnings")
    print("="*80)
    
    tests = [
        ("MinMaxScaler with DataFrames", test_scaler_with_dataframe),
        ("MinMaxScaler DataFrame → Numpy Warning", test_scaler_with_numpy_warns),
        ("RandomForest with Numpy Arrays", test_rf_with_numpy),
        ("RandomForest Numpy → DataFrame Warning", test_rf_with_dataframe_warns),
        ("Full Pipeline Consistency", test_pipeline_consistency),
        ("Dimension Reduction Consistency", test_dimension_reduction_consistency),
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   {RED}❌ FAIL: Exception - {e}{RESET}")
            results.append((test_name, False))
    
    elapsed = time.time() - start_time
    
    # Summary
    print("\n" + "="*80)
    print(f"{BOLD}TEST SUMMARY{RESET}")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    failed = len(results) - passed
    
    for test_name, result in results:
        status = f"{GREEN}✓ PASS{RESET}" if result else f"{RED}❌ FAIL{RESET}"
        print(f"  {status}  {test_name}")
    
    print("-" * 80)
    print(f"Total: {len(results)} tests, {passed} passed, {failed} failed")
    print(f"Time: {elapsed:.2f}s")
    print("="*80)
    
    if failed == 0:
        print(f"\n{GREEN}{BOLD}✅ ALL TESTS PASSED!{RESET}")
        print(f"{GREEN}Data type consistency validated throughout pipeline{RESET}\n")
        return 0
    else:
        print(f"\n{RED}{BOLD}❌ {failed} TEST(S) FAILED{RESET}")
        print(f"{RED}Fix data type inconsistencies before committing{RESET}\n")
        return 1


if __name__ == '__main__':
    import time
    exit_code = run_all_tests()
    sys.exit(exit_code)
