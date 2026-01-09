"""
Quick test script to verify HIGH PRIORITY improvements are working.

Run this after installing XGBoost to ensure everything works correctly.

Usage:
    python test_improvements.py
"""

def test_xgboost_import():
    """Test 1: Verify XGBoost is installed"""
    print("="*60)
    print("TEST 1: XGBoost Installation")
    print("="*60)
    try:
        import xgboost as xgb
        print(f"✅ XGBoost version: {xgb.__version__}")
        return True
    except ImportError as e:
        print(f"❌ XGBoost not installed!")
        print(f"   Install with: pip install xgboost>=1.7.6")
        return False

def test_function_signatures():
    """Test 2: Verify function signatures are updated"""
    print("\n" + "="*60)
    print("TEST 2: Function Signatures")
    print("="*60)
    
    import inspect
    import ml_builder
    
    # Check tune_lstm_model defaults
    sig = inspect.signature(ml_builder.tune_lstm_model)
    params = sig.parameters
    
    checks = []
    
    # Test max_trials default
    if params['max_trials'].default == 25:
        print("✅ LSTM max_trials default: 25")
        checks.append(True)
    else:
        print(f"❌ LSTM max_trials default: {params['max_trials'].default} (expected 25)")
        checks.append(False)
    
    # Test epochs default
    if params['epochs'].default == 50:
        print("✅ LSTM epochs default: 50")
        checks.append(True)
    else:
        print(f"❌ LSTM epochs default: {params['epochs'].default} (expected 50)")
        checks.append(False)
    
    # Check train_and_validate_models includes xgb_trials
    sig2 = inspect.signature(ml_builder.train_and_validate_models)
    if 'xgb_trials' in sig2.parameters:
        print("✅ train_and_validate_models has xgb_trials parameter")
        checks.append(True)
    else:
        print("❌ train_and_validate_models missing xgb_trials parameter")
        checks.append(False)
    
    return all(checks)

def test_xgboost_functions():
    """Test 3: Verify XGBoost functions exist"""
    print("\n" + "="*60)
    print("TEST 3: XGBoost Functions")
    print("="*60)
    
    import ml_builder
    
    functions = [
        'build_xgboost_model',
        'tune_xgboost_model',
        'evaluate_xgboost_model'
    ]
    
    checks = []
    for func_name in functions:
        if hasattr(ml_builder, func_name):
            print(f"✅ {func_name} exists")
            checks.append(True)
        else:
            print(f"❌ {func_name} not found")
            checks.append(False)
    
    return all(checks)

def test_bayesian_optimization():
    """Test 4: Verify Bayesian Optimization is used"""
    print("\n" + "="*60)
    print("TEST 4: Bayesian Optimization")
    print("="*60)
    
    import inspect
    import ml_builder
    
    # Read the source code
    source = inspect.getsource(ml_builder.tune_lstm_model)
    
    if 'BayesianOptimization' in source:
        print("✅ BayesianOptimization found in tune_lstm_model")
        return True
    else:
        print("❌ BayesianOptimization not found (still using RandomSearch?)")
        return False

def test_ensemble_logic():
    """Test 5: Verify ensemble logic exists"""
    print("\n" + "="*60)
    print("TEST 5: Ensemble Implementation")
    print("="*60)
    
    import inspect
    import ml_builder
    
    source = inspect.getsource(ml_builder.train_and_validate_models)
    
    checks = []
    
    # Check for ensemble weights calculation
    if 'ensemble_weights' in source or 'lstm_weight' in source:
        print("✅ Ensemble weight calculation found")
        checks.append(True)
    else:
        print("❌ Ensemble weight calculation not found")
        checks.append(False)
    
    # Check for ensemble predictions
    if 'ensemble_train_pred' in source or 'ensemble prediction' in source.lower():
        print("✅ Ensemble prediction logic found")
        checks.append(True)
    else:
        print("❌ Ensemble prediction logic not found")
        checks.append(False)
    
    # Check training_history includes ensemble
    if "'ensemble'" in source:
        print("✅ Ensemble tracked in training_history")
        checks.append(True)
    else:
        print("❌ Ensemble not in training_history")
        checks.append(False)
    
    return all(checks)

def test_lstm_layers():
    """Test 6: Verify LSTM max layers reduced to 5"""
    print("\n" + "="*60)
    print("TEST 6: LSTM Layer Reduction")
    print("="*60)
    
    import inspect
    import ml_builder
    
    source = inspect.getsource(ml_builder.build_lstm_model)
    
    # Look for max_amount_layers = 5
    if 'max_amount_layers = 5' in source or 'max_amount_layers=5' in source:
        print("✅ max_amount_layers set to 5")
        return True
    elif 'max_amount_layers = 20' in source:
        print("❌ max_amount_layers still set to 20 (should be 5)")
        return False
    else:
        print("⚠️  Could not verify max_amount_layers value")
        return None

def main():
    """Run all tests"""
    print("\n" + "🔬 TESTING HIGH PRIORITY IMPROVEMENTS")
    print("="*60 + "\n")
    
    results = []
    
    # Run tests
    results.append(("XGBoost Installation", test_xgboost_import()))
    results.append(("Function Signatures", test_function_signatures()))
    results.append(("XGBoost Functions", test_xgboost_functions()))
    results.append(("Bayesian Optimization", test_bayesian_optimization()))
    results.append(("Ensemble Logic", test_ensemble_logic()))
    results.append(("LSTM Layer Reduction", test_lstm_layers()))
    
    # Summary
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    skipped = sum(1 for _, result in results if result is None)
    
    for test_name, result in results:
        status = "✅ PASS" if result is True else ("❌ FAIL" if result is False else "⚠️  SKIP")
        print(f"{status} - {test_name}")
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("="*60 + "\n")
    
    if failed == 0:
        print("🎉 All tests passed! Implementation is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Review the output above.")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
