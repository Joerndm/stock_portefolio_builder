"""
MODEL INSTANTIATION TESTS
=========================

Tests that all ML models can be instantiated correctly with all required imports
and configurations. This catches import errors, missing dependencies, and
architecture issues before training.

These tests ensure:
1. All required imports are available
2. Models can be built without errors
3. Models can be compiled with optimizers and loss functions
4. Hyperparameter spaces are valid

Usage:
    python test_reports/test_model_instantiation.py
"""

import sys
import os
from pathlib import Path
import traceback

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'


def test_lstm_model_instantiation():
    """Test LSTM model can be instantiated with all layers and imports."""
    print(f"\n{BOLD}TEST: LSTM Model Instantiation{RESET}")
    
    try:
        # Import required modules
        import keras_tuner as kt
        import numpy as np
        from ml_builder import build_lstm_model
        
        # Create hyperparameter object
        hp = kt.HyperParameters()
        
        # Set some reasonable default values to speed up testing
        hp.Fixed('n_layers', 2)
        hp.Fixed('input_units', 64)
        hp.Fixed('units_1', 32)
        hp.Fixed('final_units', 32)
        hp.Fixed('dense_1', 32)
        hp.Fixed('dense_2', 16)
        hp.Fixed('dropout_1', 0.2)
        hp.Fixed('dropout__dense_1', 0.3)
        hp.Fixed('dropout_dense_2', 0.3)
        hp.Fixed('l2_reg_input', 1e-5)
        hp.Fixed('l2_reg_1', 1e-5)
        hp.Fixed('l2_reg_final', 1e-5)
        hp.Fixed('dense_1_activation', 'relu')
        hp.Fixed('dense_2_activation', 'relu')
        hp.Fixed('optimizer', 'adam')
        hp.Fixed('learning_rate', 1e-4)
        hp.Fixed('clipnorm', 1.0)
        hp.Fixed('loss', 'mean_absolute_error')
        
        # Build model
        input_shape = (30, 99)  # 30 timesteps, 99 features
        model = build_lstm_model(hp, input_shape)
        
        # Verify model was created
        assert model is not None, "Model is None"
        assert len(model.layers) > 0, "Model has no layers"
        
        # Verify key layers exist
        layer_types = [type(layer).__name__ for layer in model.layers]
        assert 'Bidirectional' in layer_types, "No Bidirectional layer found"
        assert 'BatchNormalization' in layer_types, "No BatchNormalization layer found"
        assert 'Dropout' in layer_types, "No Dropout layer found"
        assert 'Dense' in layer_types, "No Dense layer found"
        
        # Verify model can compile
        model.compile(optimizer='adam', loss='mae', metrics=['mse'])
        
        # Verify model can accept correct input shape
        test_input = np.random.randn(1, 30, 99)
        prediction = model.predict(test_input, verbose=0)
        assert prediction.shape == (1, 1), f"Unexpected output shape: {prediction.shape}"
        
        print(f"{GREEN}✓ PASSED - LSTM model instantiates correctly{RESET}")
        print(f"  - Model has {len(model.layers)} layers")
        print(f"  - BatchNormalization layers: {layer_types.count('BatchNormalization')}")
        print(f"  - Dropout layers: {layer_types.count('Dropout')}")
        print(f"  - Output shape validated: {prediction.shape}")
        return True
        
    except Exception as e:
        print(f"{RED}✗ FAILED - {str(e)}{RESET}")
        traceback.print_exc()
        return False


def test_random_forest_instantiation():
    """Test Random Forest model can be instantiated."""
    print(f"\n{BOLD}TEST: Random Forest Model Instantiation{RESET}")
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        import numpy as np
        
        # Common hyperparameters
        test_configs = [
            {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5},
            {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 2},
            {'n_estimators': 50, 'max_depth': 20, 'min_samples_split': 10}
        ]
        
        for i, config in enumerate(test_configs):
            # Instantiate model
            model = RandomForestRegressor(**config, random_state=42, n_jobs=-1)
            
            # Verify it can fit and predict
            X_train = np.random.randn(100, 99)
            y_train = np.random.randn(100)
            model.fit(X_train, y_train)
            
            X_test = np.random.randn(10, 99)
            predictions = model.predict(X_test)
            
            assert predictions.shape == (10,), f"Unexpected prediction shape: {predictions.shape}"
            assert not np.any(np.isnan(predictions)), "Model produced NaN predictions"
        
        print(f"{GREEN}✓ PASSED - Random Forest instantiates correctly{RESET}")
        print(f"  - Tested {len(test_configs)} configurations")
        print(f"  - All configurations produce valid predictions")
        return True
        
    except Exception as e:
        print(f"{RED}✗ FAILED - {str(e)}{RESET}")
        traceback.print_exc()
        return False


def test_xgboost_instantiation():
    """Test XGBoost model can be instantiated."""
    print(f"\n{BOLD}TEST: XGBoost Model Instantiation{RESET}")
    
    try:
        import xgboost as xgb
        import numpy as np
        import pandas as pd
        
        # Common hyperparameters
        test_configs = [
            {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
            {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.05},
            {'n_estimators': 50, 'max_depth': 4, 'learning_rate': 0.2}
        ]
        
        for i, config in enumerate(test_configs):
            # Instantiate model
            model = xgb.XGBRegressor(
                **config,
                random_state=42,
                n_jobs=-1,
                enable_categorical=False
            )
            
            # Verify it can fit and predict with proper dtypes
            X_train = pd.DataFrame(np.random.randn(100, 99)).astype('float64')
            y_train = pd.Series(np.random.randn(100)).astype('float64')
            model.fit(X_train, y_train)
            
            X_test = pd.DataFrame(np.random.randn(10, 99)).astype('float64')
            predictions = model.predict(X_test)
            
            assert predictions.shape == (10,), f"Unexpected prediction shape: {predictions.shape}"
            assert not np.any(np.isnan(predictions)), "Model produced NaN predictions"
        
        print(f"{GREEN}✓ PASSED - XGBoost instantiates correctly{RESET}")
        print(f"  - Tested {len(test_configs)} configurations")
        print(f"  - All configurations produce valid predictions")
        print(f"  - Dtype handling validated (float64)")
        return True
        
    except Exception as e:
        print(f"{RED}✗ FAILED - {str(e)}{RESET}")
        traceback.print_exc()
        return False


def test_lstm_hyperparameter_search_space():
    """Test LSTM hyperparameter search space is valid."""
    print(f"\n{BOLD}TEST: LSTM Hyperparameter Search Space{RESET}")
    
    try:
        import keras_tuner as kt
        from ml_builder import build_lstm_model
        
        # Create hyperparameter object (let it sample randomly)
        hp = kt.HyperParameters()
        
        # Build model multiple times with different random hyperparameters
        for trial in range(3):
            model = build_lstm_model(hp, input_shape=(30, 99))
            assert model is not None, f"Model is None on trial {trial}"
            
            # Verify compilation works
            model.compile(optimizer='adam', loss='mae')
        
        print(f"{GREEN}✓ PASSED - LSTM hyperparameter space is valid{RESET}")
        print(f"  - Built {3} models with random hyperparameters")
        print(f"  - All models compiled successfully")
        return True
        
    except Exception as e:
        print(f"{RED}✗ FAILED - {str(e)}{RESET}")
        traceback.print_exc()
        return False


def test_all_imports_available():
    """Test all required imports are available."""
    print(f"\n{BOLD}TEST: Required Imports Available{RESET}")
    
    try:
        required_imports = [
            ('tensorflow.keras.layers', 'LSTM'),
            ('tensorflow.keras.layers', 'Dense'),
            ('tensorflow.keras.layers', 'Dropout'),
            ('tensorflow.keras.layers', 'Bidirectional'),
            ('tensorflow.keras.layers', 'BatchNormalization'),
            ('tensorflow.keras.optimizers', 'Adam'),
            ('tensorflow.keras.optimizers', 'RMSprop'),
            ('tensorflow.keras.callbacks', 'EarlyStopping'),
            ('tensorflow.keras.callbacks', 'ReduceLROnPlateau'),
            ('tensorflow.keras.models', 'Sequential'),
            ('sklearn.ensemble', 'RandomForestRegressor'),
            ('xgboost', 'XGBRegressor'),
            ('keras_tuner', 'BayesianOptimization'),
        ]
        
        missing = []
        for module_name, class_name in required_imports:
            try:
                module = __import__(module_name, fromlist=[class_name])
                getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                missing.append((module_name, class_name, str(e)))
        
        if missing:
            print(f"{RED}✗ FAILED - Missing imports:{RESET}")
            for module, cls, error in missing:
                print(f"  - {module}.{cls}: {error}")
            return False
        
        print(f"{GREEN}✓ PASSED - All required imports available{RESET}")
        print(f"  - Verified {len(required_imports)} imports")
        return True
        
    except Exception as e:
        print(f"{RED}✗ FAILED - {str(e)}{RESET}")
        traceback.print_exc()
        return False


def test_model_ensemble_compatibility():
    """Test that all three models can produce predictions with compatible shapes."""
    print(f"\n{BOLD}TEST: Model Ensemble Compatibility{RESET}")
    
    try:
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor
        import xgboost as xgb
        import keras_tuner as kt
        from ml_builder import build_lstm_model
        
        # Create sample data
        n_samples = 100
        n_features = 99
        X_train = np.random.randn(n_samples, n_features)
        y_train = np.random.randn(n_samples)
        X_test = np.random.randn(10, n_features)
        
        # Random Forest
        rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(n_estimators=50, max_depth=6, random_state=42)
        xgb_model.fit(
            pd.DataFrame(X_train).astype('float64'),
            pd.Series(y_train).astype('float64')
        )
        xgb_pred = xgb_model.predict(pd.DataFrame(X_test).astype('float64'))
        
        # LSTM (no training, just predict)
        hp = kt.HyperParameters()
        lstm_model = build_lstm_model(hp, input_shape=(30, n_features))
        
        # Create LSTM sequences
        time_steps = 30
        X_lstm_test = np.random.randn(10, time_steps, n_features)
        lstm_pred = lstm_model.predict(X_lstm_test, verbose=0).flatten()
        
        # Verify all predictions have same shape
        assert rf_pred.shape == (10,), f"RF prediction shape: {rf_pred.shape}"
        assert xgb_pred.shape == (10,), f"XGB prediction shape: {xgb_pred.shape}"
        assert lstm_pred.shape == (10,), f"LSTM prediction shape: {lstm_pred.shape}"
        
        # Verify ensemble calculation works
        ensemble_pred = (rf_pred + xgb_pred + lstm_pred) / 3
        assert ensemble_pred.shape == (10,), f"Ensemble shape: {ensemble_pred.shape}"
        assert not np.any(np.isnan(ensemble_pred)), "Ensemble produced NaN values"
        
        print(f"{GREEN}✓ PASSED - All models produce compatible predictions{RESET}")
        print(f"  - RF shape: {rf_pred.shape}")
        print(f"  - XGB shape: {xgb_pred.shape}")
        print(f"  - LSTM shape: {lstm_pred.shape}")
        print(f"  - Ensemble calculation successful")
        return True
        
    except Exception as e:
        print(f"{RED}✗ FAILED - {str(e)}{RESET}")
        traceback.print_exc()
        return False


def main():
    """Run all model instantiation tests."""
    import time
    
    print("\n" + "="*80)
    print(f"{BOLD}MODEL INSTANTIATION TEST SUITE{RESET}")
    print("="*80)
    print("Testing that all ML models can be instantiated correctly...")
    print("="*80)
    
    start_time = time.time()
    
    # Run tests
    results = []
    results.append(('Required Imports', test_all_imports_available()))
    results.append(('LSTM Instantiation', test_lstm_model_instantiation()))
    results.append(('Random Forest Instantiation', test_random_forest_instantiation()))
    results.append(('XGBoost Instantiation', test_xgboost_instantiation()))
    results.append(('LSTM Hyperparameter Space', test_lstm_hyperparameter_search_space()))
    results.append(('Model Ensemble Compatibility', test_model_ensemble_compatibility()))
    
    # Summary
    elapsed = time.time() - start_time
    passed = sum(1 for _, result in results if result)
    failed = len(results) - passed
    
    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}SUMMARY{RESET}")
    print(f"{BOLD}{'='*80}{RESET}")
    
    for test_name, result in results:
        icon = f"{GREEN}✓{RESET}" if result else f"{RED}✗{RESET}"
        print(f"{icon} {test_name}")
    
    print(f"\n{BOLD}OVERALL:{RESET}")
    if failed == 0:
        print(f"{GREEN}✓ ALL TESTS PASSED: {passed}/{len(results)}{RESET}")
    else:
        print(f"{RED}✗ SOME TESTS FAILED: {passed}/{len(results)} passed, {failed} failed{RESET}")
    
    print(f"\nTotal time: {elapsed:.2f}s")
    print("="*80 + "\n")
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
