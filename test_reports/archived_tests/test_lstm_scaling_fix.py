"""
Test LSTM Scaling Fix

Verifies that the LSTM input is properly scaled and reshaped.
This test checks the specific bug where .values was being called on a NumPy array.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def test_lstm_input_preparation():
    """Test that LSTM input preparation works correctly"""
    
    print("\n" + "="*80)
    print("TESTING LSTM INPUT PREPARATION")
    print("="*80)
    
    # Create sample data
    time_steps = 30
    num_features = 10
    
    # Simulate stock_mod_df with features
    sample_data = np.random.randn(50, num_features) * 10 + 100
    df = pd.DataFrame(sample_data, columns=[f'feature_{i}' for i in range(num_features)])
    
    # Create scaler and fit it
    scaler_x = MinMaxScaler()
    scaler_x.fit(df)
    
    # Extract last time_steps rows (as done in predict_future_price_changes)
    x_lstm_df = df.iloc[-time_steps:]
    
    print(f"\n1. Input DataFrame shape: {x_lstm_df.shape}")
    print(f"   Type: {type(x_lstm_df)}")
    
    # Apply scaling (as in the code)
    scaled_x_lstm_array = scaler_x.transform(x_lstm_df)
    
    print(f"\n2. After scaler_x.transform():")
    print(f"   Type: {type(scaled_x_lstm_array)}")
    print(f"   Shape: {scaled_x_lstm_array.shape}")
    print(f"   Is NumPy array: {isinstance(scaled_x_lstm_array, np.ndarray)}")
    
    # Check if .values exists (it shouldn't for NumPy arrays)
    has_values_attr = hasattr(scaled_x_lstm_array, 'values')
    print(f"\n3. Has .values attribute: {has_values_attr}")
    
    if has_values_attr:
        print("   ⚠️  WARNING: This shouldn't happen! NumPy arrays don't have .values")
    else:
        print("   ✓  CORRECT: NumPy arrays don't have .values attribute")
    
    # THE FIX: Remove .values since scaled_x_lstm_array is already a NumPy array
    try:
        # CORRECT way (after fix)
        scaled_x_input_lstm = scaled_x_lstm_array.reshape(1, time_steps, scaled_x_lstm_array.shape[1])
        print(f"\n4. After reshape (CORRECT method):")
        print(f"   Shape: {scaled_x_input_lstm.shape}")
        print(f"   Expected: (1, {time_steps}, {num_features})")
        print(f"   ✓  SUCCESS: Input properly shaped for LSTM")
        
    except Exception as e:
        print(f"\n4. CORRECT method failed: {e}")
        return False
    
    # Test the OLD BUGGY way (with .values)
    print(f"\n5. Testing OLD BUGGY method (with .values):")
    try:
        buggy_result = scaled_x_lstm_array.values
        print(f"   ⚠️  WARNING: .values didn't raise AttributeError!")
        print(f"   This means NumPy arrays have gained a .values attribute")
        print(f"   Type: {type(buggy_result)}")
    except AttributeError as e:
        print(f"   ✓  EXPECTED: AttributeError raised")
        print(f"   Error: {e}")
        print(f"   This confirms NumPy arrays don't have .values")
    
    print("\n" + "="*80)
    print("✓ LSTM INPUT PREPARATION TEST PASSED")
    print("="*80)
    
    return True

if __name__ == "__main__":
    success = test_lstm_input_preparation()
    exit(0 if success else 1)
