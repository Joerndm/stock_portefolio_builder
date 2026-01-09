"""
Test LSTM Input Robustness

Verifies that LSTM input preparation handles both DataFrame and NumPy array
inputs correctly, regardless of what the scaler returns.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def test_lstm_input_with_dataframe_scaler_output():
    """Test LSTM input when scaler returns DataFrame (edge case)"""
    
    print("\n" + "="*80)
    print("TESTING LSTM INPUT ROBUSTNESS - DATAFRAME SCALER OUTPUT")
    print("="*80)
    
    time_steps = 30
    num_features = 10
    
    # Simulate stock_mod_df
    sample_data = np.random.randn(50, num_features) * 10 + 100
    df = pd.DataFrame(sample_data, columns=[f'feature_{i}' for i in range(num_features)])
    
    # Create scaler
    scaler_x = MinMaxScaler()
    scaler_x.fit(df)
    
    # Extract last time_steps rows
    x_lstm_df = df.iloc[-time_steps:]
    
    print(f"\n1. Input type: {type(x_lstm_df)}")
    
    # THE ROBUST FIX: Handle both DataFrame and NumPy array inputs
    x_lstm_array = x_lstm_df.values if hasattr(x_lstm_df, 'values') else np.array(x_lstm_df)
    print(f"2. After .values conversion: {type(x_lstm_array)}, shape: {x_lstm_array.shape}")
    
    scaled_x_lstm_array = scaler_x.transform(x_lstm_array)
    print(f"3. After scaler.transform(): {type(scaled_x_lstm_array)}, shape: {scaled_x_lstm_array.shape}")
    
    # Handle case where scaler might return DataFrame (shouldn't happen with sklearn, but be defensive)
    if hasattr(scaled_x_lstm_array, 'values'):
        print("   ⚠️  Scaler returned DataFrame - converting to NumPy array")
        scaled_x_lstm_array = scaled_x_lstm_array.values
    
    print(f"4. Final array type: {type(scaled_x_lstm_array)}, shape: {scaled_x_lstm_array.shape}")
    
    # Now reshape should work
    try:
        scaled_x_input_lstm = scaled_x_lstm_array.reshape(1, time_steps, scaled_x_lstm_array.shape[1])
        print(f"5. ✓ Reshape successful: {scaled_x_input_lstm.shape}")
        print(f"   Expected shape: (1, {time_steps}, {num_features})")
        
        if scaled_x_input_lstm.shape == (1, time_steps, num_features):
            print("   ✓ Shape is correct!")
            return True
        else:
            print("   ✗ Shape mismatch!")
            return False
            
    except AttributeError as e:
        print(f"5. ✗ Reshape failed: {e}")
        return False

def test_lstm_input_with_numpy_array_input():
    """Test LSTM input when input is already NumPy array"""
    
    print("\n" + "="*80)
    print("TESTING LSTM INPUT ROBUSTNESS - NUMPY ARRAY INPUT")
    print("="*80)
    
    time_steps = 30
    num_features = 10
    
    # Simulate input as NumPy array directly
    x_lstm_array = np.random.randn(time_steps, num_features) * 10 + 100
    
    # Create and fit scaler
    scaler_x = MinMaxScaler()
    scaler_x.fit(x_lstm_array)
    
    print(f"\n1. Input type: {type(x_lstm_array)}, shape: {x_lstm_array.shape}")
    
    # THE ROBUST FIX: Handle both DataFrame and NumPy array inputs
    x_lstm_array_converted = x_lstm_array.values if hasattr(x_lstm_array, 'values') else np.array(x_lstm_array)
    print(f"2. After conversion check: {type(x_lstm_array_converted)}, shape: {x_lstm_array_converted.shape}")
    
    scaled_x_lstm_array = scaler_x.transform(x_lstm_array_converted)
    print(f"3. After scaler.transform(): {type(scaled_x_lstm_array)}, shape: {scaled_x_lstm_array.shape}")
    
    # Handle case where scaler might return DataFrame
    if hasattr(scaled_x_lstm_array, 'values'):
        print("   ⚠️  Scaler returned DataFrame - converting to NumPy array")
        scaled_x_lstm_array = scaled_x_lstm_array.values
    
    print(f"4. Final array type: {type(scaled_x_lstm_array)}, shape: {scaled_x_lstm_array.shape}")
    
    # Now reshape should work
    try:
        scaled_x_input_lstm = scaled_x_lstm_array.reshape(1, time_steps, scaled_x_lstm_array.shape[1])
        print(f"5. ✓ Reshape successful: {scaled_x_input_lstm.shape}")
        print(f"   Expected shape: (1, {time_steps}, {num_features})")
        
        if scaled_x_input_lstm.shape == (1, time_steps, num_features):
            print("   ✓ Shape is correct!")
            return True
        else:
            print("   ✗ Shape mismatch!")
            return False
            
    except AttributeError as e:
        print(f"5. ✗ Reshape failed: {e}")
        return False

if __name__ == "__main__":
    test1 = test_lstm_input_with_dataframe_scaler_output()
    test2 = test_lstm_input_with_numpy_array_input()
    
    print("\n" + "="*80)
    if test1 and test2:
        print("✓ ALL ROBUSTNESS TESTS PASSED")
        print("="*80)
        exit(0)
    else:
        print("✗ SOME TESTS FAILED")
        print("="*80)
        exit(1)
