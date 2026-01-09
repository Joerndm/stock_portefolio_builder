# LSTM Mode Collapse Fix - Architecture Improvements

## Problem Diagnosis
The LSTM model was experiencing **mode collapse** - outputting constant predictions of -16.9% regardless of input features:
- LSTM scaled output: **-0.0736** (constant across all predictions)
- Input features were varying correctly (mean 0.3788-0.3789, std 0.2990)
- RF and XGB models working correctly with 1-2% variation
- Direction accuracy: 50% (random guess level)
- **Root Cause**: Model learned to predict training data mean instead of responding to inputs

## Implemented Improvements

### 1. Batch Normalization ✅
**Location**: [ml_builder.py](ml_builder.py) lines ~968, 993, 1031
**Changes**:
- Added `BatchNormalization()` after first LSTM layer
- Added `BatchNormalization()` after each intermediate LSTM layer
- Added `BatchNormalization()` after final LSTM layer

**Purpose**: Stabilizes training by normalizing layer inputs, reducing internal covariate shift

### 2. Enhanced Loss Function Options ✅
**Location**: [ml_builder.py](ml_builder.py) line ~1104
**Changes**:
```python
loss_choice = hp.Choice(
    "loss",
    ["mean_absolute_error", "mean_squared_error", "huber", "mean_absolute_percentage_error"]
)
```
**Added**: `mean_absolute_error` (MAE) as first option
**Purpose**: MAE is less sensitive to outliers than MSE and reduces mode collapse risk

### 3. Increased Dropout Range ✅
**Location**: [ml_builder.py](ml_builder.py) line ~998
**Changes**:
```python
# Old: min_value=0.1, max_value=0.5
# New: min_value=0.2, max_value=0.6
Dropout(hp.Float(f"dropout_{i}", min_value=0.2, max_value=0.6, step=0.1))
```
**Purpose**: Prevents overfitting to training mean by requiring more regularization

### 4. Gradient Clipping ✅
**Location**: [ml_builder.py](ml_builder.py) line ~1095
**Changes**:
```python
clipnorm = hp.Float("clipnorm", 0.5, 2.0, step=0.5)
optimizer = Adam(learning_rate=learning_rate, clipnorm=clipnorm)
```
**Purpose**: Prevents exploding gradients that can cause training instability

### 5. Improved Callbacks ✅
**Location**: [ml_builder.py](ml_builder.py) line ~1188
**Changes**:
- Monitor `val_mean_absolute_error` instead of `val_loss`
- Increased patience range: 10-40 epochs (was 5-30)
- Added `min_delta=0.0001` to require meaningful improvement
- Added `min_lr=1e-7` to ReduceLROnPlateau

**Purpose**: Better stopping criteria to avoid premature convergence to local minima

## Next Steps

### 1. Delete Old Tuning Results (REQUIRED)
The old LSTM model is broken and needs to be retrained from scratch:
```powershell
# Delete partial tuning directory
Remove-Item -Recurse -Force "$env:TEMP\temp_tuning_dir\LSTM_tuning_DEMANT.CO"

# Delete finished tuning directory
Remove-Item -Recurse -Force ".\tuning_dir\LSTM_tuning_DEMANT.CO"
```

### 2. Run Diagnostic (RECOMMENDED)
Before retraining, check if training data has bias:
```powershell
python diagnose_lstm_training.py
# Enter stock symbol: DEMANT.CO
```

**What to look for**:
- Mean 1D daily returns (should be close to 0, not -16.9%)
- Distribution of positive/negative days
- Confirmation that LSTM learned training mean

### 3. Retrain LSTM Model
Run your existing training script. The new architecture will:
- Try MAE loss first (prioritized in loss_choice list)
- Use higher dropout (0.2-0.6 instead of 0.1-0.5)
- Apply batch normalization after each LSTM layer
- Use gradient clipping to prevent exploding gradients
- Monitor MAE for early stopping

**Expected Training Time**: Similar to before (depends on max_trials)

### 4. Verify Improvements
After retraining, check for:

✅ **LSTM outputs varying**: Not constant -0.0736
```python
# Debug output should show variation:
# Day 1: LSTM scaled: -0.023, unscaled: -5.2%
# Day 2: LSTM scaled: 0.015, unscaled: +3.1%
# Day 3: LSTM scaled: -0.041, unscaled: -9.7%
```

✅ **Direction accuracy > 60%**: Currently 50% (random)

✅ **Mean Absolute Error < 3%**: Currently ~16%

✅ **Predictions responsive to inputs**: Correlation with market changes

✅ **Ensemble improvement**: LSTM no longer dragging down RF+XGB

## Success Criteria

| Metric | Before (Broken) | Target (Fixed) |
|--------|----------------|----------------|
| LSTM scaled output | -0.0736 (constant) | Varies -0.1 to +0.05 |
| LSTM unscaled output | -16.9% (constant) | Varies -10% to +10% |
| Direction accuracy | 50% (random) | >60% |
| Mean Absolute Error | ~16% | <3% |
| Response to inputs | None | Strong correlation |

## Troubleshooting

### If LSTM still predicts constant value after retraining:

**Option A**: Check training data
```powershell
python diagnose_lstm_training.py
```
If training mean ≈ -16.9%, the data itself is biased. Consider:
- Verifying 1D calculation correctness
- Checking for data errors
- Using data normalization/detrending

**Option B**: Temporarily disable LSTM
Modify ensemble calculation in [ml_builder.py](ml_builder.py):
```python
# Weight LSTM at 0, use only RF + XGB
forecast_price_change = (0.0 * forecast_lstm + 0.5 * forecast_rf + 0.5 * forecast_xgb)
```

**Option C**: Try alternative architecture
- Use GRU layers instead of LSTM
- Add residual connections
- Implement attention mechanism
- Use smaller model (less capacity to memorize)

## Architecture Summary

**Before (Mode Collapse)**:
- Loss: MSE (sensitive to outliers)
- Dropout: 0.1-0.5
- No batch normalization
- No gradient clipping
- Monitor: val_loss

**After (Improved)**:
- Loss: MAE prioritized (less sensitive to outliers)
- Dropout: 0.2-0.6 (stronger regularization)
- Batch normalization after each LSTM layer
- Gradient clipping: 0.5-2.0
- Monitor: val_mean_absolute_error

## References

**Related Files**:
- `ml_builder.py` - LSTM architecture and training
- `diagnose_lstm_training.py` - Training data analysis tool
- `test_lstm_scaling_fix.py` - Scaling behavior tests
- `test_lstm_input_robustness.py` - Input handling tests

**Debug Locations**:
- Historical predictions debug: [ml_builder.py](ml_builder.py) lines ~2195-2210
- Future predictions: [ml_builder.py](ml_builder.py) lines ~2580-2620
- Training architecture: [ml_builder.py](ml_builder.py) lines ~942-1110
- Hyperparameter tuning: [ml_builder.py](ml_builder.py) lines ~1166-1220
