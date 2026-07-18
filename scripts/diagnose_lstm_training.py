"""
Diagnose LSTM Training Data Issue

This script checks if the training data has a severe negative bias
that caused the LSTM to learn to predict a constant value.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import pandas as pd
import numpy as np
import db_interactions

# Load the same stock data
stock_symbol = input("Enter stock symbol (e.g., DEMANT.CO): ").strip()
stock_data_df = db_interactions.import_stock_dataset(stock_symbol)
stock_data_df["date"] = pd.to_datetime(stock_data_df["date"])
stock_data_df = stock_data_df.dropna(axis=0, how="any")
stock_data_df = stock_data_df.dropna(axis=1, how="any")

print("\n" + "="*80)
print("TRAINING DATA DIAGNOSIS")
print("="*80)

# Check the target variable (1D - daily returns)
if "1D" in stock_data_df.columns:
    target = stock_data_df["1D"]
    
    print(f"\nTarget Variable (1D - Daily Returns) Statistics:")
    print(f"  Total samples: {len(target)}")
    print(f"  Mean: {target.mean():.6f} ({target.mean()*100:.3f}%)")
    print(f"  Median: {target.median():.6f} ({target.median()*100:.3f}%)")
    print(f"  Std Dev: {target.std():.6f}")
    print(f"  Min: {target.min():.6f} ({target.min()*100:.3f}%)")
    print(f"  Max: {target.max():.6f} ({target.max()*100:.3f}%)")
    
    # Check distribution
    positive_days = (target > 0).sum()
    negative_days = (target < 0).sum()
    neutral_days = (target == 0).sum()
    
    print(f"\n  Distribution:")
    print(f"    Positive days: {positive_days} ({positive_days/len(target)*100:.1f}%)")
    print(f"    Negative days: {negative_days} ({negative_days/len(target)*100:.1f}%)")
    print(f"    Neutral days: {neutral_days} ({neutral_days/len(target)*100:.1f}%)")
    
    # Check if there's a severe bias
    if abs(target.mean()) > 0.01:  # More than 1% average daily change
        print(f"\n  ⚠️ WARNING: Severe bias detected!")
        print(f"     Average daily change of {target.mean()*100:.3f}% is unrealistic")
        print(f"     This will cause the model to learn a constant prediction")
    
    # Check training split (last 65% of data)
    train_size = int(len(target) * 0.65)
    train_target = target.iloc[:train_size]
    
    print(f"\n  Training Set (first 65% = {train_size} samples):")
    print(f"    Mean: {train_target.mean():.6f} ({train_target.mean()*100:.3f}%)")
    print(f"    Median: {train_target.median():.6f} ({train_target.median()*100:.3f}%)")
    
    # The LSTM is predicting -0.0736 scaled, which unscales to -0.169 (-16.9%)
    # This suggests the scaler learned from data with -16.9% mean
    print(f"\n  🔍 LSTM is predicting: -16.9% constantly")
    print(f"     Training mean: {train_target.mean()*100:.3f}%")
    
    if abs(train_target.mean() - (-0.169)) < 0.05:
        print(f"     ✓ CONFIRMED: LSTM learned to predict the training mean!")
        print(f"     This happens when the model collapses to the simplest solution")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    print("1. Check if stock data has errors (unlikely to have -16.9% daily decline)")
    print("2. Verify 1D calculation is correct (should be daily % change)")
    print("3. Retrain LSTM with:")
    print("   - More regularization (dropout, L2)")
    print("   - Different architecture")
    print("   - Better loss function (MAE instead of MSE)")
    print("   - Data normalization/detrending")
    
else:
    print("\n❌ ERROR: '1D' column not found in dataset")

print("\n" + "="*80)
