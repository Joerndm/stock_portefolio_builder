"""
Example demonstrating why feature selection on entire dataset causes data leakage
and overly optimistic performance estimates.
"""

import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Set random seed for reproducibility
np.random.seed(42)

# Create synthetic dataset
n_samples = 1000
n_features = 100

# Generate random features
X = np.random.randn(n_samples, n_features)

# True relationship: only features 0, 1, 2 actually matter
# The rest are noise
y = (3 * X[:, 0] + 
     2 * X[:, 1] + 
     1.5 * X[:, 2] + 
     np.random.randn(n_samples) * 0.5)  # Add small noise

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("="*80)
print("DATA LEAKAGE DEMONSTRATION")
print("="*80)
print(f"True important features: [0, 1, 2]")
print(f"Total features: {n_features}")
print(f"Selecting top 10 features...")
print()

# ============================================================================
# WRONG APPROACH: Feature selection on entire dataset
# ============================================================================
print("❌ WRONG: Feature selection using ALL data (train + test)")
print("-"*80)

# Fit selector on ALL data (WRONG!)
selector_wrong = SelectKBest(f_regression, k=10)
selector_wrong.fit(np.vstack([X_train, X_test]), np.concatenate([y_train, y_test]))

# Get selected features
selected_wrong = selector_wrong.get_support()
selected_features_wrong = [i for i, selected in enumerate(selected_wrong) if selected]
print(f"Selected features: {selected_features_wrong}")
print(f"Contains true features [0,1,2]? {all(f in selected_features_wrong for f in [0,1,2])}")

# Transform datasets
X_train_wrong = selector_wrong.transform(X_train)
X_test_wrong = selector_wrong.transform(X_test)

# Train model
model_wrong = LinearRegression()
model_wrong.fit(X_train_wrong, y_train)

# Evaluate
train_score_wrong = r2_score(y_train, model_wrong.predict(X_train_wrong))
test_score_wrong = r2_score(y_test, model_wrong.predict(X_test_wrong))

print(f"Training R²: {train_score_wrong:.4f}")
print(f"Test R²:     {test_score_wrong:.4f}")
print(f"Difference:  {abs(train_score_wrong - test_score_wrong):.4f}")
print()

# ============================================================================
# CORRECT APPROACH: Feature selection on training data only
# ============================================================================
print("✅ CORRECT: Feature selection using TRAINING data only")
print("-"*80)

# Fit selector on TRAINING data only (CORRECT!)
selector_correct = SelectKBest(f_regression, k=10)
selector_correct.fit(X_train, y_train)

# Get selected features
selected_correct = selector_correct.get_support()
selected_features_correct = [i for i, selected in enumerate(selected_correct) if selected]
print(f"Selected features: {selected_features_correct}")
print(f"Contains true features [0,1,2]? {all(f in selected_features_correct for f in [0,1,2])}")

# Transform datasets
X_train_correct = selector_correct.transform(X_train)
X_test_correct = selector_correct.transform(X_test)

# Train model
model_correct = LinearRegression()
model_correct.fit(X_train_correct, y_train)

# Evaluate
train_score_correct = r2_score(y_train, model_correct.predict(X_train_correct))
test_score_correct = r2_score(y_test, model_correct.predict(X_test_correct))

print(f"Training R²: {train_score_correct:.4f}")
print(f"Test R²:     {test_score_correct:.4f}")
print(f"Difference:  {abs(train_score_correct - test_score_correct):.4f}")
print()

# ============================================================================
# ANALYSIS
# ============================================================================
print("="*80)
print("ANALYSIS")
print("="*80)
print(f"Test R² difference: {test_score_wrong - test_score_correct:.4f}")
print()
print("Key Insights:")
print("1. The 'wrong' approach may show HIGHER test scores due to information leakage")
print("2. The 'correct' approach gives HONEST estimates of real-world performance")
print("3. In production, you won't have test data to select features from")
print("4. The 'correct' approach better simulates deployment conditions")
print()
print("⚠️  The small differences in this example would be MUCH larger with:")
print("   - Smaller datasets")
print("   - More noise in the data")
print("   - More irrelevant features")
print("   - Non-linear relationships")
print("="*80)
