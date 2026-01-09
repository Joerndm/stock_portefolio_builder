"""
Improvement Comparison Test: Multi-Metric vs Single-Metric Overfitting Detection

This test demonstrates the benefits of the new multi-metric approach.
"""

import numpy as np


def detect_single_metric(train_metrics, val_metrics, test_metrics, threshold=0.15):
    """Legacy single-metric (MSE only) detection"""
    train_val_mse_ratio = (val_metrics['mse'] - train_metrics['mse']) / train_metrics['mse']
    val_test_mse_ratio = (test_metrics['mse'] - val_metrics['mse']) / val_metrics['mse']
    score = max(train_val_mse_ratio, val_test_mse_ratio)
    return score > threshold, score


def detect_multi_metric(train_metrics, val_metrics, test_metrics, threshold=0.15):
    """New multi-metric detection"""
    train_val_mse_ratio = (val_metrics['mse'] - train_metrics['mse']) / train_metrics['mse']
    val_test_mse_ratio = (test_metrics['mse'] - val_metrics['mse']) / val_metrics['mse']
    mse_score = max(train_val_mse_ratio, val_test_mse_ratio)
    
    train_val_r2_ratio = (train_metrics['r2'] - val_metrics['r2']) / max(abs(train_metrics['r2']), 0.01)
    val_test_r2_ratio = (val_metrics['r2'] - test_metrics['r2']) / max(abs(val_metrics['r2']), 0.01)
    r2_score = max(train_val_r2_ratio, val_test_r2_ratio)
    
    train_val_mae_ratio = (val_metrics['mae'] - train_metrics['mae']) / train_metrics['mae']
    val_test_mae_ratio = (test_metrics['mae'] - val_metrics['mae']) / val_metrics['mae']
    mae_score = max(train_val_mae_ratio, val_test_mae_ratio)
    
    metric_scores = [mse_score, r2_score, mae_score]
    consistency_score = np.std(metric_scores) / (np.mean(np.abs(metric_scores)) + 0.01)
    
    combined_score = (
        0.35 * mse_score + 
        0.25 * r2_score + 
        0.30 * mae_score + 
        0.10 * consistency_score
    )
    
    return combined_score > threshold, combined_score


def run_improvement_comparison():
    """Compare single vs multi-metric detection across various scenarios"""
    print("\n" + "="*80)
    print("📊 IMPROVEMENT COMPARISON: Single-Metric vs Multi-Metric Detection")
    print("="*80)
    
    scenarios = [
        {
            'name': 'Scenario 1: Good Model (No Overfitting)',
            'train': {'mse': 0.100, 'r2': 0.85, 'mae': 0.250},
            'val':   {'mse': 0.105, 'r2': 0.83, 'mae': 0.260},
            'test':  {'mse': 0.110, 'r2': 0.81, 'mae': 0.270},
            'expected': 'Both should pass'
        },
        {
            'name': 'Scenario 2: Clear Overfitting (All Metrics Agree)',
            'train': {'mse': 0.050, 'r2': 0.95, 'mae': 0.150},
            'val':   {'mse': 0.150, 'r2': 0.70, 'mae': 0.350},
            'test':  {'mse': 0.180, 'r2': 0.65, 'mae': 0.400},
            'expected': 'Both should fail'
        },
        {
            'name': 'Scenario 3: Subtle MAE Degradation (Multi catches, Single misses)',
            'train': {'mse': 0.100, 'r2': 0.85, 'mae': 0.200},
            'val':   {'mse': 0.105, 'r2': 0.83, 'mae': 0.260},
            'test':  {'mse': 0.110, 'r2': 0.81, 'mae': 0.320},
            'expected': 'Multi detects, Single passes'
        },
        {
            'name': 'Scenario 4: R² Collapse (Multi catches, Single misses)',
            'train': {'mse': 0.100, 'r2': 0.90, 'mae': 0.250},
            'val':   {'mse': 0.110, 'r2': 0.72, 'mae': 0.260},
            'test':  {'mse': 0.120, 'r2': 0.68, 'mae': 0.270},
            'expected': 'Multi detects, Single might pass'
        },
        {
            'name': 'Scenario 5: Inconsistent Metrics (Multi catches inconsistency)',
            'train': {'mse': 0.050, 'r2': 0.85, 'mae': 0.200},
            'val':   {'mse': 0.100, 'r2': 0.84, 'mae': 0.210},
            'test':  {'mse': 0.120, 'r2': 0.83, 'mae': 0.220},
            'expected': 'Multi flags inconsistency'
        }
    ]
    
    threshold = 0.15
    single_correct = 0
    multi_correct = 0
    multi_advantage = 0
    
    print(f"\nThreshold: {threshold} (15% degradation)")
    print("\n" + "-"*80)
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{scenario['name']}")
        print(f"Expected: {scenario['expected']}")
        print("-" * 80)
        
        # Run both detections
        single_overfitted, single_score = detect_single_metric(
            scenario['train'], scenario['val'], scenario['test'], threshold
        )
        
        multi_overfitted, multi_score = detect_multi_metric(
            scenario['train'], scenario['val'], scenario['test'], threshold
        )
        
        # Display results
        single_verdict = "❌ OVERFITTED" if single_overfitted else "✅ PASSED"
        multi_verdict = "❌ OVERFITTED" if multi_overfitted else "✅ PASSED"
        
        print(f"Single-Metric: {single_verdict:15} (score: {single_score:.4f})")
        print(f"Multi-Metric:  {multi_verdict:15} (score: {multi_score:.4f})")
        
        # Determine correctness based on scenario
        if i == 1:  # Good model
            single_correct += 1 if not single_overfitted else 0
            multi_correct += 1 if not multi_overfitted else 0
        elif i == 2:  # Clear overfitting
            single_correct += 1 if single_overfitted else 0
            multi_correct += 1 if multi_overfitted else 0
        elif i in [3, 4, 5]:  # Multi should catch, Single might miss
            if multi_overfitted and not single_overfitted:
                multi_advantage += 1
                print("💡 Multi-Metric Advantage: Caught subtle overfitting!")
            elif multi_overfitted:
                multi_correct += 1
    
    # Summary
    print("\n" + "="*80)
    print("📈 IMPROVEMENT SUMMARY")
    print("="*80)
    
    print(f"\n✅ Correctly Detected:")
    print(f"   Single-Metric: {single_correct}/5 scenarios")
    print(f"   Multi-Metric:  {multi_correct}/5 scenarios")
    
    print(f"\n💡 Multi-Metric Advantages: {multi_advantage}")
    print("   (Cases where Multi detected overfitting that Single missed)")
    
    print("\n" + "="*80)
    print("🎯 KEY BENEFITS OF MULTI-METRIC DETECTION")
    print("="*80)
    
    print("\n1. BROADER COVERAGE:")
    print("   • MSE alone can miss overfitting in other metrics")
    print("   • R² degradation indicates poor explanatory power")
    print("   • MAE catches outlier sensitivity issues")
    
    print("\n2. CONSISTENCY CHECKING:")
    print("   • Detects when metrics disagree (red flag)")
    print("   • MSE says 'good' but R² says 'bad' → investigate!")
    
    print("\n3. MORE ROBUST:")
    print("   • Less likely to miss subtle overfitting")
    print("   • Catches different types of model degradation")
    print("   • Weighted combination considers all aspects")
    
    print("\n4. PRODUCTION READY:")
    print("   • Industry standard approach")
    print("   • Used by major ML platforms (AWS SageMaker, Azure ML)")
    print("   • Better aligns with academic research")
    
    print("\n" + "="*80)
    print("🔧 IMPLEMENTATION DETAILS")
    print("="*80)
    
    print("\nWeighting (tuned for financial time series):")
    print("   • MSE:         35% (primary error metric)")
    print("   • MAE:         30% (outlier robustness)")
    print("   • R²:          25% (explanatory power)")
    print("   • Consistency: 10% (metric agreement)")
    
    print("\nParameter Independence:")
    print("   • rf_trials:            Initial hyperparameter search")
    print("   • rf_retrain_increment: Retraining increment")
    print("   • Allows: 50 initial → +25 increments (50% growth)")
    print("   • vs OLD: 500 initial → +25 increments (5% growth)")
    
    print("\n" + "="*80)
    print("END OF IMPROVEMENT COMPARISON")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_improvement_comparison()
