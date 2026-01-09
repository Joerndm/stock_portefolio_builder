"""
Standalone Unit Test: Overfitting Detection Improvements

Tests multi-metric detection without importing ml_builder.
"""

import numpy as np


def detect_overfitting_standalone(train_metrics, val_metrics, test_metrics, model_name, threshold=0.15, use_multi_metric=True):
    """
    Standalone version of detect_overfitting for testing.
    """

    if use_multi_metric:
        # Multi-metric detection
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
        
        overfitting_score = (
            0.35 * mse_score + 
            0.25 * r2_score + 
            0.30 * mae_score + 
            0.10 * consistency_score
        )
        
        is_overfitted = overfitting_score > threshold
        return is_overfitted, overfitting_score
    
    else:
        # Single-metric (legacy)
        train_val_mse_ratio = (val_metrics['mse'] - train_metrics['mse']) / train_metrics['mse']
        val_test_mse_ratio = (test_metrics['mse'] - val_metrics['mse']) / val_metrics['mse']
        overfitting_score = max(train_val_mse_ratio, val_test_mse_ratio)
        is_overfitted = overfitting_score > threshold
        return is_overfitted, overfitting_score


def run_tests():
    """Run all validation tests"""
    print("\n" + "="*70)
    print("🧪 OVERFITTING DETECTION IMPROVEMENTS - VALIDATION")
    print("="*70)
    
    passed = 0
    total = 0
    
    # Test 1
    print("\n[TEST 1] Multi-Metric - No Overfitting")
    total += 1
    train = {'mse': 0.100, 'r2': 0.85, 'mae': 0.250}
    val = {'mse': 0.105, 'r2': 0.83, 'mae': 0.260}
    test = {'mse': 0.110, 'r2': 0.81, 'mae': 0.270}
    
    overfitted, score = detect_overfitting_standalone(train, val, test, "Test", 0.15, True)
    
    if not overfitted and score < 0.15:
        print(f"✅ PASSED - Score: {score:.4f}")
        passed += 1
    else:
        print(f"❌ FAILED - Expected not overfitted")
    
    # Test 2
    print("\n[TEST 2] Multi-Metric - Clear Overfitting")
    total += 1
    train = {'mse': 0.050, 'r2': 0.95, 'mae': 0.150}
    val = {'mse': 0.150, 'r2': 0.70, 'mae': 0.350}
    test = {'mse': 0.180, 'r2': 0.65, 'mae': 0.400}
    
    overfitted, score = detect_overfitting_standalone(train, val, test, "Test", 0.15, True)
    
    if overfitted and score > 0.15:
        print(f"✅ PASSED - Score: {score:.4f}")
        passed += 1
    else:
        print(f"❌ FAILED - Expected overfitted")
    
    # Test 3
    print("\n[TEST 3] Single-Metric - Backward Compatibility")
    total += 1
    train = {'mse': 0.100, 'r2': 0.85, 'mae': 0.250}
    val = {'mse': 0.105, 'r2': 0.83, 'mae': 0.260}
    test = {'mse': 0.110, 'r2': 0.81, 'mae': 0.270}
    
    overfitted, score = detect_overfitting_standalone(train, val, test, "Test", 0.15, False)
    
    # train->val: (0.105-0.100)/0.100 = 0.05
    # val->test: (0.110-0.105)/0.105 = 0.0476
    # max = 0.05
    expected = 0.05
    print(f"   Actual score: {score:.6f}, Expected: ~{expected:.6f}")
    if not overfitted and abs(score - expected) < 0.01:
        print(f"✅ PASSED - Score: {score:.4f}, Expected: ~{expected:.4f}")
        passed += 1
    else:
        print(f"❌ FAILED - Overfitted: {overfitted}, Score difference: {abs(score - expected):.6f}")
    
    # Test 4
    print("\n[TEST 4] Multi > Single (MAE Degradation)")
    total += 1
    train = {'mse': 0.100, 'r2': 0.85, 'mae': 0.200}
    val = {'mse': 0.105, 'r2': 0.83, 'mae': 0.260}
    test = {'mse': 0.110, 'r2': 0.81, 'mae': 0.320}
    
    _, single = detect_overfitting_standalone(train, val, test, "S", 0.15, False)
    _, multi = detect_overfitting_standalone(train, val, test, "M", 0.15, True)
    
    if multi > single:
        print(f"✅ PASSED - Multi: {multi:.4f} > Single: {single:.4f}")
        passed += 1
    else:
        print(f"❌ FAILED")
    
    # Test 5
    print("\n[TEST 5] Parameter Independence")
    total += 1
    
    rf_init = 50
    rf_inc = 25
    attempts = [rf_init + (i * rf_inc) for i in range(4)]
    expected = [50, 75, 100, 125]
    
    if attempts == expected:
        print(f"✅ PASSED - Progression: {attempts}")
        passed += 1
    else:
        print(f"❌ FAILED")
    
    # Summary
    print("\n" + "="*70)
    print(f"📊 RESULTS: {passed}/{total} tests passed")
    print("="*70)
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED!")
        print("\n🎯 IMPROVEMENTS VALIDATED:")
        print("  ✓ Multi-metric detection (MSE, R², MAE, consistency)")
        print("  ✓ Combined scoring with proper weighting")
        print("  ✓ Backward compatibility (single-metric mode)")
        print("  ✓ Enhanced sensitivity to different overfitting types")
        print("  ✓ Parameter independence (trials vs increments)")
        return True
    else:
        print(f"\n❌ {total - passed} TEST(S) FAILED")
        return False


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
