"""
Test Feature Coverage in predict_future_price_changes
======================================================

This test validates that all features created in stock_data_fetch.py are properly
handled in ml_builder.py's predict_future_price_changes function.

Purpose:
- Ensures all technical indicators calculated are also handled in predictions
- Validates both historical_prediction_dataset_df and future_df loops
- Prevents missing features from causing prediction failures

Author: Test Suite
Date: December 17, 2025
"""

import re
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_features_from_stock_data_fetch():
    """
    Extracts all features calculated in stock_data_fetch.py
    
    Returns:
        dict: Dictionary with feature categories and their features
    """
    features = {
        'returns': ['1M', '3M', '6M', '9M', '1Y', '2Y', '3Y', '4Y', '5Y'],
        'moving_averages': ['sma_5', 'sma_20', 'sma_40', 'sma_120', 'sma_200'],
        'exponential_ma': ['ema_5', 'ema_20', 'ema_40', 'ema_120', 'ema_200'],
        'standard_deviations': ['std_Div_5', 'std_Div_20', 'std_Div_40', 'std_Div_120', 'std_Div_200'],
        'bollinger_bands': ['bollinger_Band_5_2STD', 'bollinger_Band_20_2STD', 'bollinger_Band_40_2STD', 
                           'bollinger_Band_120_2STD', 'bollinger_Band_200_2STD'],
        'valuation_ratios': ['p_s', 'p_e', 'p_b', 'p_fcf'],
        'momentum': ['momentum'],
        'technical_indicators': ['RSI_14', 'macd', 'macd_histogram', 'macd_signal', 'ATR_14'],
        'volume_indicators': ['volume_sma_20', 'volume_ema_20', 'volume_ratio', 'vwap', 'obv'],
        'volatility_indicators': ['volatility_5d', 'volatility_20d', 'volatility_60d']
    }
    
    return features


def extract_handled_features_from_ml_builder():
    """
    Extracts features handled in predict_future_price_changes function
    
    Returns:
        set: Set of features handled in the function
    """
    ml_builder_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'ml_builder.py'
    )
    
    with open(ml_builder_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the predict_future_price_changes function
    func_match = re.search(
        r'def predict_future_price_changes\(.*?\):(.*?)(?=\ndef\s|\nclass\s|\Z)',
        content,
        re.DOTALL
    )
    
    if not func_match:
        raise ValueError("Could not find predict_future_price_changes function")
    
    func_content = func_match.group(1)
    
    # Extract features from if/elif statements
    handled_features = set()
    
    # Pattern 1: if feature == "feature_name"
    pattern1 = re.finditer(r'if\s+feature\s*==\s*["\']([^"\']+)["\']', func_content)
    for match in pattern1:
        handled_features.add(match.group(1))
    
    # Pattern 2: elif feature == "feature_name"
    pattern2 = re.finditer(r'elif\s+feature\s*==\s*["\']([^"\']+)["\']', func_content)
    for match in pattern2:
        handled_features.add(match.group(1))
    
    # Pattern 3: future_df["feature_name"] assignments
    pattern3 = re.finditer(r'future_df\[["\']([^"\']+)["\']\]', func_content)
    for match in pattern3:
        feature = match.group(1)
        # Only add if it's a technical feature (not date, close_Price, etc.)
        if feature not in ['date', 'close_Price', '1D', 'open_Price']:
            handled_features.add(feature)
    
    return handled_features


def test_feature_coverage():
    """
    Main test function to validate feature coverage
    """
    print("="*80)
    print("FEATURE COVERAGE TEST")
    print("="*80)
    print()
    
    # Get all features from stock_data_fetch
    all_features = extract_features_from_stock_data_fetch()
    
    # Flatten all features into a single set
    expected_features = set()
    for category, features in all_features.items():
        expected_features.update(features)
    
    print(f"📊 Total features created in stock_data_fetch.py: {len(expected_features)}")
    print()
    
    # Get handled features from ml_builder
    handled_features = extract_handled_features_from_ml_builder()
    
    print(f"✓ Features handled in predict_future_price_changes: {len(handled_features)}")
    print()
    
    # Find missing features
    missing_features = expected_features - handled_features
    
    # Find extra features (handled but not in expected)
    extra_features = handled_features - expected_features
    
    # Results
    print("-"*80)
    print("ANALYSIS RESULTS")
    print("-"*80)
    print()
    
    if missing_features:
        print("❌ MISSING FEATURES (created but NOT handled in predictions):")
        print()
        
        # Group missing features by category
        for category, features in all_features.items():
            category_missing = [f for f in features if f in missing_features]
            if category_missing:
                print(f"  {category.upper().replace('_', ' ')}:")
                for feature in sorted(category_missing):
                    print(f"    - {feature}")
                print()
        
        print(f"  Total missing: {len(missing_features)}")
        print()
    else:
        print("✅ All features are properly handled!")
        print()
    
    if extra_features:
        print("⚠️ EXTRA FEATURES (handled but not in expected list):")
        for feature in sorted(extra_features):
            print(f"  - {feature}")
        print()
        print("  Note: These might be deprecated or fundamental features")
        print()
    
    # Coverage statistics
    coverage_pct = (len(handled_features & expected_features) / len(expected_features)) * 100
    print("-"*80)
    print("COVERAGE STATISTICS")
    print("-"*80)
    print(f"  Expected features:  {len(expected_features)}")
    print(f"  Handled features:   {len(handled_features & expected_features)}")
    print(f"  Missing features:   {len(missing_features)}")
    print(f"  Coverage:           {coverage_pct:.1f}%")
    print()
    
    # Detailed breakdown by category
    print("-"*80)
    print("COVERAGE BY CATEGORY")
    print("-"*80)
    print()
    
    for category, features in all_features.items():
        total = len(features)
        handled = len([f for f in features if f in handled_features])
        missing = total - handled
        pct = (handled / total) * 100 if total > 0 else 0
        
        status = "✅" if missing == 0 else "❌"
        print(f"{status} {category.upper().replace('_', ' ')}: {handled}/{total} ({pct:.0f}%)")
        
        if missing > 0:
            missing_list = [f for f in features if f not in handled_features]
            print(f"   Missing: {', '.join(missing_list)}")
        print()
    
    # Return test result
    print("="*80)
    if missing_features:
        print("❌ TEST FAILED: Some features are not handled in predictions")
        print()
        print("RECOMMENDED FIXES:")
        print("="*80)
        print()
        print("Add the following code blocks to predict_future_price_changes function")
        print("in the 'for feature in short_term_dynamic_list:' loop:")
        print()
        
        for feature in sorted(missing_features):
            # Generate example code based on feature type
            if 'std_Div_' in feature:
                period = feature.split('_')[-1]
                print(f'    elif feature == "{feature}":')
                print(f'        future_df["{feature}"] = stock_mod_df.iloc[-{period}:]["close_Price"].std()')
                print()
            elif 'bollinger_Band_' in feature:
                parts = feature.split('_')
                period = parts[2]
                print(f'    elif feature == "{feature}":')
                print(f'        std_div_{period} = stock_mod_df.iloc[-{period}:]["close_Price"].std()')
                print(f'        sma_{period} = stock_mod_df.iloc[-{period}:]["close_Price"].mean()')
                print(f'        bollinger_Band_{period}_Upper = sma_{period} + (std_div_{period} * 2)')
                print(f'        bollinger_Band_{period}_Lower = sma_{period} - (std_div_{period} * 2)')
                print(f'        future_df["{feature}"] = bollinger_Band_{period}_Upper - bollinger_Band_{period}_Lower')
                print()
            elif 'ema_' in feature:
                period = feature.split('_')[-1]
                print(f'    elif feature == "{feature}":')
                print(f'        future_df["{feature}"] = stock_mod_df.iloc[-{period}:]["close_Price"].ewm(span={period}, adjust=False).mean().iloc[-1]')
                print()
            elif 'sma_' in feature:
                period = feature.split('_')[-1]
                print(f'    elif feature == "{feature}":')
                print(f'        future_df["{feature}"] = stock_mod_df.iloc[-{period}:]["close_Price"].mean()')
                print()
        
        print()
        print("="*80)
        return False
    else:
        print("✅ TEST PASSED: All features are properly handled")
        print("="*80)
        return True


def test_short_term_dynamic_list_completeness():
    """
    Test that short_term_dynamic_list in ml_builder.py includes all dynamic features
    """
    print()
    print("="*80)
    print("SHORT_TERM_DYNAMIC_LIST COMPLETENESS TEST")
    print("="*80)
    print()
    
    ml_builder_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'ml_builder.py'
    )
    
    with open(ml_builder_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find short_term_dynamic_list definition
    list_match = re.search(
        r'short_term_dynamic_list\s*=\s*\[(.*?)\]',
        content,
        re.DOTALL
    )
    
    if not list_match:
        print("❌ Could not find short_term_dynamic_list definition")
        return False
    
    list_content = list_match.group(1)
    
    # Extract features from the list
    features_in_list = set()
    pattern = re.finditer(r'["\']([^"\']+)["\']', list_content)
    for match in pattern:
        feature = match.group(1)
        # Skip comments
        if '#' not in feature:
            features_in_list.add(feature)
    
    # Get all dynamic features
    all_features = extract_features_from_stock_data_fetch()
    
    # Fundamental features (static, not in dynamic list)
    fundamental_features = {
        'revenue', 'eps', 'book_Value_Per_Share', 'free_Cash_Flow_Per_Share',
        'average_shares', 'operating_Cash_Flow', 'capital_Expenditure'
    }
    
    # Expected dynamic features (all except fundamentals)
    expected_dynamic = set()
    for category, features in all_features.items():
        # Skip if these are fundamental
        if category != 'fundamental':
            expected_dynamic.update(features)
    
    # Find missing from dynamic list
    missing_from_list = expected_dynamic - features_in_list
    
    print(f"📋 Features in short_term_dynamic_list: {len(features_in_list)}")
    print(f"📊 Expected dynamic features: {len(expected_dynamic)}")
    print()
    
    if missing_from_list:
        print("❌ MISSING FROM short_term_dynamic_list:")
        for feature in sorted(missing_from_list):
            print(f"  - {feature}")
        print()
        print("These features should be added to short_term_dynamic_list")
        print("="*80)
        return False
    else:
        print("✅ short_term_dynamic_list is complete!")
        print("="*80)
        return True


if __name__ == "__main__":
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "  FEATURE COVERAGE VALIDATION TEST SUITE".center(78) + "║")
    print("║" + "  Ensures all features are handled in predictions".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    print("\n")
    
    # Run tests
    test1_passed = test_feature_coverage()
    test2_passed = test_short_term_dynamic_list_completeness()
    
    # Summary
    print()
    print("="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    if test1_passed and test2_passed:
        print("✅ ALL TESTS PASSED")
        print("   All features are properly handled in predictions")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        if not test1_passed:
            print("   - Feature coverage test failed: some features missing")
        if not test2_passed:
            print("   - short_term_dynamic_list completeness test failed")
        print()
        print("Please review the recommended fixes above and update ml_builder.py")
        sys.exit(1)
