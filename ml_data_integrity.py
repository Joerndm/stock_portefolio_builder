"""
ML Pipeline Data Integrity Module

This module ensures data integrity for the ML pipeline.

Features:
    - Validates data consistency for ML training
    - Tracks data source metadata through the ML pipeline
    - Ensures ratio columns are properly populated
    - Validates feature availability before model training

The module integrates with:
    - db_interactions.py: For data import
    - stock_data_fetch.py: For ratio calculations with TTM/annual fallback
    - ml_builder.py: For model training with source-aware features

Data Flow:
    1. import_stock_dataset() fetches combined stock data with ratios
    2. validate_ml_features() checks all required features are present
    3. handle_missing_ratio_data() provides fallback for missing ratios
    4. track_data_source() adds metadata for pipeline monitoring

Note:
    TTM ratio calculations happen at data fetch time via calculate_ratios_ttm_with_fallback()
    and are stored in stock_ratio_data table. No separate TTM table is needed.

Author: Stock Portfolio Builder
Last Modified: 2026
"""
import os
import sys
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np

# Add project root to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import db_interactions
from ttm_financial_calculator import TTMFinancialCalculator


class MLDataIntegrityChecker:
    """
    Validates and ensures data integrity for ML pipeline training.
    """
    
    # Required features for basic ML models
    REQUIRED_PRICE_FEATURES = [
        'date', 'ticker', 'close_Price', 'open_Price', 'high_Price', 'low_Price',
        'trade_Volume', '1D'  # 1D return is the target variable
    ]
    
    # Technical indicator features
    TECHNICAL_FEATURES = [
        'RSI_14', 'macd', 'macd_histogram', 'macd_signal', 'ATR_14',
        'sma_5', 'sma_20', 'sma_40', 'sma_120', 'sma_200',
        'ema_5', 'ema_20', 'ema_40', 'ema_120', 'ema_200',
        'std_Div_5', 'std_Div_20', 'std_Div_40', 'std_Div_120', 'std_Div_200',
        'bollinger_Band_5_2STD', 'bollinger_Band_20_2STD', 
        'bollinger_Band_40_2STD', 'bollinger_Band_120_2STD', 'bollinger_Band_200_2STD',
        'momentum'
    ]
    
    # Volume-based features
    VOLUME_FEATURES = [
        'volume_sma_20', 'volume_ema_20', 'volume_ratio', 'vwap', 'obv'
    ]
    
    # Volatility features
    VOLATILITY_FEATURES = [
        'volatility_5d', 'volatility_20d', 'volatility_60d'
    ]
    
    # Fundamental financial features (forward-filled from financial statements)
    FUNDAMENTAL_FEATURES = [
        'revenue', 'eps', 'book_Value_Per_Share', 'free_Cash_Flow_Per_Share',
        'average_shares'
    ]
    
    # Ratio features (calculated from financial data + price)
    RATIO_FEATURES = ['p_s', 'p_e', 'p_b', 'p_fcf']
    
    # Period return features
    RETURN_FEATURES = ['1M', '3M', '6M', '9M', '1Y', '2Y', '3Y', '4Y', '5Y']
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the data integrity checker.
        
        Args:
            verbose: Whether to print detailed messages
        """
        self.verbose = verbose
        self.ttm_calculator = TTMFinancialCalculator()
        
    def _log(self, message: str):
        """Print message if verbose mode enabled."""
        if self.verbose:
            print(message)
    
    def get_all_expected_features(self) -> List[str]:
        """Get complete list of expected ML features."""
        return (
            self.REQUIRED_PRICE_FEATURES +
            self.TECHNICAL_FEATURES +
            self.VOLUME_FEATURES +
            self.VOLATILITY_FEATURES +
            self.FUNDAMENTAL_FEATURES +
            self.RATIO_FEATURES +
            self.RETURN_FEATURES
        )
    
    def validate_ml_features(
        self, 
        df: pd.DataFrame, 
        required_only: bool = False
    ) -> Dict[str, Any]:
        """
        Validate that required features exist in the DataFrame.
        
        Args:
            df: DataFrame to validate
            required_only: If True, only check critical features
            
        Returns:
            Validation result dictionary
        """
        result = {
            'valid': True,
            'missing_critical': [],
            'missing_optional': [],
            'present_features': [],
            'data_quality': {}
        }
        
        if df.empty:
            result['valid'] = False
            result['missing_critical'] = ['DataFrame is empty']
            return result
        
        all_features = self.get_all_expected_features()
        critical_features = self.REQUIRED_PRICE_FEATURES + self.RATIO_FEATURES
        
        # Check each feature
        for feature in all_features:
            if feature in df.columns:
                result['present_features'].append(feature)
                
                # Check data quality
                null_pct = df[feature].isnull().mean()
                inf_pct = np.isinf(df[feature].replace([np.inf, -np.inf], np.nan).dropna()).mean() if df[feature].dtype in ['float64', 'float32', 'int64'] else 0
                
                result['data_quality'][feature] = {
                    'null_pct': null_pct,
                    'inf_pct': inf_pct,
                    'dtype': str(df[feature].dtype)
                }
            else:
                if feature in critical_features:
                    result['missing_critical'].append(feature)
                else:
                    result['missing_optional'].append(feature)
        
        # Set validity based on critical features
        if result['missing_critical']:
            result['valid'] = False
            
        return result
    
    def handle_missing_ratio_data(
        self, 
        df: pd.DataFrame, 
        ticker: str,
        prefer_ttm: bool = True
    ) -> pd.DataFrame:
        """
        Handle missing ratio data by calculating from available sources.
        
        This method attempts to fill in missing ratio columns (p_s, p_e, p_b, p_fcf)
        using either TTM or annual financial data.
        
        Args:
            df: DataFrame with potential missing ratios
            ticker: Stock ticker symbol
            prefer_ttm: Whether to prefer TTM calculations
            
        Returns:
            DataFrame with ratio data filled where possible
        """
        result = df.copy()
        
        # Check which ratios are missing
        missing_ratios = [r for r in self.RATIO_FEATURES if r not in result.columns or result[r].isnull().all()]
        
        if not missing_ratios:
            self._log(f"✓ All ratio features present for {ticker}")
            return result
        
        self._log(f"⚠️ Missing ratio features for {ticker}: {missing_ratios}")
        self._log(f"  Attempting to calculate from {'TTM' if prefer_ttm else 'annual'} data...")
        
        try:
            # Get price data from DataFrame
            price_data = result[['date', 'close_Price']].copy()
            price_data['ticker'] = ticker
            
            # Calculate ratios using TTM calculator
            ratios_df = self.ttm_calculator.calculate_ratios_with_source_tracking(
                ticker, price_data, prefer_ttm=prefer_ttm
            )
            
            if not ratios_df.empty:
                # Map calculated ratios to expected column names
                ratio_mapping = {
                    'P/S': 'p_s',
                    'P/E': 'p_e',
                    'P/B': 'p_b',
                    'P/FCF': 'p_fcf'
                }
                
                for calc_col, db_col in ratio_mapping.items():
                    if calc_col in ratios_df.columns and (db_col not in result.columns or result[db_col].isnull().all()):
                        # Merge on date
                        ratios_df['date'] = pd.to_datetime(ratios_df['date'])
                        result['date'] = pd.to_datetime(result['date'])
                        
                        merge_df = ratios_df[['date', calc_col]].rename(columns={calc_col: f'{db_col}_new'})
                        result = result.merge(merge_df, on='date', how='left')
                        
                        # Fill missing values
                        if db_col not in result.columns:
                            result[db_col] = result[f'{db_col}_new']
                        else:
                            result[db_col] = result[db_col].fillna(result[f'{db_col}_new'])
                        
                        result = result.drop(columns=[f'{db_col}_new'])
                
                # Add source tracking if not present
                if 'ratio_data_source' in ratios_df.columns:
                    source = ratios_df['ratio_data_source'].iloc[0]
                    result['ratio_data_source'] = source
                    self._log(f"✓ Ratios filled from {source} data")
                    
        except Exception as e:
            self._log(f"❌ Could not calculate ratios: {e}")
            # Fill with NaN rather than failing
            for ratio in missing_ratios:
                if ratio not in result.columns:
                    result[ratio] = np.nan
        
        return result
    
    def prepare_ml_dataset(
        self,
        ticker: str,
        prefer_ttm: bool = True,
        min_rows: int = 252  # Minimum 1 year of trading days
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Prepare a complete ML-ready dataset with data integrity checks.
        
        This method:
        1. Imports data using TTM preference
        2. Validates all required features
        3. Fills missing ratio data if needed
        4. Returns metadata about data sources
        
        Args:
            ticker: Stock ticker symbol
            prefer_ttm: Whether to prefer TTM ratios
            min_rows: Minimum required rows for training
            
        Returns:
            Tuple of (cleaned DataFrame, metadata dictionary)
        """
        metadata = {
            'ticker': ticker,
            'data_source': 'unknown',
            'ratio_source': 'unknown',
            'rows_original': 0,
            'rows_final': 0,
            'features_used': [],
            'features_missing': [],
            'warnings': [],
            'validation': None
        }
        
        try:
            # Import data
            self._log(f"\n📊 Preparing ML dataset for {ticker}...")
            
            try:
                df = db_interactions.import_stock_dataset(stock_ticker=ticker)
                metadata['data_source'] = 'import_stock_dataset'
            except Exception as e:
                self._log(f"  Import failed: {e}")
                metadata['warnings'].append(f'Import failed: {e}')
                return pd.DataFrame(), metadata
            
            if df.empty:
                metadata['warnings'].append('Empty dataset returned from database')
                return pd.DataFrame(), metadata
            
            metadata['rows_original'] = len(df)
            
            # Track ratio source
            if 'ratio_source' in df.columns:
                metadata['ratio_source'] = df['ratio_source'].iloc[0]
            elif 'ratio_data_source' in df.columns:
                metadata['ratio_source'] = df['ratio_data_source'].iloc[0]
            else:
                metadata['ratio_source'] = 'unknown'
            
            # Validate features
            validation = self.validate_ml_features(df)
            metadata['validation'] = validation
            
            if validation['missing_critical']:
                self._log(f"⚠️ Missing critical features: {validation['missing_critical']}")
                
                # Try to fill ratio features
                df = self.handle_missing_ratio_data(df, ticker, prefer_ttm)
                
                # Re-validate
                validation = self.validate_ml_features(df)
                metadata['validation'] = validation
            
            # Convert date column
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            # Drop rows with NaN in critical columns
            critical_cols = [c for c in self.REQUIRED_PRICE_FEATURES if c in df.columns]
            df = df.dropna(subset=critical_cols)
            
            metadata['rows_final'] = len(df)
            metadata['features_used'] = validation['present_features']
            metadata['features_missing'] = validation['missing_critical'] + validation['missing_optional']
            
            # Check minimum rows
            if len(df) < min_rows:
                metadata['warnings'].append(
                    f"Insufficient data: {len(df)} rows (need {min_rows})"
                )
            
            self._log(f"✓ Dataset prepared: {len(df)} rows, {len(validation['present_features'])} features")
            self._log(f"  Ratio data source: {metadata['ratio_source']}")
            
            return df, metadata
            
        except Exception as e:
            metadata['warnings'].append(f"Error preparing dataset: {e}")
            self._log(f"❌ Error preparing dataset: {e}")
            return pd.DataFrame(), metadata
    
    def validate_data_consistency(
        self,
        ticker: str,
        df: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """
        Validate that ratio data exists for the stock.
        
        Args:
            ticker: Stock ticker symbol
            df: Optional DataFrame to validate (imports if not provided)
            
        Returns:
            Consistency validation results
        """
        result = {
            'ticker': ticker,
            'consistent': True,
            'ratio_data_available': False,
            'recommendations': []
        }
        
        try:
            # Check ratio data availability
            result['ratio_data_available'] = db_interactions.does_stock_exists_stock_ratio_data(ticker)
            
            if not result['ratio_data_available']:
                result['consistent'] = False
                result['recommendations'].append('No ratio data available - calculate from financial statements')
                return result
            
            # Verify we can import ratio data
            ratio_df = db_interactions.import_stock_ratio_data(stock_ticker=ticker)
            
            if ratio_df.empty:
                result['consistent'] = False
                result['recommendations'].append('Ratio data exists but is empty')
            else:
                # Check for required ratio columns
                required_ratios = ['p_e', 'p_s', 'p_b', 'p_fcf']
                missing_ratios = [r for r in required_ratios if r not in ratio_df.columns]
                
                if missing_ratios:
                    result['recommendations'].append(f'Missing ratio columns: {missing_ratios}')
                
                # Check for excessive NaN values
                for ratio in required_ratios:
                    if ratio in ratio_df.columns:
                        nan_pct = ratio_df[ratio].isna().sum() / len(ratio_df)
                        if nan_pct > 0.5:
                            result['recommendations'].append(
                                f'{ratio.upper()} has {nan_pct:.1%} NaN values'
                            )
                
        except Exception as e:
            result['consistent'] = False
            result['recommendations'].append(f'Error during validation: {e}')
        
        return result


def prepare_ml_dataset_with_integrity_check(
    ticker: str,
    prefer_ttm: bool = True,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function to prepare ML dataset with full integrity checks.
    
    This is the recommended entry point for ML pipeline data preparation.
    
    Args:
        ticker: Stock ticker symbol
        prefer_ttm: Whether to prefer TTM ratios (default: True)
        verbose: Whether to print progress messages
        
    Returns:
        Tuple of (DataFrame, metadata dictionary)
        
    Example:
        >>> df, meta = prepare_ml_dataset_with_integrity_check('AAPL')
        >>> print(f"Data source: {meta['ratio_source']}")
        >>> print(f"Rows: {meta['rows_final']}")
    """
    checker = MLDataIntegrityChecker(verbose=verbose)
    return checker.prepare_ml_dataset(ticker, prefer_ttm=prefer_ttm)


def validate_ratio_data_consistency(ticker: str) -> Dict[str, Any]:
    """
    Validate consistency between TTM and annual ratio data for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Consistency validation results
    """
    checker = MLDataIntegrityChecker(verbose=False)
    return checker.validate_data_consistency(ticker)


# Demo/testing
if __name__ == "__main__":
    print("Testing ML Data Integrity Checker")
    print("="*60)
    
    # Test with a sample ticker
    ticker = "DEMANT.CO"
    
    checker = MLDataIntegrityChecker(verbose=True)
    
    # Test 1: Prepare dataset
    print("\n1. Preparing ML dataset...")
    df, metadata = checker.prepare_ml_dataset(ticker, prefer_ttm=True)
    
    print(f"\nMetadata:")
    print(f"  Ticker: {metadata['ticker']}")
    print(f"  Data source: {metadata['data_source']}")
    print(f"  Ratio source: {metadata['ratio_source']}")
    print(f"  Rows: {metadata['rows_original']} → {metadata['rows_final']}")
    print(f"  Features: {len(metadata['features_used'])} present, {len(metadata['features_missing'])} missing")
    
    if metadata['warnings']:
        print(f"  Warnings: {metadata['warnings']}")
    
    # Test 2: Validate consistency
    print("\n2. Validating data consistency...")
    consistency = checker.validate_data_consistency(ticker)
    
    print(f"  TTM available: {consistency['ttm_available']}")
    print(f"  Annual available: {consistency['annual_available']}")
    print(f"  Consistent: {consistency['consistent']}")
    
    if consistency['recommendations']:
        print(f"  Recommendations:")
        for rec in consistency['recommendations']:
            print(f"    - {rec}")
    
    # Test 3: Feature validation
    if not df.empty:
        print("\n3. Feature validation details...")
        validation = checker.validate_ml_features(df)
        
        # Show data quality for ratio features
        print("  Ratio feature quality:")
        for ratio in checker.RATIO_FEATURES:
            if ratio in validation['data_quality']:
                quality = validation['data_quality'][ratio]
                print(f"    {ratio}: {quality['null_pct']:.1%} null, dtype={quality['dtype']}")
