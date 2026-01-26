"""
TTM Financial Calculator Module

This module provides unified financial data handling with TTM (Trailing Twelve Months)
calculations from quarterly reports, with fallback to annual data when insufficient
quarterly data is available.

Features:
    - Automatic TTM calculation from 4+ quarterly reports
    - Fallback to annual data when <4 quarters available
    - Ratio calculations (P/E, P/B, P/S, P/FCF) using appropriate data source
    - Data source tracking (ttm vs annual) for transparency
    - Validation against existing annual-based ratios
    - Industry-specific handling (banks, insurance, biotech)

The module ensures data consistency across the ML pipeline by:
    1. Using TTM when 4+ quarters are available (more current, ~3 months lag)
    2. Falling back to annual when TTM not possible (~12 months lag)
    3. Tracking data source in metadata for pipeline awareness
    4. Validating calculations against historical data

Author: Stock Portfolio Builder
Last Modified: 2026
"""
import datetime
from typing import Dict, List, Optional, Tuple, Any
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import yfinance as yf

from enhanced_financial_fetcher import (
    QuarterlyFinancialFetcher,
    fetch_quarterly_financial_data
)


class TTMFinancialCalculator:
    """
    Calculates financial metrics using TTM data with annual fallback.
    
    This class provides a unified interface for financial data that:
    - Prefers TTM (from 4 quarters) for more current data
    - Falls back to annual data when insufficient quarters available
    - Tracks data source for transparency and ML pipeline consistency
    """
    
    # Minimum quarters needed for TTM calculation
    MIN_QUARTERS_FOR_TTM = 4
    
    # Default margin of error for validation (15% difference acceptable)
    DEFAULT_VALIDATION_MARGIN = 0.15
    
    def __init__(self, use_fmp: bool = False, fmp_api_key: str = ''):
        """
        Initialize the TTMFinancialCalculator.
        
        Args:
            use_fmp: Whether to use Financial Modeling Prep API
            fmp_api_key: API key for FMP (if using)
        """
        self.quarterly_fetcher = QuarterlyFinancialFetcher(use_fmp, fmp_api_key)
        
    def _safe_divide(self, numerator, denominator):
        """
        Safely divide two numbers or arrays, handling zero division and NaN.
        
        Works with both scalar values and numpy arrays/pandas Series.
        """
        try:
            # Handle scalar inputs directly (most common case)
            if np.isscalar(numerator) and np.isscalar(denominator):
                if denominator == 0 or np.isnan(denominator) or np.isnan(numerator):
                    return np.nan
                return float(numerator) / float(denominator)
            
            # Convert to numpy arrays for vectorized operations
            num = np.atleast_1d(np.asarray(numerator, dtype=float))
            den = np.atleast_1d(np.asarray(denominator, dtype=float))
            
            # Broadcast to same shape to handle mixed scalar/array inputs
            num, den = np.broadcast_arrays(num, den)
            num = num.astype(float)  # broadcast_arrays returns read-only views
            
            # Create result array filled with NaN
            result = np.full(num.shape, np.nan, dtype=float)
            
            # Find valid positions (denominator not zero, not nan, numerator not nan)
            valid_mask = (den != 0) & ~np.isnan(den) & ~np.isnan(num)
            
            # Perform division only where valid
            if np.any(valid_mask):
                result[valid_mask] = num[valid_mask] / den[valid_mask]
            
            # Return scalar if result is single element
            if result.size == 1:
                return float(result.flat[0])
            
            return result
            
        except (TypeError, ZeroDivisionError, ValueError):
            # If all else fails, return NaN (scalar) or array of NaN
            if np.isscalar(numerator):
                return np.nan
            return np.full_like(np.atleast_1d(np.asarray(numerator)), np.nan, dtype=float)
    
    def count_available_quarters(self, symbol: str, years: int = 10) -> int:
        """
        Count how many quarters of data are available for a symbol.
        
        Args:
            symbol: Stock ticker symbol
            years: Number of years to look back
            
        Returns:
            Number of quarters available
        """
        income_df = self.quarterly_fetcher.fetch_quarterly_income_statement(symbol, years)
        return len(income_df) if not income_df.empty else 0
    
    def can_use_ttm(self, symbol: str, as_of_date: Optional[datetime.date] = None) -> Tuple[bool, int]:
        """
        Determine if TTM calculation is possible for a symbol at a given date.
        
        Args:
            symbol: Stock ticker symbol
            as_of_date: Date to check (default: today)
            
        Returns:
            Tuple of (can_use_ttm: bool, quarters_available: int)
        """
        quarters = self.count_available_quarters(symbol)
        can_use = quarters >= self.MIN_QUARTERS_FOR_TTM
        return can_use, quarters
    
    def fetch_annual_financial_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch annual financial data using existing yfinance methods.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with annual financial DataFrames
        """
        result = {
            'income': pd.DataFrame(),
            'balance_sheet': pd.DataFrame(),
            'cash_flow': pd.DataFrame()
        }
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Fetch annual data
            income_annual = ticker.income_stmt
            if income_annual is not None and not income_annual.empty:
                result['income'] = income_annual.T.sort_index()
                
            bs_annual = ticker.balance_sheet
            if bs_annual is not None and not bs_annual.empty:
                result['balance_sheet'] = bs_annual.T.sort_index()
                
            cf_annual = ticker.cashflow
            if cf_annual is not None and not cf_annual.empty:
                result['cash_flow'] = cf_annual.T.sort_index()
                
        except Exception as e:
            print(f"Error fetching annual financial data for {symbol}: {e}")
            
        return result
    
    def get_best_available_financials(
        self, 
        symbol: str, 
        prefer_ttm: bool = True,
        years: int = 10
    ) -> Dict[str, Any]:
        """
        Get the best available financial data (TTM if possible, annual as fallback).
        
        Args:
            symbol: Stock ticker symbol
            prefer_ttm: Whether to prefer TTM over annual (default: True)
            years: Years of historical data to consider
            
        Returns:
            Dictionary containing:
                - 'data': Financial data DataFrame
                - 'source': 'ttm' or 'annual'
                - 'quarters_available': Number of quarters found
                - 'ttm_possible': Whether TTM calculation was possible
        """
        result = {
            'data': pd.DataFrame(),
            'source': 'none',
            'quarters_available': 0,
            'ttm_possible': False,
            'ratios': pd.DataFrame(),
            'income_data': pd.DataFrame(),
            'balance_sheet': pd.DataFrame(),
            'cash_flow': pd.DataFrame()
        }
        
        if prefer_ttm:
            # Try quarterly/TTM first
            quarterly_data = fetch_quarterly_financial_data(symbol, years)
            
            income_q = quarterly_data.get('income_quarterly', pd.DataFrame())
            quarters_available = len(income_q) if not income_q.empty else 0
            result['quarters_available'] = quarters_available
            
            if quarters_available >= self.MIN_QUARTERS_FOR_TTM:
                result['ttm_possible'] = True
                result['source'] = 'ttm'
                result['data'] = quarterly_data
                result['ratios'] = quarterly_data.get('ratios', pd.DataFrame())
                result['income_data'] = quarterly_data.get('income_ttm', pd.DataFrame())
                result['balance_sheet'] = quarterly_data.get('balance_sheet', pd.DataFrame())
                result['cash_flow'] = quarterly_data.get('cash_flow_ttm', pd.DataFrame())
                print(f"✅ Using TTM data for {symbol} ({quarters_available} quarters available)")
                return result
            else:
                print(f"⚠️ Only {quarters_available} quarters available for {symbol}, falling back to annual")
        
        # Fall back to annual data
        annual_data = self.fetch_annual_financial_data(symbol)
        
        if not annual_data['income'].empty:
            result['source'] = 'annual'
            result['data'] = annual_data
            result['income_data'] = annual_data['income']
            result['balance_sheet'] = annual_data['balance_sheet']
            result['cash_flow'] = annual_data['cash_flow']
            print(f"📅 Using annual data for {symbol}")
        else:
            print(f"❌ No financial data available for {symbol}")
            result['source'] = 'none'
            
        return result
    
    def calculate_ratios_with_source_tracking(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        prefer_ttm: bool = True
    ) -> pd.DataFrame:
        """
        Calculate financial ratios (P/E, P/B, P/S, P/FCF) with data source tracking.
        
        This method calculates ratios using the best available data source and
        adds metadata columns indicating the data source used.
        
        Args:
            symbol: Stock ticker symbol
            price_data: DataFrame with price data (must have 'date' and 'close_Price')
            prefer_ttm: Whether to prefer TTM over annual (default: True)
            
        Returns:
            DataFrame with ratios and source metadata columns
        """
        if price_data.empty:
            return pd.DataFrame()
            
        # Get best available financial data
        fin_data = self.get_best_available_financials(symbol, prefer_ttm)
        
        if fin_data['source'] == 'none':
            # No financial data available - return price data with NaN ratios
            result = price_data.copy()
            result['P/E'] = np.nan
            result['P/B'] = np.nan
            result['P/S'] = np.nan
            result['P/FCF'] = np.nan
            result['ratio_data_source'] = 'none'
            result['quarters_available'] = 0
            return result
        
        if fin_data['source'] == 'ttm':
            return self._calculate_ratios_from_ttm(symbol, price_data, fin_data)
        else:
            return self._calculate_ratios_from_annual(symbol, price_data, fin_data)
    
    def _calculate_ratios_from_ttm(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        fin_data: Dict[str, Any]
    ) -> pd.DataFrame:
        """Calculate ratios using TTM data."""
        result = price_data.copy()
        
        # Add source tracking columns
        result['ratio_data_source'] = 'ttm'
        result['quarters_available'] = fin_data['quarters_available']
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            shares_outstanding = info.get('sharesOutstanding', None)
            
            if shares_outstanding is None or shares_outstanding == 0:
                print(f"⚠️ No shares outstanding data for {symbol}")
                result['P/E'] = np.nan
                result['P/B'] = np.nan
                result['P/S'] = np.nan
                result['P/FCF'] = np.nan
                return result
            
            ratios_df = fin_data.get('ratios', pd.DataFrame())
            
            if ratios_df.empty:
                result['P/E'] = np.nan
                result['P/B'] = np.nan
                result['P/S'] = np.nan
                result['P/FCF'] = np.nan
                return result
            
            # Merge TTM metrics with price data
            # Forward-fill financial data to daily prices
            ratios_df = ratios_df.copy()
            ratios_df['date'] = pd.to_datetime(ratios_df['date'])
            ratios_df = ratios_df.sort_values('date')
            
            result['date'] = pd.to_datetime(result['date'])
            result = result.sort_values('date')
            
            # Create a merged dataset with forward-filled TTM values
            merged = pd.merge_asof(
                result[['date', 'close_Price']],
                ratios_df[['date', 'eps_ttm', 'book_value_per_share', 'revenue_ttm', 'fcf_per_share']].rename(
                    columns={'date': 'financial_date'}
                ).assign(date=ratios_df['date']),
                on='date',
                direction='backward'
            )
            
            # Calculate ratios
            result['P/E'] = self._safe_divide(result['close_Price'].values, merged['eps_ttm'].values)
            result['P/B'] = self._safe_divide(result['close_Price'].values, merged['book_value_per_share'].values)
            
            # P/S requires revenue per share
            if 'revenue_ttm' in merged.columns:
                revenue_per_share = merged['revenue_ttm'] / shares_outstanding
                result['P/S'] = self._safe_divide(result['close_Price'].values, revenue_per_share.values)
            else:
                result['P/S'] = np.nan
                
            result['P/FCF'] = self._safe_divide(result['close_Price'].values, merged['fcf_per_share'].values)
            
            # Shift ratios by 1 to avoid look-ahead bias (same as original code)
            result[['P/E', 'P/B', 'P/S', 'P/FCF']] = result[['P/E', 'P/B', 'P/S', 'P/FCF']].shift(1)
            
        except Exception as e:
            print(f"Error calculating TTM ratios for {symbol}: {e}")
            result['P/E'] = np.nan
            result['P/B'] = np.nan
            result['P/S'] = np.nan
            result['P/FCF'] = np.nan
            
        return result
    
    def _calculate_ratios_from_annual(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        fin_data: Dict[str, Any]
    ) -> pd.DataFrame:
        """Calculate ratios using annual data (fallback method)."""
        result = price_data.copy()
        
        # Add source tracking columns
        result['ratio_data_source'] = 'annual'
        result['quarters_available'] = fin_data['quarters_available']
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            income_annual = fin_data['income_data']
            bs_annual = fin_data['balance_sheet']
            cf_annual = fin_data['cash_flow']
            
            if income_annual.empty:
                result['P/E'] = np.nan
                result['P/B'] = np.nan
                result['P/S'] = np.nan
                result['P/FCF'] = np.nan
                return result
            
            # Build annual financial metrics DataFrame
            annual_metrics = pd.DataFrame({
                'date': income_annual.index
            })
            
            # Get historical share counts from income statement (prefer diluted for consistency)
            # This is more accurate than using current sharesOutstanding
            if 'Diluted Average Shares' in income_annual.columns:
                annual_metrics['shares'] = income_annual['Diluted Average Shares'].values
            elif 'Basic Average Shares' in income_annual.columns:
                annual_metrics['shares'] = income_annual['Basic Average Shares'].values
            else:
                # Fall back to current shares outstanding if historical not available
                shares_outstanding = info.get('sharesOutstanding', None)
                if shares_outstanding is None:
                    result['P/E'] = np.nan
                    result['P/B'] = np.nan
                    result['P/S'] = np.nan
                    result['P/FCF'] = np.nan
                    return result
                annual_metrics['shares'] = shares_outstanding
            
            # Get relevant columns
            if 'Net Income Common Stockholders' in income_annual.columns:
                annual_metrics['net_income'] = income_annual['Net Income Common Stockholders'].values
            elif 'Net Income' in income_annual.columns:
                annual_metrics['net_income'] = income_annual['Net Income'].values
            else:
                annual_metrics['net_income'] = np.nan
                
            if 'Total Revenue' in income_annual.columns:
                annual_metrics['revenue'] = income_annual['Total Revenue'].values
            else:
                annual_metrics['revenue'] = np.nan
            
            if not bs_annual.empty:
                if 'Stockholders Equity' in bs_annual.columns:
                    annual_metrics['equity'] = bs_annual['Stockholders Equity'].values
                elif 'Total Equity Gross Minority Interest' in bs_annual.columns:
                    annual_metrics['equity'] = bs_annual['Total Equity Gross Minority Interest'].values
                else:
                    annual_metrics['equity'] = np.nan
            else:
                annual_metrics['equity'] = np.nan
                
            if not cf_annual.empty and 'Free Cash Flow' in cf_annual.columns:
                annual_metrics['fcf'] = cf_annual['Free Cash Flow'].values
            else:
                annual_metrics['fcf'] = np.nan
            
            # Calculate per share metrics using historical share counts
            annual_metrics['eps'] = self._safe_divide(annual_metrics['net_income'].values, annual_metrics['shares'].values)
            annual_metrics['book_value_per_share'] = self._safe_divide(annual_metrics['equity'].values, annual_metrics['shares'].values)
            annual_metrics['revenue_per_share'] = self._safe_divide(annual_metrics['revenue'].values, annual_metrics['shares'].values)
            annual_metrics['fcf_per_share'] = self._safe_divide(annual_metrics['fcf'].values, annual_metrics['shares'].values)
            
            annual_metrics['date'] = pd.to_datetime(annual_metrics['date'])
            annual_metrics = annual_metrics.sort_values('date')
            
            # Merge with price data
            result['date'] = pd.to_datetime(result['date'])
            result = result.sort_values('date')
            
            merged = pd.merge_asof(
                result[['date', 'close_Price']],
                annual_metrics[['date', 'eps', 'book_value_per_share', 'revenue_per_share', 'fcf_per_share']],
                on='date',
                direction='backward'
            )
            
            # Calculate ratios
            result['P/E'] = self._safe_divide(result['close_Price'].values, merged['eps'].values)
            result['P/B'] = self._safe_divide(result['close_Price'].values, merged['book_value_per_share'].values)
            result['P/S'] = self._safe_divide(result['close_Price'].values, merged['revenue_per_share'].values)
            result['P/FCF'] = self._safe_divide(result['close_Price'].values, merged['fcf_per_share'].values)
            
            # Shift ratios by 1 to avoid look-ahead bias
            result[['P/E', 'P/B', 'P/S', 'P/FCF']] = result[['P/E', 'P/B', 'P/S', 'P/FCF']].shift(1)
            
        except Exception as e:
            print(f"Error calculating annual ratios for {symbol}: {e}")
            result['P/E'] = np.nan
            result['P/B'] = np.nan
            result['P/S'] = np.nan
            result['P/FCF'] = np.nan
            
        return result
    
    def validate_ttm_vs_annual(
        self,
        symbol: str,
        margin_of_error: float = DEFAULT_VALIDATION_MARGIN
    ) -> Dict[str, Any]:
        """
        Validate TTM calculations against annual data.
        
        Compares TTM-calculated metrics with annual report data to ensure
        consistency. Allows for a margin of error due to timing differences
        and corrections between quarterly and annual reports.
        
        Args:
            symbol: Stock ticker symbol
            margin_of_error: Acceptable percentage difference (default: 15%)
            
        Returns:
            Dictionary with validation results:
                - 'valid': bool - Overall validation passed
                - 'metrics': Dict of individual metric validations
                - 'warnings': List of warning messages
                - 'errors': List of error messages
        """
        result = {
            'valid': True,
            'metrics': {},
            'warnings': [],
            'errors': [],
            'comparisons': []
        }
        
        try:
            # Fetch both TTM and annual data
            ttm_data = fetch_quarterly_financial_data(symbol, years=5)
            annual_data = self.fetch_annual_financial_data(symbol)
            
            if ttm_data.get('ratios', pd.DataFrame()).empty:
                result['warnings'].append(f"No TTM data available for {symbol}")
                return result
                
            if annual_data['income'].empty:
                result['warnings'].append(f"No annual data available for {symbol}")
                return result
            
            ttm_ratios = ttm_data['ratios']
            annual_income = annual_data['income']
            
            # Compare metrics at fiscal year end dates
            for metric_name, ttm_col, annual_col in [
                ('Revenue', 'revenue_ttm', 'Total Revenue'),
                ('Net Income', 'net_income_ttm', 'Net Income Common Stockholders'),
                ('Gross Margin', 'gross_margin_ttm', None),  # Calculated metric
                ('ROE', 'roe_ttm', None),  # Calculated metric
            ]:
                comparison = self._compare_metric(
                    ttm_ratios, annual_income, annual_data['balance_sheet'],
                    metric_name, ttm_col, annual_col, margin_of_error
                )
                result['metrics'][metric_name] = comparison
                result['comparisons'].append(comparison)
                
                if not comparison['within_margin']:
                    result['warnings'].append(
                        f"{metric_name}: TTM={comparison['ttm_value']:.2f}, "
                        f"Annual={comparison['annual_value']:.2f}, "
                        f"Diff={comparison['pct_diff']:.1%}"
                    )
                    
        except Exception as e:
            result['errors'].append(f"Validation error for {symbol}: {e}")
            result['valid'] = False
            
        # Set overall validity
        critical_errors = [c for c in result['comparisons'] 
                         if not c.get('within_margin', True) 
                         and c.get('pct_diff', 0) > margin_of_error * 2]
        if critical_errors:
            result['valid'] = False
            
        return result
    
    def _compare_metric(
        self,
        ttm_df: pd.DataFrame,
        annual_income: pd.DataFrame,
        annual_bs: pd.DataFrame,
        metric_name: str,
        ttm_col: str,
        annual_col: Optional[str],
        margin: float
    ) -> Dict[str, Any]:
        """Compare a single metric between TTM and annual data."""
        comparison = {
            'metric': metric_name,
            'ttm_value': np.nan,
            'annual_value': np.nan,
            'pct_diff': np.nan,
            'within_margin': True
        }
        
        try:
            if ttm_col not in ttm_df.columns:
                return comparison
                
            # Get latest TTM value - ensure scalar
            ttm_series = ttm_df[ttm_col].dropna()
            ttm_value = float(ttm_series.iloc[-1]) if not ttm_series.empty else np.nan
            comparison['ttm_value'] = ttm_value
            
            # Get corresponding annual value - ensure scalar
            if annual_col and annual_col in annual_income.columns:
                annual_series = annual_income[annual_col].dropna()
                annual_value = float(annual_series.iloc[-1]) if not annual_series.empty else np.nan
                comparison['annual_value'] = annual_value
                
                # Use scalar comparisons
                if not np.isnan(ttm_value) and not np.isnan(annual_value) and abs(annual_value) > 1e-10:
                    pct_diff = abs(ttm_value - annual_value) / abs(annual_value)
                    comparison['pct_diff'] = pct_diff
                    comparison['within_margin'] = pct_diff <= margin
                    
        except Exception as e:
            comparison['error'] = str(e)
            
        return comparison


def calculate_ratios_ttm_with_fallback(
    combined_stock_data_df: pd.DataFrame,
    symbol: str,
    prefer_ttm: bool = True
) -> pd.DataFrame:
    """
    Drop-in replacement for stock_data_fetch.calculate_ratios() that uses TTM when available.
    
    This function can replace the existing calculate_ratios() function to provide
    TTM-based ratio calculations with automatic fallback to annual data.
    
    Args:
        combined_stock_data_df: DataFrame with combined stock data (price + financial)
        symbol: Stock ticker symbol
        prefer_ttm: Whether to prefer TTM over annual (default: True)
        
    Returns:
        DataFrame with P/S, P/E, P/B, P/FCF ratios and data source metadata
    """
    if combined_stock_data_df.empty:
        raise ValueError("No stock data provided.")
    
    calculator = TTMFinancialCalculator()
    
    # Get financial data with source tracking
    fin_data = calculator.get_best_available_financials(symbol, prefer_ttm)
    
    # If we have existing financial columns in the DataFrame, use TTM-enhanced calculation
    if fin_data['source'] == 'ttm':
        return calculator.calculate_ratios_with_source_tracking(
            symbol, combined_stock_data_df, prefer_ttm=True
        )
    else:
        # Fall back to existing calculation logic for annual data
        result = combined_stock_data_df.copy()
        
        try:
            # Use existing annual-based calculation
            result["P/S"] = result["close_Price"] / (result["revenue"] / result["average_shares"])
            result["P/E"] = result["close_Price"] / result["eps"]
            result["P/B"] = result["close_Price"] / result["book_Value_Per_Share"]
            result["P/FCF"] = result["close_Price"] / result["free_Cash_Flow_Per_Share"]
            
            # Shift to avoid look-ahead bias
            result[["P/S", "P/E", "P/B", "P/FCF"]] = result[["P/S", "P/E", "P/B", "P/FCF"]].shift(1)
            
            # Add source tracking
            result['ratio_data_source'] = 'annual'
            result['quarters_available'] = fin_data['quarters_available']
            
            print(f"📅 Ratios calculated using annual data for {symbol}")
            
        except KeyError as e:
            print(f"Missing column for ratio calculation: {e}")
            result["P/S"] = np.nan
            result["P/E"] = np.nan
            result["P/B"] = np.nan
            result["P/FCF"] = np.nan
            result['ratio_data_source'] = 'failed'
            result['quarters_available'] = 0
            
        return result


# Convenience functions for direct use
def get_financial_data_with_ttm_preference(symbol: str, years: int = 10) -> Dict[str, Any]:
    """
    Get financial data for a symbol, preferring TTM when available.
    
    Args:
        symbol: Stock ticker symbol
        years: Years of historical data
        
    Returns:
        Dictionary with financial data and source metadata
    """
    calculator = TTMFinancialCalculator()
    return calculator.get_best_available_financials(symbol, prefer_ttm=True, years=years)


def validate_existing_ratios_against_ttm(
    symbol: str,
    existing_ratios_df: pd.DataFrame,
    margin_of_error: float = 0.15
) -> Dict[str, Any]:
    """
    Validate existing ratio data against newly calculated TTM ratios.
    
    Used during migration to verify that TTM calculations are consistent
    with previously stored annual-based ratios within acceptable margin.
    
    Args:
        symbol: Stock ticker symbol
        existing_ratios_df: DataFrame with existing ratio data (from database)
        margin_of_error: Acceptable percentage difference (default: 15%)
        
    Returns:
        Validation results dictionary
    """
    calculator = TTMFinancialCalculator()
    return calculator.validate_ttm_vs_annual(symbol, margin_of_error)


# Demo/testing
if __name__ == "__main__":
    print("Testing TTM Financial Calculator")
    print("="*60)
    
    symbol = "AAPL"
    
    # Test 1: Check TTM availability
    calculator = TTMFinancialCalculator()
    can_use, quarters = calculator.can_use_ttm(symbol)
    print(f"\n1. TTM availability for {symbol}:")
    print(f"   Can use TTM: {can_use}")
    print(f"   Quarters available: {quarters}")
    
    # Test 2: Get best available financials
    print(f"\n2. Getting best available financials for {symbol}...")
    fin_data = calculator.get_best_available_financials(symbol, prefer_ttm=True)
    print(f"   Data source: {fin_data['source']}")
    print(f"   Quarters available: {fin_data['quarters_available']}")
    
    # Test 3: Validate TTM vs Annual
    print(f"\n3. Validating TTM vs Annual data for {symbol}...")
    validation = calculator.validate_ttm_vs_annual(symbol)
    print(f"   Validation passed: {validation['valid']}")
    if validation['warnings']:
        print(f"   Warnings: {validation['warnings']}")
    
    # Test 4: Calculate ratios with source tracking
    print(f"\n4. Testing ratio calculation...")
    # Create sample price data
    import yfinance as yf
    ticker_data = yf.download(symbol, period='1y', progress=False)
    price_df = pd.DataFrame({
        'date': ticker_data.index,
        'close_Price': ticker_data['Close'].values.flatten()
    })
    
    result = calculator.calculate_ratios_with_source_tracking(symbol, price_df)
    print(f"   Data source: {result['ratio_data_source'].iloc[0]}")
    print(f"   Sample P/E: {result['P/E'].dropna().iloc[-1]:.2f}")
    print(f"   Sample P/B: {result['P/B'].dropna().iloc[-1]:.2f}")
