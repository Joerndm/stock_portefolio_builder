"""
Enhanced Financial Data Fetcher Module

This module provides enhanced functionality for fetching financial data with support for:
- Quarterly financial reports with TTM (Trailing Twelve Months) calculations
- Extended historical data (5-10 years when available)
- Multiple data sources with fallback options
- Industry-specific financial calculations

The module is designed to replace/complement the financial data fetching in stock_data_fetch.py
with more granular quarterly data that provides a more current view of company financials.

Features:
    - Quarterly financial statements (income statement, balance sheet, cash flow)
    - Automatic TTM calculations from quarterly data
    - Support for 5-10 years of historical data
    - Growth rate calculations on both quarterly and TTM basis
    - Industry-specific handling (banks, insurance, biotech)
    - Financial ratios calculated from latest TTM data

Data Sources:
    - yfinance (primary - quarterly data)
    - Financial Modeling Prep API (optional, requires free API key)
    
Dependencies:
    - pandas
    - yfinance
    - requests (for FMP API)

Author: Stock Portfolio Builder
Last Modified: 2026
"""
import os
import datetime
from typing import Dict, List, Optional, Tuple, Union
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import yfinance as yf

# Optional: Financial Modeling Prep API key (free tier: 250 requests/day)
# Register at: https://financialmodelingprep.com/developer/docs/
FMP_API_KEY = os.environ.get('FMP_API_KEY', '')


class QuarterlyFinancialFetcher:
    """
    Fetches quarterly financial data and calculates TTM metrics.
    
    Attributes:
        use_fmp: Whether to use Financial Modeling Prep API as data source
        fmp_api_key: API key for FMP (if using)
    """
    
    def __init__(self, use_fmp: bool = False, fmp_api_key: str = ''):
        """
        Initialize the QuarterlyFinancialFetcher.
        
        Args:
            use_fmp: Whether to use FMP API (provides 10+ years of data)
            fmp_api_key: API key for Financial Modeling Prep
        """
        self.use_fmp = use_fmp and bool(fmp_api_key or FMP_API_KEY)
        self.fmp_api_key = fmp_api_key or FMP_API_KEY
        
    # Epsilon threshold for near-zero denominators
    RATIO_EPSILON = 0.01
    
    def _safe_divide(self, numerator, denominator):
        """
        Safely divide two numbers or arrays, handling zero division, near-zero
        denominators, and NaN values.
        
        Works with both scalar values and numpy arrays/pandas Series.
        """
        try:
            eps = self.RATIO_EPSILON
            
            # Handle scalar case (most common in this module)
            if np.isscalar(numerator) and np.isscalar(denominator):
                if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
                    return np.nan
                if abs(denominator) < eps:
                    return np.nan
                return numerator / denominator
            
            # Convert and broadcast arrays to compatible shapes
            num = np.atleast_1d(np.asarray(numerator, dtype=float))
            den = np.atleast_1d(np.asarray(denominator, dtype=float))
            
            # Broadcast to same shape
            num, den = np.broadcast_arrays(num, den)
            num = num.astype(float)  # broadcast_arrays returns read-only, need writable copy for result
            
            result = np.full(num.shape, np.nan, dtype=float)
            valid_mask = (np.abs(den) >= eps) & ~np.isnan(den) & ~np.isnan(num)
            
            if np.any(valid_mask):
                result[valid_mask] = num[valid_mask] / den[valid_mask]
            
            # Return scalar if result is single element
            if result.size == 1:
                return float(result.flat[0])
            
            return result
            
        except (TypeError, ZeroDivisionError, ValueError):
            return np.nan
    
    def _calculate_growth(self, current, previous):
        """Calculate growth rate between two values."""
        try:
            # Handle scalar case
            if np.isscalar(previous) and np.isscalar(current):
                if previous == 0 or pd.isna(previous) or pd.isna(current):
                    return np.nan
                return self._safe_divide(current - previous, abs(previous))
            
            # Handle array case - ensure at least 1D
            prev = np.atleast_1d(np.asarray(previous, dtype=float))
            curr = np.atleast_1d(np.asarray(current, dtype=float))
            valid_mask = (prev != 0) & ~np.isnan(prev) & ~np.isnan(curr)
            
            if not np.any(valid_mask):
                if np.isscalar(current):
                    return np.nan
                return np.full_like(curr, np.nan, dtype=float)
            
            return self._safe_divide(curr - prev, np.abs(prev))
            
        except (TypeError, ValueError):
            return np.nan
    
    def fetch_quarterly_income_statement(self, symbol: str, years: int = 10, retry_count: int = 3) -> pd.DataFrame:
        """
        Fetch quarterly income statement data.
        
        Args:
            symbol: Stock ticker symbol
            years: Number of years of historical data to fetch
            retry_count: Number of retries on rate limiting
            
        Returns:
            DataFrame with quarterly income statement data
        """
        import time
        
        for attempt in range(retry_count):
            try:
                ticker = yf.Ticker(symbol)
                
                # yfinance provides quarterly financials via get_income_stmt method
                # Try the newer API first, fall back to property
                try:
                    quarterly_income = ticker.get_income_stmt(freq='quarterly')
                except:
                    quarterly_income = ticker.quarterly_income_stmt
                
                if quarterly_income is None or quarterly_income.empty:
                    # Also try the quarterly_financials property as fallback
                    try:
                        quarterly_income = ticker.quarterly_financials
                    except:
                        pass
                
                if quarterly_income is None or quarterly_income.empty:
                    print(f"No quarterly income statement data available for {symbol}")
                    return pd.DataFrame()
                
                # Transpose to have dates as rows
                df = quarterly_income.T.copy()
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                
                # Filter to requested years
                cutoff_date = datetime.datetime.now() - relativedelta(years=years)
                df = df[df.index >= cutoff_date]
                
                # Standardize column names
                df = self._standardize_income_columns(df)
                
                return df
                
            except Exception as e:
                error_str = str(e).lower()
                if '429' in error_str or 'too many requests' in error_str:
                    wait_time = (attempt + 1) * 30  # 30s, 60s, 90s
                    print(f"Rate limited fetching {symbol}. Waiting {wait_time}s (attempt {attempt + 1}/{retry_count})")
                    time.sleep(wait_time)
                else:
                    print(f"Error fetching quarterly income statement for {symbol}: {e}")
                    return pd.DataFrame()
        
        print(f"Failed to fetch quarterly income statement for {symbol} after {retry_count} attempts")
        return pd.DataFrame()
    
    def fetch_quarterly_balance_sheet(self, symbol: str, years: int = 10) -> pd.DataFrame:
        """
        Fetch quarterly balance sheet data.
        
        Args:
            symbol: Stock ticker symbol
            years: Number of years of historical data to fetch
            
        Returns:
            DataFrame with quarterly balance sheet data
        """
        try:
            ticker = yf.Ticker(symbol)
            
            quarterly_bs = ticker.quarterly_balancesheet
            
            if quarterly_bs is None or quarterly_bs.empty:
                print(f"No quarterly balance sheet data available for {symbol}")
                return pd.DataFrame()
            
            df = quarterly_bs.T.copy()
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            cutoff_date = datetime.datetime.now() - relativedelta(years=years)
            df = df[df.index >= cutoff_date]
            
            df = self._standardize_balance_sheet_columns(df)
            
            return df
            
        except Exception as e:
            print(f"Error fetching quarterly balance sheet for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_quarterly_cash_flow(self, symbol: str, years: int = 10) -> pd.DataFrame:
        """
        Fetch quarterly cash flow data.
        
        Args:
            symbol: Stock ticker symbol
            years: Number of years of historical data to fetch
            
        Returns:
            DataFrame with quarterly cash flow data
        """
        try:
            ticker = yf.Ticker(symbol)
            
            quarterly_cf = ticker.quarterly_cashflow
            
            if quarterly_cf is None or quarterly_cf.empty:
                print(f"No quarterly cash flow data available for {symbol}")
                return pd.DataFrame()
            
            df = quarterly_cf.T.copy()
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            cutoff_date = datetime.datetime.now() - relativedelta(years=years)
            df = df[df.index >= cutoff_date]
            
            df = self._standardize_cash_flow_columns(df)
            
            return df
            
        except Exception as e:
            print(f"Error fetching quarterly cash flow for {symbol}: {e}")
            return pd.DataFrame()
    
    def _annual_cashflow_fallback(self, symbol: str, years: int = 10) -> pd.DataFrame:
        """
        Derive TTM-like cash flow from annual reports when quarterly data is unavailable.
        
        Annual figures are already 12-month totals, so they serve as a direct TTM proxy.
        Each annual report date is treated as the TTM value for that date.
        
        Args:
            symbol: Stock ticker symbol
            years: Number of years of data to fetch
            
        Returns:
            DataFrame with columns [date, operating_cash_flow_ttm, free_cash_flow_ttm, capex_ttm]
        """
        try:
            ticker = yf.Ticker(symbol)
            annual_cf = ticker.cashflow
            
            if annual_cf is None or annual_cf.empty:
                return pd.DataFrame()
            
            df = annual_cf.T.copy()
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            cutoff_date = datetime.datetime.now() - relativedelta(years=years)
            df = df[df.index >= cutoff_date]
            
            if df.empty:
                return pd.DataFrame()
            
            df = self._standardize_cash_flow_columns(df)
            
            result = pd.DataFrame({'date': df.index})
            
            col_map = {
                'operating_cash_flow': 'operating_cash_flow_ttm',
                'free_cash_flow': 'free_cash_flow_ttm',
                'capex': 'capex_ttm',
            }
            for src, dst in col_map.items():
                result[dst] = df[src].values if src in df.columns else np.nan
            
            print(f"   ↳ Using annual cash flow as TTM fallback for {symbol} ({len(result)} years)")
            return result
            
        except Exception as e:
            print(f"   ↳ Annual cash flow fallback failed for {symbol}: {e}")
            return pd.DataFrame()
    
    def _standardize_income_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize income statement column names."""
        # Map both spaced and non-spaced versions (yfinance sometimes returns either)
        column_mapping = {
            # Spaced versions
            'Total Revenue': 'revenue',
            'Gross Profit': 'gross_profit',
            'Operating Income': 'operating_income',
            'Net Income': 'net_income',
            'Net Income Common Stockholders': 'net_income_common',  # Use different name to avoid duplicates
            'Basic EPS': 'eps_basic',
            'Diluted EPS': 'eps_diluted',
            'Basic Average Shares': 'shares_basic',
            'Diluted Average Shares': 'shares_diluted',
            'EBITDA': 'ebitda',
            'EBIT': 'ebit',
            'Interest Expense': 'interest_expense',
            'Research And Development': 'rd_expense',
            'Selling General And Administration': 'sga_expense',
            'Cost Of Revenue': 'cost_of_revenue',
            # Non-spaced versions (yfinance sometimes returns these)
            'TotalRevenue': 'revenue',
            'OperatingRevenue': 'operating_revenue',  # Different from total revenue
            'GrossProfit': 'gross_profit',
            'OperatingIncome': 'operating_income',
            'NetIncome': 'net_income',
            'NetIncomeCommonStockholders': 'net_income_common',
            'BasicEPS': 'eps_basic',
            'DilutedEPS': 'eps_diluted',
            'BasicAverageShares': 'shares_basic',
            'DilutedAverageShares': 'shares_diluted',
            'InterestExpense': 'interest_expense',
            'CostOfRevenue': 'cost_of_revenue',
        }
        
        df = df.rename(columns=column_mapping)
        
        # Remove duplicate columns (keep first occurrence)
        df = df.loc[:, ~df.columns.duplicated()]
        
        # If we don't have net_income but have net_income_common, use that
        if 'net_income' not in df.columns and 'net_income_common' in df.columns:
            df['net_income'] = df['net_income_common']
            
        # If we don't have revenue but have operating_revenue, use that
        if 'revenue' not in df.columns and 'operating_revenue' in df.columns:
            df['revenue'] = df['operating_revenue']
            
        return df
    
    def _standardize_balance_sheet_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize balance sheet column names."""
        column_mapping = {
            # Spaced versions
            'Total Assets': 'total_assets',
            'Total Liabilities Net Minority Interest': 'total_liabilities',
            'Total Equity Gross Minority Interest': 'total_equity',
            'Stockholders Equity': 'stockholders_equity',  # Don't duplicate, use different name
            'Current Assets': 'current_assets',
            'Current Liabilities': 'current_liabilities',
            'Cash And Cash Equivalents': 'cash',
            'Cash Cash Equivalents And Short Term Investments': 'cash_and_investments',
            'Total Debt': 'total_debt',
            'Long Term Debt': 'long_term_debt',
            'Short Term Debt': 'short_term_debt',
            'Inventory': 'inventory',
            'Accounts Receivable': 'accounts_receivable',
            'Accounts Payable': 'accounts_payable',
            'Goodwill': 'goodwill',
            'Intangible Assets': 'intangible_assets',
            # Non-spaced versions
            'TotalAssets': 'total_assets',
            'TotalLiabilitiesNetMinorityInterest': 'total_liabilities',
            'TotalEquityGrossMinorityInterest': 'total_equity',
            'StockholdersEquity': 'stockholders_equity',  # Don't duplicate
            'CurrentAssets': 'current_assets',
            'CurrentLiabilities': 'current_liabilities',
            'CashAndCashEquivalents': 'cash',
            'CashCashEquivalentsAndShortTermInvestments': 'cash_and_investments',
            'TotalDebt': 'total_debt',
            'LongTermDebt': 'long_term_debt',
            'ShortTermDebt': 'short_term_debt',
            'AccountsReceivable': 'accounts_receivable',
            'AccountsPayable': 'accounts_payable',
            'IntangibleAssets': 'intangible_assets',
        }
        
        df = df.rename(columns=column_mapping)
        
        # Remove duplicate columns (keep first occurrence)
        df = df.loc[:, ~df.columns.duplicated()]
        
        # If we don't have total_equity but have stockholders_equity, use that
        if 'total_equity' not in df.columns and 'stockholders_equity' in df.columns:
            df['total_equity'] = df['stockholders_equity']
            
        return df
    
    def _standardize_cash_flow_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize cash flow column names."""
        column_mapping = {
            # Spaced versions
            'Operating Cash Flow': 'operating_cash_flow',
            'Capital Expenditure': 'capex',
            'Free Cash Flow': 'free_cash_flow',
            'Investing Cash Flow': 'investing_cash_flow',
            'Financing Cash Flow': 'financing_cash_flow',
            'Dividends Paid': 'dividends_paid',
            'Repurchase Of Capital Stock': 'share_repurchases',
            'Issuance Of Capital Stock': 'stock_issuance',
            'Issuance Of Debt': 'debt_issuance',
            'Repayment Of Debt': 'debt_repayment',
            # Non-spaced versions
            'OperatingCashFlow': 'operating_cash_flow',
            'CapitalExpenditure': 'capex',
            'FreeCashFlow': 'free_cash_flow',
            'InvestingCashFlow': 'investing_cash_flow',
            'FinancingCashFlow': 'financing_cash_flow',
            'DividendsPaid': 'dividends_paid',
            'RepurchaseOfCapitalStock': 'share_repurchases',
            'IssuanceOfCapitalStock': 'stock_issuance',
            'IssuanceOfDebt': 'debt_issuance',
            'RepaymentOfDebt': 'debt_repayment',
        }
        
        df = df.rename(columns=column_mapping)
        return df
    
    def calculate_ttm(self, quarterly_df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Calculate TTM (Trailing Twelve Months) values from quarterly data.
        
        For flow metrics (income statement, cash flow), sums the last 4 quarters.
        For point-in-time metrics (balance sheet), uses the latest value.
        
        Args:
            quarterly_df: DataFrame with quarterly data, index should be dates
            columns: List of column names to calculate TTM for
            
        Returns:
            DataFrame with TTM values for each quarter-end date
        """
        if quarterly_df.empty:
            return pd.DataFrame()
        
        ttm_records = []
        
        # Ensure data is sorted by date
        quarterly_df = quarterly_df.sort_index()
        
        # For each quarter, calculate TTM
        for i in range(3, len(quarterly_df)):
            date = quarterly_df.index[i]
            ttm_data = {'date': date}
            
            for col in columns:
                if col in quarterly_df.columns:
                    # Sum last 4 quarters for TTM
                    ttm_value = quarterly_df[col].iloc[i-3:i+1].sum()
                    ttm_data[f'{col}_ttm'] = ttm_value
                    
                    # Also include quarterly value
                    ttm_data[f'{col}_q'] = quarterly_df[col].iloc[i]
            
            ttm_records.append(ttm_data)
        
        return pd.DataFrame(ttm_records)
    
    def calculate_financial_ratios_ttm(
        self,
        income_ttm: pd.DataFrame,
        balance_sheet: pd.DataFrame,
        cash_flow_ttm: pd.DataFrame,
        symbol: str
    ) -> pd.DataFrame:
        """
        Calculate financial ratios using TTM data.
        
        Args:
            income_ttm: TTM income statement data
            balance_sheet: Balance sheet data (point-in-time)
            cash_flow_ttm: TTM cash flow data
            symbol: Stock ticker symbol
            
        Returns:
            DataFrame with calculated financial ratios
        """
        try:
            # Guard: if income_ttm is empty or missing 'date' column, return early
            if income_ttm.empty or 'date' not in income_ttm.columns:
                return pd.DataFrame()
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            shares_outstanding = info.get('sharesOutstanding', None)
            industry = info.get('industry', '').lower()
            
            ratios = []
            
            # Align dates across all dataframes
            common_dates = set(income_ttm['date']) & set(balance_sheet.index)
            if not common_dates:
                print(f"No common dates found for {symbol}")
                return pd.DataFrame()
            
            for date in sorted(common_dates):
                income_row = income_ttm[income_ttm['date'] == date].iloc[0]
                bs_row = balance_sheet.loc[date]
                
                cf_row = None
                if not cash_flow_ttm.empty and date in cash_flow_ttm['date'].values:
                    cf_row = cash_flow_ttm[cash_flow_ttm['date'] == date].iloc[0]
                
                ratio_data = {
                    'date': date,
                    'ticker': symbol,
                }
                
                # Profitability Ratios
                revenue_ttm = income_row.get('revenue_ttm', np.nan)
                gross_profit_ttm = income_row.get('gross_profit_ttm', np.nan)
                operating_income_ttm = income_row.get('operating_income_ttm', np.nan)
                net_income_ttm = income_row.get('net_income_ttm', np.nan)
                
                ratio_data['gross_margin_ttm'] = self._safe_divide(gross_profit_ttm, revenue_ttm)
                ratio_data['operating_margin_ttm'] = self._safe_divide(operating_income_ttm, revenue_ttm)
                ratio_data['net_margin_ttm'] = self._safe_divide(net_income_ttm, revenue_ttm)
                
                # Balance Sheet Ratios
                total_assets = bs_row.get('total_assets', np.nan)
                total_equity = bs_row.get('total_equity', np.nan)
                total_liabilities = bs_row.get('total_liabilities', np.nan)
                current_assets = bs_row.get('current_assets', np.nan)
                current_liabilities = bs_row.get('current_liabilities', np.nan)
                cash = bs_row.get('cash', bs_row.get('cash_and_investments', np.nan))
                inventory = bs_row.get('inventory', 0)
                
                ratio_data['roa_ttm'] = self._safe_divide(net_income_ttm, total_assets)
                ratio_data['roe_ttm'] = self._safe_divide(net_income_ttm, total_equity)
                ratio_data['debt_to_equity'] = self._safe_divide(total_liabilities, total_equity)
                ratio_data['current_ratio'] = self._safe_divide(current_assets, current_liabilities)
                ratio_data['quick_ratio'] = self._safe_divide(current_assets - inventory, current_liabilities)
                
                # Per Share Metrics
                if shares_outstanding:
                    ratio_data['book_value_per_share'] = self._safe_divide(total_equity, shares_outstanding)
                    ratio_data['eps_ttm'] = self._safe_divide(net_income_ttm, shares_outstanding)
                    
                    if cf_row is not None:
                        fcf_ttm = cf_row.get('free_cash_flow_ttm', np.nan)
                        ratio_data['fcf_per_share'] = self._safe_divide(fcf_ttm, shares_outstanding)
                
                # Growth Metrics (calculated in separate pass)
                ratio_data['revenue_ttm'] = revenue_ttm
                ratio_data['net_income_ttm'] = net_income_ttm
                ratio_data['total_assets'] = total_assets
                ratio_data['total_equity'] = total_equity
                
                ratios.append(ratio_data)
            
            df = pd.DataFrame(ratios)
            
            # Calculate YoY growth rates (comparing to same quarter previous year)
            if len(df) >= 4:
                df['revenue_growth_yoy'] = df['revenue_ttm'].pct_change(periods=4, fill_method=None)
                df['net_income_growth_yoy'] = df['net_income_ttm'].pct_change(periods=4, fill_method=None)
                df['assets_growth_yoy'] = df['total_assets'].pct_change(periods=4, fill_method=None)
            
            return df
            
        except Exception as e:
            print(f"Error calculating ratios for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_complete_quarterly_financials(
        self,
        symbol: str,
        years: int = 10,
        calculate_ttm: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch complete quarterly financial data for a symbol.
        
        Args:
            symbol: Stock ticker symbol
            years: Number of years of historical data
            calculate_ttm: Whether to calculate TTM metrics
            
        Returns:
            Dictionary containing:
                - 'income_quarterly': Quarterly income statement
                - 'balance_sheet': Quarterly balance sheet
                - 'cash_flow_quarterly': Quarterly cash flow
                - 'income_ttm': TTM income statement metrics
                - 'cash_flow_ttm': TTM cash flow metrics
                - 'ratios': Financial ratios
        """
        result = {}
        
        # Fetch quarterly data
        income_q = self.fetch_quarterly_income_statement(symbol, years)
        balance_sheet = self.fetch_quarterly_balance_sheet(symbol, years)
        cash_flow_q = self.fetch_quarterly_cash_flow(symbol, years)
        
        result['income_quarterly'] = income_q
        result['balance_sheet'] = balance_sheet
        result['cash_flow_quarterly'] = cash_flow_q
        
        if calculate_ttm and not income_q.empty:
            # Calculate TTM for flow metrics
            income_cols = ['revenue', 'gross_profit', 'operating_income', 'net_income', 'ebitda']
            income_cols = [c for c in income_cols if c in income_q.columns]
            result['income_ttm'] = self.calculate_ttm(income_q, income_cols)
            
            if not cash_flow_q.empty:
                cf_cols = ['operating_cash_flow', 'free_cash_flow', 'capex']
                cf_cols = [c for c in cf_cols if c in cash_flow_q.columns]
                result['cash_flow_ttm'] = self.calculate_ttm(cash_flow_q, cf_cols)
            else:
                # Fallback: derive annual cash flow as TTM proxy when quarterly is unavailable
                result['cash_flow_ttm'] = self._annual_cashflow_fallback(symbol, years)
            
            # Calculate financial ratios
            if not balance_sheet.empty:
                result['ratios'] = self.calculate_financial_ratios_ttm(
                    result['income_ttm'],
                    balance_sheet,
                    result['cash_flow_ttm'],
                    symbol
                )
            else:
                result['ratios'] = pd.DataFrame()
        
        return result


def fetch_quarterly_financial_data(symbol: str, years: int = 10) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to fetch quarterly financial data with TTM calculations.
    
    This function is designed to be a drop-in enhancement for the existing
    fetch_stock_financial_data function in stock_data_fetch.py.
    
    Args:
        symbol: Stock ticker symbol
        years: Number of years of historical data to fetch (default: 10)
        
    Returns:
        Dictionary containing quarterly and TTM financial data
        
    Example:
        >>> data = fetch_quarterly_financial_data('AAPL', years=10)
        >>> income_ttm = data['income_ttm']
        >>> ratios = data['ratios']
    """
    fetcher = QuarterlyFinancialFetcher()
    return fetcher.fetch_complete_quarterly_financials(symbol, years, calculate_ttm=True)


def convert_quarterly_to_annual_compatible(quarterly_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Convert quarterly/TTM data to a format compatible with the existing database schema.
    
    This allows gradual migration from annual to quarterly data while maintaining
    backward compatibility with the existing database structure.
    
    Args:
        quarterly_data: Output from fetch_quarterly_financial_data()
        
    Returns:
        DataFrame formatted to match the existing stock_financial_data structure
    """
    if 'ratios' not in quarterly_data or quarterly_data['ratios'].empty:
        return pd.DataFrame()
    
    ratios = quarterly_data['ratios'].copy()
    
    # Map to existing database column names
    column_mapping = {
        'date': 'financial_Statement_Date',
        'ticker': 'ticker',
        'revenue_ttm': 'revenue',
        'revenue_growth_yoy': 'revenue_Growth',
        'gross_margin_ttm': 'gross_Margin',
        'operating_margin_ttm': 'operating_Earning_Margin',
        'net_margin_ttm': 'net_Income_Margin',
        'net_income_ttm': 'net_Income',
        'net_income_growth_yoy': 'net_Income_Growth',
        'eps_ttm': 'eps',
        'roa_ttm': 'return_On_Assets',
        'roe_ttm': 'return_On_Equity',
        'current_ratio': 'current_Ratio',
        'quick_ratio': 'quick_Ratio',
        'debt_to_equity': 'debt_To_Equity',
        'book_value_per_share': 'book_Value_Per_Share',
        'total_assets': 'total_Assets',
        'total_equity': 'equity',
        'fcf_per_share': 'free_Cash_Flow_Per_Share',
    }
    
    # Only include columns that exist
    existing_cols = {k: v for k, v in column_mapping.items() if k in ratios.columns}
    result = ratios[list(existing_cols.keys())].rename(columns=existing_cols)
    
    return result


def transform_quarterly_to_db_schema(quarterly_df: pd.DataFrame, ticker: str, 
                                     statement_type: str = 'income') -> pd.DataFrame:
    """
    Transform quarterly data from yfinance format to database schema format.
    
    The database expects columns with _q suffix for quarterly values.
    This function:
    1. Renames columns to add _q suffix
    2. Adds ticker and fiscal_quarter_end columns
    3. Returns data ready for database export
    
    Args:
        quarterly_df: DataFrame from fetch_quarterly_income_statement, etc.
        ticker: Stock ticker symbol
        statement_type: 'income', 'balancesheet', or 'cashflow'
        
    Returns:
        DataFrame formatted for database export with _q suffixed columns
    """
    if quarterly_df.empty:
        return pd.DataFrame()
    
    df = quarterly_df.copy()
    
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Reset index to make date a column
    df = df.reset_index()
    df = df.rename(columns={'index': 'fiscal_quarter_end'})
    
    # Add ticker
    df['ticker'] = ticker
    
    if statement_type == 'income':
        # Map yfinance column names to database column names with _q suffix
        # Handle net_income vs net_income_common - prefer net_income if both exist
        column_mapping = {
            'revenue': 'revenue_q',
            'gross_profit': 'gross_profit_q',
            'operating_income': 'operating_income_q',
            'net_income': 'net_income_q',
            'eps_basic': 'eps_basic_q',
            'eps_diluted': 'eps_diluted_q',
            'shares_diluted': 'shares_diluted',  # No _q suffix for shares
            'ebitda': 'ebitda_q',
            'ebit': 'ebit_q',
            'interest_expense': 'interest_expense_q',
            'rd_expense': 'rd_expense_q',
            'sga_expense': 'sga_expense_q',
            'cost_of_revenue': 'cost_of_revenue_q',
        }
        # If net_income doesn't exist but net_income_common does, use it
        if 'net_income' not in df.columns and 'net_income_common' in df.columns:
            column_mapping['net_income_common'] = 'net_income_q'
    elif statement_type == 'balancesheet':
        # Balance sheet items are point-in-time, typically don't need _q suffix
        column_mapping = {
            'total_assets': 'total_assets',
            'total_liabilities': 'total_liabilities',
            'total_equity': 'total_equity',
            'stockholders_equity': 'stockholders_equity',
            'current_assets': 'current_assets',
            'current_liabilities': 'current_liabilities',
            'cash': 'cash',
            'cash_and_investments': 'cash_and_investments',
            'total_debt': 'total_debt',
            'long_term_debt': 'long_term_debt',
            'short_term_debt': 'short_term_debt',
            'inventory': 'inventory',
            'accounts_receivable': 'accounts_receivable',
            'accounts_payable': 'accounts_payable',
            'goodwill': 'goodwill',
            'intangible_assets': 'intangible_assets',
        }
    elif statement_type == 'cashflow':
        # Cash flow items are flow metrics, need _q suffix
        column_mapping = {
            'operating_cash_flow': 'operating_cash_flow_q',
            'capex': 'capex_q',
            'free_cash_flow': 'free_cash_flow_q',
            'investing_cash_flow': 'investing_cash_flow_q',
            'financing_cash_flow': 'financing_cash_flow_q',
            'dividends_paid': 'dividends_paid_q',
            'share_repurchases': 'share_repurchases_q',
            'stock_issuance': 'stock_issuance_q',
            'debt_issuance': 'debt_issuance_q',
            'debt_repayment': 'debt_repayment_q',
        }
    else:
        column_mapping = {}
    
    # Rename columns that exist
    existing_mappings = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=existing_mappings)
    
    # Ensure fiscal_quarter_end is date only (not datetime with time)
    df['fiscal_quarter_end'] = pd.to_datetime(df['fiscal_quarter_end']).dt.date
    
    return df


def calculate_ttm_from_quarterly(quarterly_df: pd.DataFrame, ticker: str,
                                 statement_type: str = 'income') -> pd.DataFrame:
    """
    Calculate TTM (Trailing Twelve Months) values from quarterly data.
    
    For each quarter, sums the values from that quarter and the 3 preceding quarters.
    
    Args:
        quarterly_df: DataFrame with quarterly data (already transformed with _q suffix)
        ticker: Stock ticker symbol
        statement_type: 'income', 'balancesheet', or 'cashflow'
        
    Returns:
        DataFrame with TTM columns added (e.g., revenue_ttm = sum of 4 quarters of revenue_q)
    """
    if quarterly_df.empty or len(quarterly_df) < 4:
        return quarterly_df
    
    df = quarterly_df.copy()
    
    # Sort by date
    df['fiscal_quarter_end'] = pd.to_datetime(df['fiscal_quarter_end'])
    df = df.sort_values('fiscal_quarter_end')
    
    # Columns that should be summed for TTM (flow metrics)
    if statement_type == 'income':
        ttm_columns = ['revenue_q', 'gross_profit_q', 'operating_income_q', 
                       'net_income_q', 'ebitda_q', 'ebit_q', 
                       'interest_expense_q', 'rd_expense_q', 'sga_expense_q',
                       'cost_of_revenue_q']
    elif statement_type == 'cashflow':
        ttm_columns = ['operating_cash_flow_q', 'capex_q', 'free_cash_flow_q',
                       'investing_cash_flow_q', 'financing_cash_flow_q',
                       'dividends_paid_q', 'share_repurchases_q']
    else:
        # Balance sheet items are point-in-time, no TTM needed
        return df
    
    # Calculate TTM for each flow column
    for col in ttm_columns:
        if col in df.columns:
            ttm_col = col.replace('_q', '_ttm')
            # Rolling sum of last 4 quarters, allow 3 quarters minimum
            # and proportionally scale to a full year estimate
            rolling_sum = df[col].rolling(window=4, min_periods=3).sum()
            rolling_count = df[col].rolling(window=4, min_periods=3).count()
            df[ttm_col] = rolling_sum * (4.0 / rolling_count)
    
    # Calculate growth rates
    if statement_type == 'income' and 'revenue_ttm' in df.columns:
        # YoY growth (compare to 4 quarters ago)
        df['revenue_growth_yoy'] = df['revenue_ttm'].pct_change(periods=4, fill_method=None)
        if 'net_income_ttm' in df.columns:
            df['net_income_growth_yoy'] = df['net_income_ttm'].pct_change(periods=4, fill_method=None)
        
        # QoQ growth (compare to 1 quarter ago)
        df['revenue_growth_qoq'] = df['revenue_q'].pct_change(periods=1, fill_method=None)
        if 'net_income_q' in df.columns:
            df['net_income_growth_qoq'] = df['net_income_q'].pct_change(periods=1, fill_method=None)
    
    # Convert fiscal_quarter_end back to date
    df['fiscal_quarter_end'] = df['fiscal_quarter_end'].dt.date
    
    return df


def merge_quarterly_data(db_data: pd.DataFrame, yfinance_data: pd.DataFrame, 
                         ticker: str) -> pd.DataFrame:
    """
    Merge existing database quarterly data with new yfinance data.
    
    For quarters that already exist in the database:
    - Update with yfinance data (fills in NULL columns with actual values)
    
    For new quarters not in the database:
    - Add them from yfinance
    
    Args:
        db_data: Existing data from database (may be empty or have NULL columns)
        yfinance_data: New data from yfinance (already transformed to db schema)
        ticker: Stock ticker symbol
        
    Returns:
        Merged DataFrame with updated and new quarters
    """
    if db_data.empty:
        return yfinance_data
    
    if yfinance_data.empty:
        return db_data
    
    # Convert dates to comparable format
    db_data = db_data.copy()
    yfinance_data = yfinance_data.copy()
    
    db_data['fiscal_quarter_end'] = pd.to_datetime(db_data['fiscal_quarter_end']).dt.date
    yfinance_data['fiscal_quarter_end'] = pd.to_datetime(yfinance_data['fiscal_quarter_end']).dt.date
    
    # Get existing dates from database
    existing_dates = set(db_data['fiscal_quarter_end'].tolist())
    yf_dates = set(yfinance_data['fiscal_quarter_end'].tolist())
    
    # Identify new quarters (not in database)
    new_dates = yf_dates - existing_dates
    
    # Identify quarters to update (exist in both but database may have NULL values)
    update_dates = yf_dates & existing_dates
    
    result_parts = []
    
    # For existing quarters, use yfinance data (it has actual values vs database NULLs)
    if update_dates:
        # Use yfinance data for these dates - it has actual values
        updated = yfinance_data[yfinance_data['fiscal_quarter_end'].isin(update_dates)]
        result_parts.append(updated)
    
    # For quarters only in database (not in yfinance), keep database data
    db_only_dates = existing_dates - yf_dates
    if db_only_dates:
        db_only = db_data[db_data['fiscal_quarter_end'].isin(db_only_dates)]
        result_parts.append(db_only)
    
    # Add new quarters from yfinance
    if new_dates:
        new_quarters = yfinance_data[yfinance_data['fiscal_quarter_end'].isin(new_dates)]
        result_parts.append(new_quarters)
        print(f"  Found {len(new_quarters)} new quarters for {ticker}")
    else:
        print(f"  No new quarters to add for {ticker} (updating {len(update_dates)} existing)")
    
    # Combine all parts
    if result_parts:
        combined = pd.concat(result_parts, ignore_index=True)
        combined = combined.sort_values('fiscal_quarter_end')
        return combined
    
    return yfinance_data


# Demo/testing code
if __name__ == "__main__":
    print("Testing Enhanced Financial Data Fetcher")
    print("="*50)
    
    # Test with a sample stock
    symbol = "AAPL"
    print(f"\nFetching quarterly financials for {symbol}...")
    
    data = fetch_quarterly_financial_data(symbol, years=5)
    
    print(f"\nIncome Statement (Quarterly) - {len(data['income_quarterly'])} quarters:")
    if not data['income_quarterly'].empty:
        print(data['income_quarterly'].tail())
    
    print(f"\nIncome Statement (TTM) - {len(data['income_ttm'])} periods:")
    if not data['income_ttm'].empty:
        print(data['income_ttm'].tail())
    
    print(f"\nFinancial Ratios - {len(data['ratios'])} periods:")
    if not data['ratios'].empty:
        print(data['ratios'][['date', 'gross_margin_ttm', 'roe_ttm', 'current_ratio']].tail())
    
    # Convert to annual-compatible format
    print("\nConverted to annual-compatible format:")
    annual_compat = convert_quarterly_to_annual_compatible(data)
    if not annual_compat.empty:
        print(annual_compat.tail())
