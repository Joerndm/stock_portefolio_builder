"""
TTM Data Migration and Validation Script

This script handles the migration from annual financial data to TTM (Trailing Twelve Months)
data while validating that existing data remains within acceptable margins of error.

Features:
    - Recalculates historical ratios using TTM data where 4+ quarters available
    - Falls back to annual data where insufficient quarterly data exists
    - Validates new TTM calculations against existing annual-based values
    - Generates detailed validation reports
    - Supports incremental migration (can be run multiple times safely)

Validation Logic:
    - Compares TTM-calculated P/E, P/B, P/S, P/FCF against existing values
    - Allows configurable margin of error (default: 15%) for acceptable differences
    - Flags significant deviations for manual review
    - Accounts for timing differences between quarterly and annual reports

Usage:
    python migrate_to_ttm_data.py --validate-only  # Dry run, only validate
    python migrate_to_ttm_data.py --migrate        # Actually perform migration
    python migrate_to_ttm_data.py --ticker AAPL    # Process single ticker

Author: Stock Portfolio Builder
Last Modified: 2026
"""
import os
import sys
import argparse
import datetime
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db_connectors
import db_interactions
import fetch_secrets
from ttm_financial_calculator import (
    TTMFinancialCalculator,
    validate_existing_ratios_against_ttm
)


class TTMMigrationValidator:
    """
    Validates and migrates stock ratio data from annual to TTM calculations.
    """
    
    # Ratio columns to validate and migrate
    RATIO_COLUMNS = ['p_s', 'p_e', 'p_b', 'p_fcf']
    
    # Ratios that should be strictly validated (similar between TTM and annual)
    # P/FCF is excluded because TTM P/FCF is fundamentally different from annual:
    # - Annual: Forward-fills the same value for an entire year
    # - TTM: Rolling 4-quarter data that updates quarterly
    STRICT_VALIDATION_RATIOS = ['p_s', 'p_e', 'p_b']
    
    # Column mapping from database to display names
    RATIO_DISPLAY_NAMES = {
        'p_s': 'P/S',
        'p_e': 'P/E',
        'p_b': 'P/B',
        'p_fcf': 'P/FCF'
    }
    
    def __init__(self, margin_of_error: float = 0.15, verbose: bool = True):
        """
        Initialize the migration validator.
        
        Args:
            margin_of_error: Acceptable percentage difference (default: 15%)
            verbose: Whether to print detailed progress messages
        """
        self.margin_of_error = margin_of_error
        self.verbose = verbose
        self.calculator = TTMFinancialCalculator()
        
        # Initialize database connection
        self._init_db_connection()
        
        # Results tracking
        self.validation_results = []
        self.migration_results = []
        
    def _init_db_connection(self):
        """Initialize database connection."""
        try:
            db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
            self.db_con = db_connectors.pandas_mysql_connector(
                db_host, db_user, db_pass, db_name
            )
        except Exception as e:
            print(f"FAILED to connect to database: {e}")
            raise
    
    def _log(self, message: str):
        """Print message if verbose mode enabled."""
        if self.verbose:
            # Handle unicode encoding errors for Windows console
            try:
                print(message)
            except UnicodeEncodeError:
                # Replace problematic characters with ASCII equivalents
                safe_message = message.encode('ascii', 'replace').decode('ascii')
                print(safe_message)
    
    def get_all_tickers_with_ratio_data(self) -> List[str]:
        """Get list of all tickers that have ratio data in the database."""
        query = """
            SELECT DISTINCT ticker 
            FROM stock_ratio_data 
            ORDER BY ticker
        """
        df = pd.read_sql(sql=query, con=self.db_con)
        return df['ticker'].tolist()
    
    def get_existing_ratio_data(self, ticker: str) -> pd.DataFrame:
        """
        Get existing ratio data for a ticker from the database.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with existing ratio data
        """
        query = f"""
            SELECT date, ticker, p_s, p_e, p_b, p_fcf
            FROM stock_ratio_data
            WHERE ticker = '{ticker}'
            ORDER BY date
        """
        return pd.read_sql(sql=query, con=self.db_con)
    
    def get_price_data(self, ticker: str, start_date: Optional[datetime.date] = None) -> pd.DataFrame:
        """
        Get price data for a ticker from the database.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Optional start date filter
            
        Returns:
            DataFrame with price data
        """
        if start_date:
            query = f"""
                SELECT date, ticker, close_Price
                FROM stock_price_data
                WHERE ticker = '{ticker}' AND date >= '{start_date}'
                ORDER BY date
            """
        else:
            query = f"""
                SELECT date, ticker, close_Price
                FROM stock_price_data
                WHERE ticker = '{ticker}'
                ORDER BY date
            """
        return pd.read_sql(sql=query, con=self.db_con)
    
    def calculate_ttm_ratios_for_ticker(self, ticker: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Calculate TTM-based ratios for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Tuple of (ratios_df, metadata_dict)
        """
        # Get price data
        price_data = self.get_price_data(ticker)
        
        if price_data.empty:
            return pd.DataFrame(), {'error': 'No price data available'}
        
        # Calculate ratios using TTM with fallback
        result = self.calculator.calculate_ratios_with_source_tracking(
            ticker, price_data, prefer_ttm=True
        )
        
        metadata = {
            'data_source': result['ratio_data_source'].iloc[0] if not result.empty else 'none',
            'quarters_available': result['quarters_available'].iloc[0] if not result.empty else 0
        }
        
        return result, metadata
    
    def compare_ratio_values(
        self,
        existing_df: pd.DataFrame,
        new_df: pd.DataFrame,
        ratio_col: str
    ) -> Dict[str, Any]:
        """
        Compare existing and new ratio values.
        
        Args:
            existing_df: DataFrame with existing ratios
            new_df: DataFrame with new TTM ratios
            ratio_col: Column name to compare (e.g., 'p_e')
            
        Returns:
            Comparison statistics dictionary
        """
        # Map column names
        display_name = self.RATIO_DISPLAY_NAMES.get(ratio_col, ratio_col)
        new_col = display_name  # New data uses P/E format
        
        result = {
            'ratio': display_name,
            'existing_count': 0,
            'new_count': 0,
            'matched_count': 0,
            'within_margin': 0,
            'outside_margin': 0,
            'mean_abs_diff': np.nan,
            'mean_pct_diff': np.nan,
            'max_pct_diff': np.nan,
            'sample_comparisons': []
        }
        
        if ratio_col not in existing_df.columns or new_col not in new_df.columns:
            return result
        
        # Ensure date columns are compatible
        existing_df = existing_df.copy()
        new_df = new_df.copy()
        existing_df['date'] = pd.to_datetime(existing_df['date']).dt.date
        new_df['date'] = pd.to_datetime(new_df['date']).dt.date
        
        # Convert ratio columns to numeric (handles object/string types)
        existing_df[ratio_col] = pd.to_numeric(existing_df[ratio_col], errors='coerce')
        new_df[new_col] = pd.to_numeric(new_df[new_col], errors='coerce')
        
        # Merge on date
        merged = pd.merge(
            existing_df[['date', ratio_col]].rename(columns={ratio_col: 'existing'}),
            new_df[['date', new_col]].rename(columns={new_col: 'new'}),
            on='date',
            how='inner'
        )
        
        # Filter valid comparisons (both values present and not inf)
        # Use pd.isna() for NaN check and handle inf separately with numeric conversion
        valid = merged[
            merged['existing'].notna() & 
            merged['new'].notna() &
            (merged['existing'] != 0)
        ].copy()
        
        # Now filter out inf values (data is guaranteed numeric after to_numeric)
        valid = valid[~np.isinf(valid['existing'].astype(float)) & ~np.isinf(valid['new'].astype(float))]
        
        result['existing_count'] = len(existing_df[existing_df[ratio_col].notna()])
        result['new_count'] = len(new_df[new_df[new_col].notna()])
        result['matched_count'] = len(valid)
        
        if len(valid) == 0:
            return result
        
        # Calculate differences
        valid['abs_diff'] = (valid['new'] - valid['existing']).abs()
        valid['pct_diff'] = valid['abs_diff'] / valid['existing'].abs()
        
        # Count within/outside margin
        result['within_margin'] = (valid['pct_diff'] <= self.margin_of_error).sum()
        result['outside_margin'] = (valid['pct_diff'] > self.margin_of_error).sum()
        
        # Statistics
        result['mean_abs_diff'] = valid['abs_diff'].mean()
        result['mean_pct_diff'] = valid['pct_diff'].mean()
        result['max_pct_diff'] = valid['pct_diff'].max()
        
        # Sample comparisons (first 5 outside margin)
        outside_margin_samples = valid[valid['pct_diff'] > self.margin_of_error].head(5)
        result['sample_comparisons'] = [
            {
                'date': str(row['date']),
                'existing': row['existing'],
                'new': row['new'],
                'pct_diff': row['pct_diff']
            }
            for _, row in outside_margin_samples.iterrows()
        ]
        
        return result
    
    def validate_ticker(self, ticker: str) -> Dict[str, Any]:
        """
        Validate TTM calculations against existing data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Validation results dictionary
        """
        self._log(f"\n📊 Validating {ticker}...")
        
        result = {
            'ticker': ticker,
            'status': 'pending',
            'data_source': 'none',
            'quarters_available': 0,
            'ratio_comparisons': {},
            'overall_valid': True,
            'warnings': [],
            'errors': []
        }
        
        try:
            # Get existing data
            existing_df = self.get_existing_ratio_data(ticker)
            if existing_df.empty:
                result['status'] = 'no_existing_data'
                result['warnings'].append('No existing ratio data found')
                return result
            
            # Calculate new TTM-based ratios
            new_df, metadata = self.calculate_ttm_ratios_for_ticker(ticker)
            
            if new_df.empty:
                result['status'] = 'calculation_failed'
                result['errors'].append('Failed to calculate TTM ratios')
                return result
            
            result['data_source'] = metadata['data_source']
            result['quarters_available'] = metadata['quarters_available']
            
            # Compare each ratio
            all_within_margin = True
            for ratio_col in self.RATIO_COLUMNS:
                comparison = self.compare_ratio_values(existing_df, new_df, ratio_col)
                result['ratio_comparisons'][ratio_col] = comparison
                
                # Only strictly validate certain ratios
                # P/FCF is expected to differ significantly due to TTM vs annual methodology
                if ratio_col in self.STRICT_VALIDATION_RATIOS and comparison['outside_margin'] > 0:
                    pct_outside = comparison['outside_margin'] / max(1, comparison['matched_count'])
                    if pct_outside > 0.20:  # More than 20% outside margin is concerning
                        all_within_margin = False
                        result['warnings'].append(
                            f"{comparison['ratio']}: {comparison['outside_margin']}/{comparison['matched_count']} "
                            f"values ({pct_outside:.1%}) outside {self.margin_of_error:.0%} margin"
                        )
                elif ratio_col == 'p_fcf' and comparison['outside_margin'] > 0:
                    # P/FCF differences are informational only - expected due to TTM methodology
                    pct_outside = comparison['outside_margin'] / max(1, comparison['matched_count'])
                    result['info'] = result.get('info', [])
                    result['info'].append(
                        f"{comparison['ratio']}: {comparison['outside_margin']}/{comparison['matched_count']} "
                        f"values ({pct_outside:.1%}) differ (expected - TTM vs annual methodology)"
                    )
            
            result['overall_valid'] = all_within_margin
            result['status'] = 'validated' if all_within_margin else 'needs_review'
            
            self._log(f"   Data source: {result['data_source']}")
            self._log(f"   Quarters available: {result['quarters_available']}")
            self._log(f"   Status: {'✅ Valid' if all_within_margin else '⚠️ Needs review'}")
            
        except Exception as e:
            result['status'] = 'error'
            result['errors'].append(str(e))
            self._log(f"   ❌ Error: {e}")
        
        return result
    
    def validate_all_tickers(self) -> List[Dict[str, Any]]:
        """
        Validate all tickers in the database.
        
        Returns:
            List of validation results for each ticker
        """
        tickers = self.get_all_tickers_with_ratio_data()
        self._log(f"\n[VALIDATING] {len(tickers)} tickers...")
        
        results = []
        for ticker in tickers:
            result = self.validate_ticker(ticker)
            results.append(result)
            self.validation_results.append(result)
        
        return results
    
    def migrate_ticker(
        self,
        ticker: str,
        dry_run: bool = True,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Migrate a ticker's ratio data to TTM-based calculations.
        
        Args:
            ticker: Stock ticker symbol
            dry_run: If True, only validate without writing to database
            force: If True, migrate even if validation fails
            
        Returns:
            Migration result dictionary
        """
        self._log(f"\n{'[VALIDATING]' if dry_run else '[MIGRATING]'} {ticker}...")
        
        result = {
            'ticker': ticker,
            'status': 'pending',
            'action': 'dry_run' if dry_run else 'migrate',
            'rows_updated': 0,
            'data_source': 'none',
            'validation': None,
            'errors': []
        }
        
        try:
            # First validate
            validation = self.validate_ticker(ticker)
            result['validation'] = validation
            result['data_source'] = validation['data_source']
            
            if not validation['overall_valid'] and not force:
                result['status'] = 'skipped_validation_failed'
                self._log(f"   ⏭️ Skipped - validation failed (use --force to override)")
                return result
            
            if dry_run:
                result['status'] = 'validated_dry_run'
                return result
            
            # Calculate new ratios
            new_df, metadata = self.calculate_ttm_ratios_for_ticker(ticker)
            
            if new_df.empty:
                result['status'] = 'no_new_data'
                return result
            
            # Prepare data for database
            ratio_df = new_df[['date', 'ticker', 'P/S', 'P/E', 'P/B', 'P/FCF']].copy()
            ratio_df = ratio_df.rename(columns={
                'P/S': 'p_s',
                'P/E': 'p_e', 
                'P/B': 'p_b',
                'P/FCF': 'p_fcf'
            })
            ratio_df = ratio_df.dropna(subset=['date', 'ticker'])
            
            if ratio_df.empty:
                result['status'] = 'no_valid_ratios'
                return result
            
            # Delete existing data for this ticker first
            from sqlalchemy import text
            with self.db_con.begin() as conn:
                delete_query = text(f"DELETE FROM stock_ratio_data WHERE ticker = :ticker")
                conn.execute(delete_query, {'ticker': ticker})
            
            # Insert new data
            ratio_df.to_sql(
                name='stock_ratio_data',
                con=self.db_con,
                if_exists='append',
                index=False
            )
            
            result['status'] = 'migrated'
            result['rows_updated'] = len(ratio_df)
            self._log(f"   ✅ Migrated {len(ratio_df)} rows using {result['data_source']} data")
            
        except Exception as e:
            result['status'] = 'error'
            result['errors'].append(str(e))
            self._log(f"   ❌ Error: {e}")
        
        return result
    
    def generate_validation_report(self, results: List[Dict[str, Any]]) -> str:
        """
        Generate a summary validation report.
        
        Args:
            results: List of validation results
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 70)
        report.append("TTM MIGRATION VALIDATION REPORT")
        report.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Margin of Error: {self.margin_of_error:.0%}")
        report.append("=" * 70)
        
        # Summary statistics
        total = len(results)
        valid = sum(1 for r in results if r.get('overall_valid', False))
        needs_review = sum(1 for r in results if r.get('status') == 'needs_review')
        errors = sum(1 for r in results if r.get('status') == 'error')
        ttm_sources = sum(1 for r in results if r.get('data_source') == 'ttm')
        annual_sources = sum(1 for r in results if r.get('data_source') == 'annual')
        
        report.append(f"\n📊 SUMMARY")
        report.append(f"   Total tickers validated: {total}")
        report.append(f"   ✅ Passed validation: {valid} ({valid/max(1,total):.1%})")
        report.append(f"   ⚠️ Needs review: {needs_review} ({needs_review/max(1,total):.1%})")
        report.append(f"   ❌ Errors: {errors} ({errors/max(1,total):.1%})")
        report.append(f"\n📈 DATA SOURCES")
        report.append(f"   TTM (quarterly): {ttm_sources} ({ttm_sources/max(1,total):.1%})")
        report.append(f"   Annual (fallback): {annual_sources} ({annual_sources/max(1,total):.1%})")
        
        # Tickers needing review
        review_tickers = [r for r in results if r.get('status') == 'needs_review']
        if review_tickers:
            report.append(f"\n⚠️ TICKERS NEEDING REVIEW ({len(review_tickers)})")
            for r in review_tickers[:10]:  # Show first 10
                warnings = ', '.join(r.get('warnings', [])[:2])
                report.append(f"   {r['ticker']}: {warnings}")
            if len(review_tickers) > 10:
                report.append(f"   ... and {len(review_tickers) - 10} more")
        
        # Error tickers
        error_tickers = [r for r in results if r.get('status') == 'error']
        if error_tickers:
            report.append(f"\n❌ TICKERS WITH ERRORS ({len(error_tickers)})")
            for r in error_tickers[:5]:
                errors = ', '.join(r.get('errors', [])[:1])
                report.append(f"   {r['ticker']}: {errors}")
        
        # Ratio-specific summary
        report.append(f"\n📉 RATIO COMPARISON SUMMARY")
        report.append(f"   Note: P/FCF is expected to differ significantly (TTM vs annual methodology)")
        ratio_stats = {col: {'within': 0, 'outside': 0, 'total': 0} for col in self.RATIO_COLUMNS}
        
        for r in results:
            for col in self.RATIO_COLUMNS:
                comp = r.get('ratio_comparisons', {}).get(col, {})
                ratio_stats[col]['within'] += comp.get('within_margin', 0)
                ratio_stats[col]['outside'] += comp.get('outside_margin', 0)
                ratio_stats[col]['total'] += comp.get('matched_count', 0)
        
        for col, stats in ratio_stats.items():
            display_name = self.RATIO_DISPLAY_NAMES.get(col, col)
            total_comp = stats['total']
            within = stats['within']
            pct = within / max(1, total_comp)
            # Mark P/FCF differently since differences are expected
            marker = " (expected)" if col == 'p_fcf' else ""
            report.append(f"   {display_name}: {within}/{total_comp} ({pct:.1%}) within margin{marker}")
        
        report.append("\n" + "=" * 70)
        return '\n'.join(report)
    
    def save_report(self, report: str, filename: str = 'ttm_migration_report.txt'):
        """Save validation report to file."""
        filepath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            filename
        )
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        self._log(f"\n📄 Report saved to: {filepath}")


def main():
    """Main entry point for the migration script."""
    parser = argparse.ArgumentParser(
        description='Migrate stock ratio data from annual to TTM calculations'
    )
    parser.add_argument(
        '--validate-only', '-v',
        action='store_true',
        help='Only validate without migrating'
    )
    parser.add_argument(
        '--migrate', '-m',
        action='store_true',
        help='Perform actual migration (writes to database)'
    )
    parser.add_argument(
        '--ticker', '-t',
        type=str,
        help='Process only specified ticker'
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force migration even if validation fails'
    )
    parser.add_argument(
        '--margin',
        type=float,
        default=0.15,
        help='Margin of error for validation (default: 0.15 = 15%%)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress messages'
    )
    
    args = parser.parse_args()
    
    # Default to validate-only if neither specified
    if not args.migrate and not args.validate_only:
        args.validate_only = True
    
    print("\n" + "=" * 70)
    print("TTM DATA MIGRATION SCRIPT")
    print(f"Mode: {'Validate Only' if args.validate_only else 'Migrate'}")
    print(f"Margin of Error: {args.margin:.0%}")
    print("=" * 70)
    
    validator = TTMMigrationValidator(
        margin_of_error=args.margin,
        verbose=not args.quiet
    )
    
    if args.ticker:
        # Single ticker
        if args.validate_only:
            result = validator.validate_ticker(args.ticker)
        else:
            result = validator.migrate_ticker(
                args.ticker, 
                dry_run=False, 
                force=args.force
            )
        results = [result]
    else:
        # All tickers
        if args.validate_only:
            results = validator.validate_all_tickers()
        else:
            tickers = validator.get_all_tickers_with_ratio_data()
            results = []
            for ticker in tickers:
                result = validator.migrate_ticker(ticker, dry_run=False, force=args.force)
                results.append(result)
    
    # Generate and save report
    report = validator.generate_validation_report(results)
    print(report)
    validator.save_report(report)
    
    # Return exit code based on results
    errors = sum(1 for r in results if r.get('status') == 'error')
    sys.exit(1 if errors > 0 else 0)


if __name__ == "__main__":
    main()
