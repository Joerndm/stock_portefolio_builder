# TTM Data Migration Guide

## Overview

This document describes the transition from annual financial data to TTM (Trailing Twelve Months) data for financial ratio calculations. The transition enables more current financial metrics while maintaining backward compatibility with existing annual-based data.

## What Changed

### Data Calculation Logic

| Feature | Before | After |
|---------|--------|-------|
| **P/E Ratio** | Annual EPS | TTM EPS (sum of last 4 quarters) or Annual fallback |
| **P/S Ratio** | Annual Revenue | TTM Revenue or Annual fallback |
| **P/B Ratio** | Latest Book Value | Latest Quarterly Book Value or Annual |
| **P/FCF Ratio** | Annual Free Cash Flow | TTM Free Cash Flow or Annual fallback |
| **Data Freshness** | ~12 months lag | ~3 months lag with TTM |

### Decision Logic

```
If quarterly reports >= 4:
    → Use TTM calculations (preferred)
Else:
    → Fall back to annual data
```

## New Files Created

### 1. `ttm_financial_calculator.py`
Core module for TTM calculations with annual fallback.

**Key Classes:**
- `TTMFinancialCalculator`: Main calculator class

**Key Functions:**
- `calculate_ratios_with_source_tracking()`: Calculate ratios with metadata
- `get_best_available_financials()`: Get TTM or annual data based on availability
- `validate_ttm_vs_annual()`: Compare TTM vs annual for validation

**Usage:**
```python
from ttm_financial_calculator import TTMFinancialCalculator

calculator = TTMFinancialCalculator()

# Check if TTM is possible
can_use, quarters = calculator.can_use_ttm('AAPL')

# Get best available data
fin_data = calculator.get_best_available_financials('AAPL', prefer_ttm=True)
print(f"Data source: {fin_data['source']}")  # 'ttm' or 'annual'
```

### 2. `ml_data_integrity.py`
Ensures data integrity for ML pipeline with TTM/annual handling.

**Key Classes:**
- `MLDataIntegrityChecker`: Validates and prepares ML datasets

**Key Functions:**
- `prepare_ml_dataset_with_integrity_check()`: Recommended entry point for ML data
- `validate_ratio_data_consistency()`: Check TTM vs annual consistency

**Usage:**
```python
from ml_data_integrity import prepare_ml_dataset_with_integrity_check

df, metadata = prepare_ml_dataset_with_integrity_check('AAPL', prefer_ttm=True)
print(f"Ratio source: {metadata['ratio_source']}")
print(f"Rows: {metadata['rows_final']}")
```

### 3. `test_reports/migrate_to_ttm_data.py`
Migration and validation script for historical data.

**Usage:**
```bash
# Validate only (dry run)
python test_reports/migrate_to_ttm_data.py --validate-only

# Migrate all tickers
python test_reports/migrate_to_ttm_data.py --migrate

# Migrate single ticker
python test_reports/migrate_to_ttm_data.py --migrate --ticker AAPL

# Force migration even if validation fails
python test_reports/migrate_to_ttm_data.py --migrate --force

# Custom margin of error (default: 15%)
python test_reports/migrate_to_ttm_data.py --validate-only --margin 0.20
```

## Modified Files

### 1. `stock_data_fetch.py`

**Changes:**
- Added import for TTM calculator
- Updated `calculate_ratios()` function to use TTM with fallback

**New Function Signature:**
```python
def calculate_ratios(combined_stock_data_df, stock_symbol=None, prefer_ttm=True):
    """
    Calculates P/S, P/E, P/B and P/FCF ratios using TTM data when available.
    
    Returns DataFrame with ratios and source metadata:
    - ratio_data_source: 'ttm' or 'annual'
    - quarters_available: Number of quarterly reports available
    """
```

### 2. `db_interactions.py`

**New Functions:**
- `export_stock_ratio_data_ttm()`: Export TTM ratios to database
- `import_stock_ratio_data_ttm()`: Import TTM ratios from database
- `does_ttm_ratio_data_exist()`: Check if TTM data exists
- `export_quarterly_income_stmt()`: Export quarterly income statement
- `export_quarterly_balancesheet()`: Export quarterly balance sheet
- `export_quarterly_cashflow()`: Export quarterly cash flow
- `import_stock_dataset_with_ttm()`: Import dataset preferring TTM ratios

## Database Schema Changes

New tables added (see `database_files/ddl_quarterly.sql`):
- `stock_income_stmt_quarterly`
- `stock_balancesheet_quarterly`
- `stock_cashflow_quarterly`
- `stock_ratio_data_ttm`
- `index_membership`

## Migration Steps

### Step 1: Run Database Migration
```sql
SOURCE database_files/migrate_add_quarterly_tables.sql;
```

### Step 2: Validate Existing Data
```bash
python test_reports/migrate_to_ttm_data.py --validate-only
```

This generates a validation report showing:
- Which tickers have sufficient quarterly data for TTM
- Comparison of TTM vs annual values
- Tickers needing review

### Step 3: Review Validation Report
Check `test_reports/ttm_migration_report.txt` for:
- **Passed validation**: TTM within 15% margin of annual
- **Needs review**: Significant differences (may indicate corrections between quarterly/annual)
- **Errors**: Issues that need manual investigation

### Step 4: Migrate Data
```bash
# Migrate all valid tickers
python test_reports/migrate_to_ttm_data.py --migrate

# Or migrate with force flag for tickers outside margin
python test_reports/migrate_to_ttm_data.py --migrate --force
```

### Step 5: Verify ML Pipeline
```python
from ml_data_integrity import prepare_ml_dataset_with_integrity_check

# Test with a sample ticker
df, metadata = prepare_ml_dataset_with_integrity_check('DEMANT.CO')

# Verify ratio source
assert metadata['ratio_source'] in ['ttm', 'annual']

# Verify data quality
assert metadata['rows_final'] > 252  # At least 1 year of data
```

## Margin of Error Explanation

A 15% margin of error is acceptable because:

1. **Timing Differences**: Quarterly reports sum to TTM, but annual reports may have fiscal year adjustments
2. **Restatements**: Companies sometimes restate quarterly figures when filing annual reports
3. **Currency/Rounding**: Minor differences in currency conversion or rounding
4. **Methodology**: Different calculation methods for certain metrics

Values outside 15% should be reviewed but aren't necessarily wrong - they may indicate:
- Significant quarterly restatements
- One-time adjustments in annual reports
- Different accounting treatments

## ML Pipeline Integration

### Before (Annual Only)
```python
stock_data_df = db_interactions.import_stock_dataset(ticker)
```

### After (TTM with Fallback)
```python
# Option 1: Direct import with TTM preference
stock_data_df = db_interactions.import_stock_dataset_with_ttm(ticker, prefer_ttm=True)

# Option 2: With integrity checks (recommended)
from ml_data_integrity import prepare_ml_dataset_with_integrity_check
stock_data_df, metadata = prepare_ml_dataset_with_integrity_check(ticker)
```

### Feature Tracking

The ML pipeline now tracks data source through these columns:
- `ratio_data_source`: 'ttm' or 'annual'
- `quarters_available`: Number of quarters used (0 if annual)
- `ratio_source`: Same as ratio_data_source (for backward compatibility)

## Troubleshooting

### "No TTM data available"
- Check if `stock_income_stmt_quarterly` table has data
- Verify at least 4 quarters of data exist for the ticker
- Run quarterly data fetch: `enhanced_financial_fetcher.py`

### "TTM values differ significantly from annual"
- Normal for companies with recent restatements
- Check company press releases for explanations
- Use `--force` flag if differences are understood

### "Missing ratio features"
- The system will attempt to calculate from financial statements
- Check that `stock_income_stmt_data` or quarterly tables have data
- Verify `fetch_stock_financial_data()` succeeded for the ticker

### "Import error: module not found"
- Ensure all new files are in the project root
- Run: `pip install -r requirements_PY_3_12.txt`

## Rollback

If issues arise, you can revert to annual-only data:

```python
# Force annual data import
stock_data_df = db_interactions.import_stock_dataset(ticker)  # Original function

# Or explicitly disable TTM
stock_data_df = db_interactions.import_stock_dataset_with_ttm(ticker, prefer_ttm=False)
```

The original `stock_ratio_data` table is preserved and can be used as fallback.

## Support

For issues with the TTM migration:
1. Check the validation report in `test_reports/ttm_migration_report.txt`
2. Review specific ticker with: `python test_reports/migrate_to_ttm_data.py --ticker SYMBOL --validate-only`
3. Examine data quality with `ml_data_integrity.py`
