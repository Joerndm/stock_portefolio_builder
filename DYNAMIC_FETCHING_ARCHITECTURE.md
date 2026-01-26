# Dynamic Stock Data Fetching - Architecture Recommendations

## Table of Contents
1. [Current State Analysis](#current-state-analysis)
2. [Data Source Comparison](#data-source-comparison)
3. [Recommended Architecture](#recommended-architecture)
4. [Module Splitting - Pros & Cons](#module-splitting---pros--cons)
5. [Async Fetching - Pros & Cons](#async-fetching---pros--cons)
6. [Database Schema Updates for TTM](#database-schema-updates-for-ttm)
7. [Implementation Roadmap](#implementation-roadmap)

---

## Current State Analysis

### Current Features Being Fetched/Calculated

| Category | Features | Source |
|----------|----------|--------|
| **Price Data** | open, high, low, close, volume | yfinance |
| **Returns** | 1D, 1M, 3M, 6M, 9M, 1Y, 2Y, 3Y, 4Y, 5Y | Calculated |
| **Moving Averages** | SMA/EMA (5, 20, 40, 120, 200) | Calculated |
| **Technical Indicators** | RSI-14, ATR-14, MACD, Bollinger Bands | Calculated (pandas_ta) |
| **Volume Indicators** | Volume SMA/EMA, VWAP, OBV, Volume Ratio | Calculated |
| **Volatility** | 5d, 20d, 60d volatility | Calculated |
| **Momentum** | Price momentum | Calculated |
| **Income Statement** | Revenue, Gross Profit, Operating Income, Net Income, EPS + growth rates | yfinance (annual) |
| **Balance Sheet** | Assets, Liabilities, Equity, ROA, ROE, Current/Quick Ratio, D/E | yfinance (annual) |
| **Cash Flow** | FCF, FCF per share + growth rates | yfinance (annual) |
| **Valuation Ratios** | P/S, P/E, P/B, P/FCF | Calculated |

### Current Limitations
- **Symbol sourcing**: Static CSV files
- **Financial data**: Only 3-4 years of annual reports
- **No TTM calculations**: Uses point-in-time annual data
- **Single data source**: yfinance only (Yahoo Finance)

---

## Data Source Comparison

### Free Solutions

| Provider | Price Data | Fundamentals | History | Rate Limits | Best For |
|----------|------------|--------------|---------|-------------|----------|
| **yfinance** | ✅ Excellent | ✅ 3-4 years | 15+ years | ~2000/hour | Price data, basic fundamentals |
| **Financial Modeling Prep (Free)** | ✅ Good | ✅ 10+ years quarterly | 30+ years | 250/day | Long-term fundamentals |
| **Alpha Vantage (Free)** | ✅ Good | ⚠️ Limited | 20+ years | 5/min, 500/day | Basic price data |
| **Twelve Data (Free)** | ✅ Good | ⚠️ Basic | 30+ years | 800/day | Global price data |
| **Wikipedia** | ❌ No | ❌ No | N/A | Unlimited | Index constituents |
| **OpenFIGI** | ❌ No | ❌ No | N/A | Unlimited | Symbol lookup/mapping |

### Paid Solutions (Future)

| Provider | Cost | Highlights |
|----------|------|------------|
| **Financial Modeling Prep Starter** | $15/mo | 300k API calls, quarterly data |
| **Polygon.io Starter** | $29/mo | Real-time, excellent API |
| **Tiingo** | $10/mo | Good European coverage |
| **EOD Historical Data** | $20/mo | All global exchanges |
| **IEX Cloud** | $9/mo | US-focused, reliable |
| **Quandl/Nasdaq** | $50+/mo | Premium fundamentals |

### Recommendation for Your Use Case

**Immediate (Free):**
1. Keep yfinance for price data (excellent, no rate limits for daily use)
2. Use yfinance quarterly financials with TTM calculations (built into the new module)
3. Use Wikipedia scraping for index constituents (built into dynamic_index_fetcher.py)

**Future Enhancement ($15/mo):**
- Add Financial Modeling Prep for 10+ years of quarterly fundamentals

---

## Recommended Architecture

### New Modular Structure

```
stock_portefolio_builder/
├── data_sources/                    # NEW: Data source modules
│   ├── __init__.py
│   ├── base_fetcher.py              # Abstract base class
│   ├── yfinance_fetcher.py          # yfinance implementation
│   ├── fmp_fetcher.py               # Financial Modeling Prep (optional)
│   └── alpha_vantage_fetcher.py     # Alpha Vantage (optional)
│
├── symbol_discovery/                # NEW: Symbol/index management
│   ├── __init__.py
│   ├── dynamic_index_fetcher.py     # CREATED: Fetches index constituents
│   ├── symbol_validator.py          # Validates symbols exist
│   └── exchange_mapper.py           # Maps symbols to exchanges
│
├── feature_calculation/             # NEW: Technical calculations
│   ├── __init__.py
│   ├── technical_indicators.py      # RSI, MACD, ATR, etc.
│   ├── moving_averages.py           # SMA, EMA, Bollinger
│   ├── volume_indicators.py         # VWAP, OBV, Volume ratio
│   ├── volatility_metrics.py        # Volatility calculations
│   ├── period_returns.py            # Return calculations
│   └── financial_ratios.py          # P/E, P/B, P/S, P/FCF
│
├── financial_analysis/              # NEW: Fundamental analysis
│   ├── __init__.py
│   ├── enhanced_financial_fetcher.py # CREATED: Quarterly + TTM
│   ├── ttm_calculator.py            # TTM calculations
│   └── industry_handlers.py         # Bank/Insurance specific logic
│
├── database/                        # Existing, reorganized
│   ├── __init__.py
│   ├── db_connectors.py             # Existing
│   ├── db_interactions.py           # Existing
│   └── schema/
│       ├── ddl.sql                  # Existing
│       └── ddl_quarterly.sql        # NEW: Quarterly tables
│
├── orchestration/                   # NEW: Main workflows
│   ├── __init__.py
│   ├── daily_update.py              # Daily price updates
│   ├── quarterly_update.py          # Quarterly financial updates
│   └── full_refresh.py              # Complete data refresh
│
└── stock_data_fetch.py              # Existing (gradually refactor)
```

### Migration Path

1. **Phase 1** (Immediate): Use new modules alongside existing code
   - `dynamic_index_fetcher.py` replaces CSV imports
   - `enhanced_financial_fetcher.py` adds quarterly/TTM support

2. **Phase 2** (Gradual): Extract functions into separate modules
   - Keep `stock_data_fetch.py` as orchestration layer
   - Move calculations to `feature_calculation/` modules

3. **Phase 3** (Complete): Full modular architecture
   - Each module independently testable
   - Easy to swap data sources

---

## Module Splitting - Pros & Cons

### Pros of Splitting `stock_data_fetch.py`

| Benefit | Description |
|---------|-------------|
| **Testability** | Each module can be unit tested independently |
| **Maintainability** | Easier to find and fix bugs in focused modules |
| **Reusability** | Technical indicators can be used elsewhere |
| **Parallel Development** | Multiple developers can work on different modules |
| **Clearer Dependencies** | Explicit imports show what each component needs |
| **Single Responsibility** | Each module does one thing well |
| **Easier Debugging** | Isolated failures are easier to trace |
| **Documentation** | Smaller modules are easier to document |

### Cons of Splitting

| Drawback | Description | Mitigation |
|----------|-------------|------------|
| **Import Complexity** | More files to manage | Use `__init__.py` to expose clean API |
| **Learning Curve** | New contributors need to understand structure | Clear documentation |
| **Cross-module State** | Sharing data between modules | Use DataFrames as standard interface |
| **Overhead** | More function calls | Negligible performance impact |
| **Initial Effort** | Time to refactor | Gradual migration approach |

### Recommendation

**Split gradually** into these core modules:
1. `technical_indicators.py` - All pandas_ta calculations
2. `financial_ratios.py` - P/E, P/B, P/S, P/FCF
3. `period_returns.py` - Return calculations
4. `symbol_discovery.py` - Dynamic index fetching

Keep `stock_data_fetch.py` as the orchestration layer that calls these modules.

---

## Async Fetching - Pros & Cons

### Pros of Async Fetching

| Benefit | Impact | Description |
|---------|--------|-------------|
| **Speed** | 5-10x faster | Fetch multiple stocks simultaneously |
| **Efficiency** | Better I/O utilization | Don't wait for network responses |
| **Scalability** | Handle more stocks | 500+ stocks in reasonable time |
| **Resource Usage** | Lower memory peaks | Process data as it arrives |

### Cons of Async Fetching

| Drawback | Severity | Mitigation |
|----------|----------|------------|
| **Rate Limiting** | High | Implement semaphores and delays |
| **Error Handling** | Medium | Use try/except with proper retry logic |
| **Debugging** | Medium | Use structured logging |
| **Code Complexity** | Medium | Use well-tested async patterns |
| **Database Writes** | Medium | Batch writes or async-safe connections |
| **yfinance Limitations** | Medium | yfinance is not fully async-compatible |

### Async Implementation Example

```python
import asyncio
import aiohttp
from typing import List
import pandas as pd

class AsyncStockFetcher:
    def __init__(self, max_concurrent: int = 10, delay_between_requests: float = 0.1):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.delay = delay_between_requests
    
    async def fetch_single_stock(self, session: aiohttp.ClientSession, symbol: str) -> pd.DataFrame:
        async with self.semaphore:
            try:
                # Note: yfinance is synchronous, so we'd need an async wrapper
                # or use a different data source like Financial Modeling Prep
                url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}"
                async with session.get(url) as response:
                    data = await response.json()
                    await asyncio.sleep(self.delay)  # Rate limiting
                    return pd.DataFrame(data)
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
                return pd.DataFrame()
    
    async def fetch_multiple_stocks(self, symbols: List[str]) -> pd.DataFrame:
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_single_stock(session, s) for s in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return pd.concat([r for r in results if isinstance(r, pd.DataFrame)], ignore_index=True)

# Usage
# asyncio.run(fetcher.fetch_multiple_stocks(['AAPL', 'GOOGL', 'MSFT']))
```

### Recommendation

**Implement async for:**
- Fetching price data (high volume, network-bound)
- Validating symbols (many small requests)
- Fetching index constituents (multiple sources)

**Keep synchronous for:**
- Technical indicator calculations (CPU-bound)
- Database writes (needs transaction safety)
- Financial statement processing (complex logic)

---

## Database Schema Updates for TTM

### New Tables for Quarterly Data

```sql
-- New quarterly income statement table
CREATE TABLE `stock_income_stmt_quarterly` (
  `fiscal_quarter_end` DATE NOT NULL,
  `fiscal_year` INT,
  `fiscal_quarter` INT,  -- 1, 2, 3, or 4
  `ticker` VARCHAR(255) NOT NULL,
  `revenue` FLOAT,
  `gross_profit` FLOAT,
  `operating_income` FLOAT,
  `net_income` FLOAT,
  `eps_basic` FLOAT,
  `eps_diluted` FLOAT,
  `shares_diluted` FLOAT,
  `ebitda` FLOAT,
  -- TTM calculated values
  `revenue_ttm` FLOAT,
  `gross_profit_ttm` FLOAT,
  `operating_income_ttm` FLOAT,
  `net_income_ttm` FLOAT,
  `eps_ttm` FLOAT,
  `ebitda_ttm` FLOAT,
  -- Margins (TTM)
  `gross_margin_ttm` FLOAT,
  `operating_margin_ttm` FLOAT,
  `net_margin_ttm` FLOAT,
  -- Growth rates (YoY)
  `revenue_growth_yoy` FLOAT,
  `net_income_growth_yoy` FLOAT,
  `eps_growth_yoy` FLOAT,
  PRIMARY KEY (`fiscal_quarter_end`, `ticker`),
  FOREIGN KEY (`ticker`) REFERENCES `stock_info_data`(`ticker`)
);

-- New quarterly balance sheet table
CREATE TABLE `stock_balancesheet_quarterly` (
  `fiscal_quarter_end` DATE NOT NULL,
  `ticker` VARCHAR(255) NOT NULL,
  `total_assets` FLOAT,
  `total_liabilities` FLOAT,
  `total_equity` FLOAT,
  `current_assets` FLOAT,
  `current_liabilities` FLOAT,
  `cash_and_equivalents` FLOAT,
  `total_debt` FLOAT,
  `inventory` FLOAT,
  -- Ratios (point-in-time)
  `current_ratio` FLOAT,
  `quick_ratio` FLOAT,
  `debt_to_equity` FLOAT,
  -- Per share metrics
  `book_value_per_share` FLOAT,
  -- Returns (using TTM income)
  `roa_ttm` FLOAT,
  `roe_ttm` FLOAT,
  `roic_ttm` FLOAT,
  PRIMARY KEY (`fiscal_quarter_end`, `ticker`),
  FOREIGN KEY (`ticker`) REFERENCES `stock_info_data`(`ticker`)
);

-- New quarterly cash flow table
CREATE TABLE `stock_cashflow_quarterly` (
  `fiscal_quarter_end` DATE NOT NULL,
  `ticker` VARCHAR(255) NOT NULL,
  `operating_cash_flow` FLOAT,
  `capex` FLOAT,
  `free_cash_flow` FLOAT,
  -- TTM values
  `operating_cash_flow_ttm` FLOAT,
  `free_cash_flow_ttm` FLOAT,
  `fcf_per_share_ttm` FLOAT,
  -- Growth
  `fcf_growth_yoy` FLOAT,
  PRIMARY KEY (`fiscal_quarter_end`, `ticker`),
  FOREIGN KEY (`ticker`) REFERENCES `stock_info_data`(`ticker`)
);

-- Extended ratio table with TTM support
CREATE TABLE `stock_ratio_data_ttm` (
  `date` DATE NOT NULL,
  `ticker` VARCHAR(255) NOT NULL,
  `p_s_ttm` FLOAT,
  `p_e_ttm` FLOAT,
  `p_b` FLOAT,
  `p_fcf_ttm` FLOAT,
  `ev_ebitda_ttm` FLOAT,
  `ev_revenue_ttm` FLOAT,
  `peg_ratio` FLOAT,  -- P/E / EPS growth
  PRIMARY KEY (`date`, `ticker`),
  FOREIGN KEY (`ticker`) REFERENCES `stock_info_data`(`ticker`)
);
```

---

## Implementation Roadmap

### Phase 1: Immediate (1-2 weeks)
- [x] Create `dynamic_index_fetcher.py` for index constituents
- [x] Create `enhanced_financial_fetcher.py` for quarterly/TTM data
- [ ] Test with C25 and S&P 500 indices
- [ ] Update `stock_data_fetch.py` to use new modules

### Phase 2: Database Update (1 week)
- [ ] Create new quarterly tables (above DDL)
- [ ] Update `db_interactions.py` for quarterly data
- [ ] Add migration script for existing data

### Phase 3: Feature Calculation Refactor (2 weeks)
- [ ] Extract technical indicators to separate module
- [ ] Extract volume indicators to separate module
- [ ] Extract volatility calculations to separate module
- [ ] Update tests

### Phase 4: Async Implementation (2 weeks)
- [ ] Add async price data fetching
- [ ] Implement rate limiting
- [ ] Add retry logic
- [ ] Update orchestration layer

### Phase 5: Additional Data Sources (Ongoing)
- [ ] Add Financial Modeling Prep integration
- [ ] Add fallback data sources
- [ ] Implement data quality checks

---

## Quick Start Guide

### Using the New Dynamic Index Fetcher

```python
from dynamic_index_fetcher import dynamic_fetch_index_data

# Fetch C25 (Denmark) and S&P 500 (US) with market indices
symbols_df = dynamic_fetch_index_data(
    indices=['C25', 'SP500', 'DAX40'],  # Add more indices as needed
    include_market_indices=True,         # Adds ^VIX, ^GSPC, etc.
    export_csv=True                      # Save to dynamic_symbol_list.csv
)

# Get the symbol list for processing
stock_tickers_list = symbols_df["Symbol"].tolist()

# Available indices:
# Denmark: C25
# USA: SP500, NASDAQ100, DOW30
# Germany: DAX40
# France: CAC40
# UK: FTSE100
# Netherlands: AEX25
# Sweden: OMX30
# Spain: IBEX35
# Finland: OMXH25
# Switzerland: SMI
```

### Using the Enhanced Financial Fetcher

```python
from enhanced_financial_fetcher import fetch_quarterly_financial_data

# Fetch 10 years of quarterly data with TTM calculations
data = fetch_quarterly_financial_data('AAPL', years=10)

# Access different data components
income_quarterly = data['income_quarterly']
income_ttm = data['income_ttm']
balance_sheet = data['balance_sheet']
cash_flow_ttm = data['cash_flow_ttm']
ratios = data['ratios']

# Convert to format compatible with existing database
from enhanced_financial_fetcher import convert_quarterly_to_annual_compatible
annual_format = convert_quarterly_to_annual_compatible(data)
```

---

## Notes on External Data Sources

### Euroinvestor, Nordnet, Børsen
These Danish financial sites typically require:
- User authentication (login required)
- Web scraping (against ToS usually)
- No official API

**Recommendation**: Not recommended for automated data fetching. Use for manual verification only.

### Google Finance
- No official API
- Web scraping is fragile and against ToS
- Limited historical data

**Recommendation**: Use only as reference, not as data source.

### Best Free Approach
1. **Index constituents**: Wikipedia (reliable, updated regularly)
2. **Price data**: yfinance (15+ years, excellent coverage)
3. **Fundamentals**: yfinance quarterly + custom TTM calculations
4. **Backup**: Financial Modeling Prep free tier (250 requests/day)
