-- ============================================
-- STOCK PORTFOLIO BUILDER - DATABASE SCHEMA
-- ============================================
-- Complete DDL for stock analysis and portfolio management system
-- 
-- This schema supports:
-- - Stock information and daily price/technical data
-- - Annual financial statements
-- - Quarterly financial statements with TTM (Trailing Twelve Months) calculations
-- - Daily valuation ratios (annual and TTM-based)
-- - ML predictions
-- - Index membership tracking
--
-- Author: Stock Portfolio Builder
-- Last Modified: January 2026
-- ============================================

DROP DATABASE IF EXISTS `stock_portefolio_builder`;

CREATE DATABASE IF NOT EXISTS `stock_portefolio_builder`;

USE `stock_portefolio_builder`;


-- ============================================
-- SECTION 1: STOCK INFORMATION TABLE
-- ============================================
-- Core reference table for stock metadata

CREATE TABLE `stock_info_data` (
  `ticker` VARCHAR(255) NOT NULL COMMENT 'Stock ticker symbol',
  `company_Name` VARCHAR(255) COMMENT 'Company name',
  `industry` VARCHAR(255) COMMENT 'Industry classification',
  PRIMARY KEY (`ticker`),
  UNIQUE (`ticker`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Core stock information and metadata';


-- ============================================
-- SECTION 2: DAILY PRICE & TECHNICAL DATA
-- ============================================
-- Historical daily prices with technical indicators

CREATE TABLE `stock_price_data` (
  `date` DATE NOT NULL COMMENT 'Trading date',
  `ticker` VARCHAR(255) NOT NULL,
  `currency` VARCHAR(255) COMMENT 'Price currency',
  `trade_Volume` BIGINT COMMENT 'Trading volume',
  
  -- OHLC prices
  `open_Price` FLOAT COMMENT 'Opening price',
  `high_Price` FLOAT COMMENT 'Highest price',
  `low_Price` FLOAT COMMENT 'Lowest price',
  `close_Price` FLOAT COMMENT 'Closing price',
  
  -- Price returns
  `1D` FLOAT COMMENT '1-day return',
  `1M` FLOAT COMMENT '1-month return',
  `3M` FLOAT COMMENT '3-month return',
  `6M` FLOAT COMMENT '6-month return',
  `9M` FLOAT COMMENT '9-month return',
  `1Y` FLOAT COMMENT '1-year return',
  `2Y` FLOAT COMMENT '2-year return',
  `3Y` FLOAT COMMENT '3-year return',
  `4Y` FLOAT COMMENT '4-year return',
  `5Y` FLOAT COMMENT '5-year return',
  
  -- Simple moving averages
  `sma_5` FLOAT COMMENT '5-day SMA',
  `sma_20` FLOAT COMMENT '20-day SMA',
  `sma_40` FLOAT COMMENT '40-day SMA',
  `sma_120` FLOAT COMMENT '120-day SMA',
  `sma_200` FLOAT COMMENT '200-day SMA',
  
  -- Exponential moving averages
  `ema_5` FLOAT COMMENT '5-day EMA',
  `ema_20` FLOAT COMMENT '20-day EMA',
  `ema_40` FLOAT COMMENT '40-day EMA',
  `ema_120` FLOAT COMMENT '120-day EMA',
  `ema_200` FLOAT COMMENT '200-day EMA',
  
  -- Standard deviations
  `std_Div_5` FLOAT COMMENT '5-day standard deviation',
  `std_Div_20` FLOAT COMMENT '20-day standard deviation',
  `std_Div_40` FLOAT COMMENT '40-day standard deviation',
  `std_Div_120` FLOAT COMMENT '120-day standard deviation',
  `std_Div_200` FLOAT COMMENT '200-day standard deviation',
  
  -- Bollinger bands
  `bollinger_Band_5_2STD` FLOAT COMMENT '5-day Bollinger Band (2 std)',
  `bollinger_Band_20_2STD` FLOAT COMMENT '20-day Bollinger Band (2 std)',
  `bollinger_Band_40_2STD` FLOAT COMMENT '40-day Bollinger Band (2 std)',
  `bollinger_Band_120_2STD` FLOAT COMMENT '120-day Bollinger Band (2 std)',
  `bollinger_Band_200_2STD` FLOAT COMMENT '200-day Bollinger Band (2 std)',
  
  -- Momentum indicators
  `momentum` FLOAT COMMENT 'Price momentum',
  `rsi_14` FLOAT COMMENT '14-day RSI',
  `atr_14` FLOAT COMMENT '14-day ATR (Average True Range)',
  
  -- MACD indicators
  `macd` FLOAT COMMENT 'MACD line',
  `macd_signal` FLOAT COMMENT 'MACD signal line',
  `macd_histogram` FLOAT COMMENT 'MACD histogram',
  
  -- Volume indicators
  `volume_sma_20` FLOAT COMMENT '20-day volume SMA',
  `volume_ema_20` FLOAT COMMENT '20-day volume EMA',
  `volume_ratio` FLOAT COMMENT 'Volume ratio',
  `vwap` FLOAT COMMENT 'Volume Weighted Average Price',
  `obv` FLOAT COMMENT 'On-Balance Volume',
  
  -- Volatility measures
  `volatility_5d` FLOAT COMMENT '5-day volatility',
  `volatility_20d` FLOAT COMMENT '20-day volatility',
  `volatility_60d` FLOAT COMMENT '60-day volatility',
  
  CONSTRAINT `PK_stock_price` PRIMARY KEY (`date`, `ticker`),
  FOREIGN KEY (`ticker`) REFERENCES `stock_info_data`(`ticker`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Daily price data with technical indicators';


-- ============================================
-- SECTION 3: ANNUAL FINANCIAL STATEMENTS
-- ============================================
-- Annual income statement, balance sheet, and cash flow data

-- Annual Income Statement
CREATE TABLE `stock_income_stmt_data` (
  `financial_Statement_Date` DATE NOT NULL COMMENT 'Fiscal year end date',
  `date_published` DATE COMMENT 'Publication date',
  `ticker` VARCHAR(255) NOT NULL,
  
  -- Revenue metrics
  `revenue` FLOAT COMMENT 'Total revenue',
  `revenue_Growth` FLOAT COMMENT 'YoY revenue growth',
  
  -- Profit metrics
  `gross_Profit` FLOAT COMMENT 'Gross profit',
  `gross_Profit_Growth` FLOAT COMMENT 'YoY gross profit growth',
  `gross_Margin` FLOAT COMMENT 'Gross profit margin',
  `gross_Margin_Growth` FLOAT COMMENT 'YoY gross margin growth',
  
  -- Operating metrics
  `operating_Earning` FLOAT COMMENT 'Operating income/EBIT',
  `operating_Earning_Growth` FLOAT COMMENT 'YoY operating income growth',
  `operating_Earning_Margin` FLOAT COMMENT 'Operating margin',
  `operating_Earning_Margin_Growth` FLOAT COMMENT 'YoY operating margin growth',
  
  -- Net income metrics
  `net_Income` FLOAT COMMENT 'Net income',
  `net_Income_Growth` FLOAT COMMENT 'YoY net income growth',
  `net_Income_Margin` FLOAT COMMENT 'Net profit margin',
  `net_Income_Margin_Growth` FLOAT COMMENT 'YoY net margin growth',
  
  -- Per share metrics
  `eps` FLOAT COMMENT 'Earnings per share',
  `eps_Growth` FLOAT COMMENT 'YoY EPS growth',
  `average_shares` FLOAT COMMENT 'Average shares outstanding',
  
  CONSTRAINT `PK_Stock_income_stmt` PRIMARY KEY (`financial_Statement_Date`, `ticker`),
  FOREIGN KEY (`ticker`) REFERENCES `stock_info_data`(`ticker`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Annual income statement data';

-- Annual Balance Sheet
CREATE TABLE `stock_balancesheet_data` (
  `financial_Statement_Date` DATE NOT NULL COMMENT 'Fiscal year end date',
  `date_Published` DATE COMMENT 'Publication date',
  `ticker` VARCHAR(255) NOT NULL,
  
  -- Asset metrics
  `total_Assets` FLOAT COMMENT 'Total assets',
  `total_Assets_Growth` FLOAT COMMENT 'YoY total assets growth',
  `current_Assets` FLOAT COMMENT 'Current assets',
  `current_Assets_Growth` FLOAT COMMENT 'YoY current assets growth',
  `cash_And_Cash_Equivalents` FLOAT COMMENT 'Cash and equivalents',
  `cash_And_Cash_Equivalents_Growth` FLOAT COMMENT 'YoY cash growth',
  
  -- Equity and liability metrics
  `equity` FLOAT COMMENT 'Total shareholders equity',
  `equity_Growth` FLOAT COMMENT 'YoY equity growth',
  `liabilities` FLOAT COMMENT 'Total liabilities',
  `liabilities_Growth` FLOAT COMMENT 'YoY liabilities growth',
  `current_liabilities` FLOAT COMMENT 'Current liabilities',
  `current_liabilities_Growth` FLOAT COMMENT 'YoY current liabilities growth',
  
  -- Book value metrics
  `book_Value` FLOAT COMMENT 'Book value (total equity)',
  `book_Value_Growth` FLOAT COMMENT 'YoY book value growth',
  `book_Value_Per_Share` FLOAT COMMENT 'Book value per share',
  `book_Value_Per_Share_Growth` FLOAT COMMENT 'YoY BVPS growth',
  
  -- Return ratios
  `return_On_Assets` FLOAT COMMENT 'ROA (Net income / Total assets)',
  `return_On_Assets_Growth` FLOAT COMMENT 'YoY ROA growth',
  `return_On_Equity` FLOAT COMMENT 'ROE (Net income / Equity)',
  `return_On_Equity_Growth` FLOAT COMMENT 'YoY ROE growth',
  
  -- Liquidity ratios
  `current_Ratio` FLOAT COMMENT 'Current assets / Current liabilities',
  `current_Ratio_Growth` FLOAT COMMENT 'YoY current ratio growth',
  `quick_Ratio` FLOAT COMMENT '(Current assets - Inventory) / Current liabilities',
  `quick_Ratio_Growth` FLOAT COMMENT 'YoY quick ratio growth',
  
  -- Leverage ratios
  `debt_To_Equity` FLOAT COMMENT 'Total debt / Total equity',
  `debt_To_Equity_Growth` FLOAT COMMENT 'YoY debt-to-equity growth',
  
  CONSTRAINT `PK_dtock_balancesheet` PRIMARY KEY (`financial_Statement_Date`, `ticker`),
  FOREIGN KEY (`ticker`) REFERENCES `stock_info_data`(`ticker`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Annual balance sheet data';

-- Annual Cash Flow
CREATE TABLE `stock_cash_flow_data` (
  `financial_Statement_Date` DATE NOT NULL COMMENT 'Fiscal year end date',
  `date_published` DATE COMMENT 'Publication date',
  `ticker` VARCHAR(255) NOT NULL,
  
  -- Free cash flow metrics
  `free_Cash_Flow` FLOAT COMMENT 'Free cash flow',
  `free_Cash_Flow_Growth` FLOAT COMMENT 'YoY FCF growth',
  `free_Cash_Flow_Per_Share` FLOAT COMMENT 'FCF per share',
  `free_Cash_Flow_Per_Share_Growth` FLOAT COMMENT 'YoY FCF per share growth',
  
  CONSTRAINT `PK_stock_cash_flow` PRIMARY KEY (`financial_Statement_Date`, `ticker`),
  FOREIGN KEY (`ticker`) REFERENCES `stock_info_data`(`ticker`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Annual cash flow data';


-- ============================================
-- SECTION 4: QUARTERLY FINANCIAL STATEMENTS
-- ============================================
-- Quarterly financials with TTM (Trailing Twelve Months) calculations

-- Quarterly Income Statement
CREATE TABLE `stock_income_stmt_quarterly` (
  `fiscal_quarter_end` DATE NOT NULL COMMENT 'End date of the fiscal quarter',
  `fiscal_year` INT COMMENT 'Fiscal year (e.g., 2026)',
  `fiscal_quarter` INT COMMENT 'Quarter number (1, 2, 3, or 4)',
  `date_published` DATE COMMENT 'Date when the report was published',
  `ticker` VARCHAR(255) NOT NULL,
  
  -- Quarterly values (single quarter)
  `revenue_q` FLOAT COMMENT 'Quarterly revenue',
  `gross_profit_q` FLOAT COMMENT 'Quarterly gross profit',
  `operating_income_q` FLOAT COMMENT 'Quarterly operating income',
  `net_income_q` FLOAT COMMENT 'Quarterly net income',
  `eps_basic_q` FLOAT COMMENT 'Quarterly basic EPS',
  `eps_diluted_q` FLOAT COMMENT 'Quarterly diluted EPS',
  `shares_diluted` FLOAT COMMENT 'Diluted shares outstanding',
  `ebitda_q` FLOAT COMMENT 'Quarterly EBITDA',
  
  -- TTM values (sum of last 4 quarters)
  `revenue_ttm` FLOAT COMMENT 'TTM revenue',
  `gross_profit_ttm` FLOAT COMMENT 'TTM gross profit',
  `operating_income_ttm` FLOAT COMMENT 'TTM operating income',
  `net_income_ttm` FLOAT COMMENT 'TTM net income',
  `eps_basic_ttm` FLOAT COMMENT 'TTM basic EPS',
  `eps_diluted_ttm` FLOAT COMMENT 'TTM diluted EPS',
  `ebitda_ttm` FLOAT COMMENT 'TTM EBITDA',
  
  -- Margins (calculated from TTM values)
  `gross_margin_ttm` FLOAT COMMENT 'Gross profit / Revenue (TTM)',
  `operating_margin_ttm` FLOAT COMMENT 'Operating income / Revenue (TTM)',
  `net_margin_ttm` FLOAT COMMENT 'Net income / Revenue (TTM)',
  `ebitda_margin_ttm` FLOAT COMMENT 'EBITDA / Revenue (TTM)',
  
  -- Growth rates (YoY comparison)
  `revenue_growth_yoy` FLOAT COMMENT 'Revenue growth vs same quarter last year',
  `gross_profit_growth_yoy` FLOAT COMMENT 'Gross profit growth YoY',
  `operating_income_growth_yoy` FLOAT COMMENT 'Operating income growth YoY',
  `net_income_growth_yoy` FLOAT COMMENT 'Net income growth YoY',
  `eps_growth_yoy` FLOAT COMMENT 'EPS growth YoY',
  
  -- Sequential growth (QoQ)
  `revenue_growth_qoq` FLOAT COMMENT 'Revenue growth vs previous quarter',
  `net_income_growth_qoq` FLOAT COMMENT 'Net income growth vs previous quarter',
  
  CONSTRAINT `PK_income_stmt_quarterly` PRIMARY KEY (`fiscal_quarter_end`, `ticker`),
  FOREIGN KEY (`ticker`) REFERENCES `stock_info_data`(`ticker`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Quarterly income statement data with TTM calculations';

-- Quarterly Balance Sheet
CREATE TABLE `stock_balancesheet_quarterly` (
  `fiscal_quarter_end` DATE NOT NULL COMMENT 'End date of the fiscal quarter',
  `fiscal_year` INT COMMENT 'Fiscal year',
  `fiscal_quarter` INT COMMENT 'Quarter number',
  `date_published` DATE COMMENT 'Date when the report was published',
  `ticker` VARCHAR(255) NOT NULL,
  
  -- Asset values
  `total_assets` FLOAT COMMENT 'Total assets',
  `current_assets` FLOAT COMMENT 'Current assets',
  `cash_and_equivalents` FLOAT COMMENT 'Cash and cash equivalents',
  `cash_and_investments` FLOAT COMMENT 'Cash, equivalents, and short-term investments',
  `accounts_receivable` FLOAT COMMENT 'Accounts receivable',
  `inventory` FLOAT COMMENT 'Inventory',
  `goodwill` FLOAT COMMENT 'Goodwill',
  `intangible_assets` FLOAT COMMENT 'Intangible assets',
  
  -- Liability values
  `total_liabilities` FLOAT COMMENT 'Total liabilities',
  `current_liabilities` FLOAT COMMENT 'Current liabilities',
  `accounts_payable` FLOAT COMMENT 'Accounts payable',
  `total_debt` FLOAT COMMENT 'Total debt (short + long term)',
  `short_term_debt` FLOAT COMMENT 'Short-term debt',
  `long_term_debt` FLOAT COMMENT 'Long-term debt',
  
  -- Equity values
  `total_equity` FLOAT COMMENT 'Total shareholders equity',
  `retained_earnings` FLOAT COMMENT 'Retained earnings',
  
  -- Calculated ratios (point-in-time)
  `current_ratio` FLOAT COMMENT 'Current assets / Current liabilities',
  `quick_ratio` FLOAT COMMENT '(Current assets - Inventory) / Current liabilities',
  `cash_ratio` FLOAT COMMENT 'Cash / Current liabilities',
  `debt_to_equity` FLOAT COMMENT 'Total debt / Total equity',
  `debt_to_assets` FLOAT COMMENT 'Total debt / Total assets',
  
  -- Per share metrics
  `book_value_per_share` FLOAT COMMENT 'Total equity / Shares outstanding',
  `tangible_book_per_share` FLOAT COMMENT '(Equity - Intangibles) / Shares',
  
  -- Return ratios (using TTM income)
  `roa_ttm` FLOAT COMMENT 'Net income (TTM) / Total assets',
  `roe_ttm` FLOAT COMMENT 'Net income (TTM) / Total equity',
  `roic_ttm` FLOAT COMMENT 'Net income (TTM) / (Equity + Debt)',
  
  -- Growth rates (YoY)
  `assets_growth_yoy` FLOAT COMMENT 'Total assets growth YoY',
  `equity_growth_yoy` FLOAT COMMENT 'Total equity growth YoY',
  `book_value_growth_yoy` FLOAT COMMENT 'Book value per share growth YoY',
  
  CONSTRAINT `PK_balancesheet_quarterly` PRIMARY KEY (`fiscal_quarter_end`, `ticker`),
  FOREIGN KEY (`ticker`) REFERENCES `stock_info_data`(`ticker`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Quarterly balance sheet data';

-- Quarterly Cash Flow
CREATE TABLE `stock_cashflow_quarterly` (
  `fiscal_quarter_end` DATE NOT NULL COMMENT 'End date of the fiscal quarter',
  `fiscal_year` INT COMMENT 'Fiscal year',
  `fiscal_quarter` INT COMMENT 'Quarter number',
  `date_published` DATE COMMENT 'Date when the report was published',
  `ticker` VARCHAR(255) NOT NULL,
  
  -- Quarterly values
  `operating_cash_flow_q` FLOAT COMMENT 'Quarterly operating cash flow',
  `capex_q` FLOAT COMMENT 'Quarterly capital expenditures',
  `free_cash_flow_q` FLOAT COMMENT 'Quarterly free cash flow (OCF - CapEx)',
  `investing_cash_flow_q` FLOAT COMMENT 'Quarterly investing cash flow',
  `financing_cash_flow_q` FLOAT COMMENT 'Quarterly financing cash flow',
  `dividends_paid_q` FLOAT COMMENT 'Quarterly dividends paid',
  `share_repurchases_q` FLOAT COMMENT 'Quarterly share repurchases',
  
  -- TTM values
  `operating_cash_flow_ttm` FLOAT COMMENT 'TTM operating cash flow',
  `capex_ttm` FLOAT COMMENT 'TTM capital expenditures',
  `free_cash_flow_ttm` FLOAT COMMENT 'TTM free cash flow',
  `dividends_paid_ttm` FLOAT COMMENT 'TTM dividends paid',
  `share_repurchases_ttm` FLOAT COMMENT 'TTM share repurchases',
  
  -- Per share metrics (TTM)
  `fcf_per_share_ttm` FLOAT COMMENT 'TTM FCF / Shares outstanding',
  `ocf_per_share_ttm` FLOAT COMMENT 'TTM OCF / Shares outstanding',
  
  -- Ratios
  `fcf_margin_ttm` FLOAT COMMENT 'TTM FCF / TTM Revenue',
  `capex_to_revenue_ttm` FLOAT COMMENT 'TTM CapEx / TTM Revenue',
  `fcf_conversion_ttm` FLOAT COMMENT 'TTM FCF / TTM Net Income',
  
  -- Growth rates
  `fcf_growth_yoy` FLOAT COMMENT 'Free cash flow growth YoY',
  `ocf_growth_yoy` FLOAT COMMENT 'Operating cash flow growth YoY',
  
  CONSTRAINT `PK_cashflow_quarterly` PRIMARY KEY (`fiscal_quarter_end`, `ticker`),
  FOREIGN KEY (`ticker`) REFERENCES `stock_info_data`(`ticker`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Quarterly cash flow data with TTM calculations';


-- ============================================
-- SECTION 5: VALUATION RATIOS
-- ============================================
-- Daily valuation ratios
-- NOTE: TTM calculations happen at data fetch time (via calculate_ratios_ttm_with_fallback)
--       and are stored in this single table. No separate TTM table needed.

CREATE TABLE `stock_ratio_data` (
  `date` DATE NOT NULL COMMENT 'Trading date',
  `ticker` VARCHAR(255) NOT NULL,
  
  -- Core valuation ratios (calculated using TTM with annual fallback at fetch time)
  `p_s` FLOAT COMMENT 'Price / Sales',
  `p_e` FLOAT COMMENT 'Price / Earnings',
  `p_b` FLOAT COMMENT 'Price / Book value',
  `p_fcf` FLOAT COMMENT 'Price / Free cash flow',
  
  -- Metadata for tracking which financial data was used
  `financial_date_used` DATE COMMENT 'Fiscal period end date of the financial report used for calculation',
  
  CONSTRAINT `PK_stock_ratio_data` PRIMARY KEY (`date`, `ticker`),
  FOREIGN KEY (`ticker`) REFERENCES `stock_info_data`(`ticker`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Daily valuation ratios (TTM preferred, annual fallback)';


-- ============================================
-- SECTION 6: ML PREDICTIONS
-- ============================================
-- Machine learning model predictions

CREATE TABLE `stock_prediction_data` (
  `ticker` VARCHAR(255) NOT NULL,
  `model_Type` VARCHAR(255) COMMENT 'Type of ML model used',
  `model_Equation` VARCHAR(255) COMMENT 'Model equation/description',
  `prediction_30_Days` FLOAT COMMENT '30-day price prediction',
  `prediction_60_Days` FLOAT COMMENT '60-day price prediction',
  `prediction_90_Days` FLOAT COMMENT '90-day price prediction',
  PRIMARY KEY (`ticker`),
  FOREIGN KEY (`ticker`) REFERENCES `stock_info_data`(`ticker`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='ML model predictions';


-- ============================================
-- SECTION 7: INDEX MEMBERSHIP (OPTIONAL)
-- ============================================
-- Track which stocks belong to which indices
-- NOTE: This table is OPTIONAL and not currently used by the ML pipeline.
--       It's provided for future index tracking functionality.
--       You can skip creating this table if not needed.

CREATE TABLE IF NOT EXISTS `index_membership` (
  `ticker` VARCHAR(255) NOT NULL,
  `index_code` VARCHAR(50) NOT NULL COMMENT 'Index code (e.g., C25, SP500, DAX40)',
  `index_name` VARCHAR(255) COMMENT 'Full index name',
  `exchange` VARCHAR(50) COMMENT 'Exchange code',
  `date_added` DATE COMMENT 'Date added to index (if known)',
  `date_removed` DATE COMMENT 'Date removed from index (if applicable)',
  `is_current` BOOLEAN DEFAULT TRUE COMMENT 'Whether currently in index',
  `last_updated` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  
  PRIMARY KEY (`ticker`, `index_code`),
  FOREIGN KEY (`ticker`) REFERENCES `stock_info_data`(`ticker`),
  INDEX `idx_index_code` (`index_code`),
  INDEX `idx_is_current` (`is_current`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Index constituent membership tracking';


-- ============================================
-- SECTION 7B: QUARTERLY FETCH METADATA TABLE
-- ============================================
-- Tracks when quarterly data was last fetched for each ticker
-- Used to implement smart caching and reduce unnecessary API calls

CREATE TABLE IF NOT EXISTS `quarterly_fetch_metadata` (
  `ticker` VARCHAR(255) NOT NULL COMMENT 'Stock ticker symbol',
  `last_fetch_date` DATE NOT NULL COMMENT 'Date when quarterly data was last fetched from API',
  `last_quarter_end` DATE COMMENT 'Most recent fiscal quarter end date in database',
  `quarters_count` INT DEFAULT 0 COMMENT 'Number of quarterly records in database',
  `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Record creation timestamp',
  `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Last update timestamp',
  PRIMARY KEY (`ticker`),
  CONSTRAINT `fk_quarterly_fetch_metadata_ticker` 
    FOREIGN KEY (`ticker`) REFERENCES `stock_info_data` (`ticker`)
    ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Tracks quarterly data fetch timestamps for smart caching';


-- ============================================
-- SECTION 8: VIEWS FOR COMMON QUERIES
-- ============================================

-- View: Latest quarterly financials for each stock
CREATE OR REPLACE VIEW `v_latest_quarterly_financials` AS
SELECT 
    i.ticker,
    i.fiscal_quarter_end,
    i.revenue_ttm,
    i.net_income_ttm,
    i.gross_margin_ttm,
    i.operating_margin_ttm,
    i.net_margin_ttm,
    i.revenue_growth_yoy,
    i.eps_diluted_ttm,
    b.total_assets,
    b.total_equity,
    b.current_ratio,
    b.debt_to_equity,
    b.roe_ttm,
    b.roa_ttm,
    c.free_cash_flow_ttm,
    c.fcf_per_share_ttm
FROM stock_income_stmt_quarterly i
LEFT JOIN stock_balancesheet_quarterly b ON i.ticker = b.ticker AND i.fiscal_quarter_end = b.fiscal_quarter_end
LEFT JOIN stock_cashflow_quarterly c ON i.ticker = c.ticker AND i.fiscal_quarter_end = c.fiscal_quarter_end
WHERE i.fiscal_quarter_end = (
    SELECT MAX(fiscal_quarter_end) 
    FROM stock_income_stmt_quarterly 
    WHERE ticker = i.ticker
);

-- View: Latest valuation ratios for each stock
CREATE OR REPLACE VIEW `v_latest_valuations` AS
SELECT r.*
FROM stock_ratio_data r
WHERE r.date = (
    SELECT MAX(date) 
    FROM stock_ratio_data 
    WHERE ticker = r.ticker
);

-- View: Current index members
CREATE OR REPLACE VIEW `v_current_index_members` AS
SELECT 
    im.index_code,
    im.index_name,
    im.ticker,
    si.company_Name,
    si.industry,
    im.exchange
FROM index_membership im
JOIN stock_info_data si ON im.ticker = si.ticker
WHERE im.is_current = TRUE
ORDER BY im.index_code, si.company_Name;


-- ============================================
-- SCHEMA DOCUMENTATION
-- ============================================
--
-- TABLE SUMMARY:
--
-- Core Tables:
--   - stock_info_data: Stock metadata (parent table)
--   - stock_price_data: Daily OHLC prices + 50+ technical indicators
--
-- Annual Financials:
--   - stock_income_stmt_data: Annual income statements
--   - stock_balancesheet_data: Annual balance sheets  
--   - stock_cash_flow_data: Annual cash flows
--
-- Quarterly Financials:
--   - stock_income_stmt_quarterly: Quarterly + TTM income data
--   - stock_balancesheet_quarterly: Quarterly balance sheets
--   - stock_cashflow_quarterly: Quarterly + TTM cash flows
--
-- Valuation Ratios:
--   - stock_ratio_data: Daily ratios (TTM preferred, annual fallback)
--     NOTE: TTM calculation happens at fetch time via calculate_ratios_ttm_with_fallback()
--
-- ML/Analytics:
--   - stock_prediction_data: ML model predictions
--
-- Optional Tables (not used by ML pipeline):
--   - index_membership: Index constituent tracking (for future use)
--
-- Useful Views:
--   - v_latest_quarterly_financials: Latest financial metrics per stock
--   - v_latest_valuations: Latest valuation ratios per stock
--   - v_current_index_members: Current index constituents (requires index_membership)
--
-- RATIO CALCULATION NOTES:
--   - Ratios are calculated at data fetch time, not stored separately for TTM/annual
--   - calculate_ratios_ttm_with_fallback() uses TTM if 4+ quarters available
--   - Falls back to annual data automatically when insufficient quarterly data
--   - This approach keeps the schema simple with one ratio table
--
-- ============================================
