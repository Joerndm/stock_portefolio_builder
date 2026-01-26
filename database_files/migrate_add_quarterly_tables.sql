-- Migration Script: Add Quarterly Tables to Existing Database
-- This script safely adds the new quarterly/TTM tables while preserving existing data
--
-- Run this ONCE to upgrade your database schema
-- Safe to run multiple times (uses IF NOT EXISTS)
--
-- Author: Stock Portfolio Builder
-- Date: 2026

USE `stock_portefolio_builder`;

-- ============================================
-- STEP 1: Verify existing tables are intact
-- ============================================
-- This just checks they exist - no modifications

SELECT 'Checking existing tables...' AS status;

SELECT 
    TABLE_NAME,
    TABLE_ROWS
FROM information_schema.TABLES 
WHERE TABLE_SCHEMA = 'stock_portefolio_builder'
ORDER BY TABLE_NAME;

-- ============================================
-- STEP 2: Add new quarterly tables
-- ============================================

-- Quarterly Income Statement
CREATE TABLE IF NOT EXISTS `stock_income_stmt_quarterly` (
  `fiscal_quarter_end` DATE NOT NULL,
  `fiscal_year` INT,
  `fiscal_quarter` INT,
  `date_published` DATE,
  `ticker` VARCHAR(255) NOT NULL,
  `revenue_q` FLOAT,
  `gross_profit_q` FLOAT,
  `operating_income_q` FLOAT,
  `net_income_q` FLOAT,
  `eps_basic_q` FLOAT,
  `eps_diluted_q` FLOAT,
  `shares_diluted` FLOAT,
  `ebitda_q` FLOAT,
  `revenue_ttm` FLOAT,
  `gross_profit_ttm` FLOAT,
  `operating_income_ttm` FLOAT,
  `net_income_ttm` FLOAT,
  `eps_basic_ttm` FLOAT,
  `eps_diluted_ttm` FLOAT,
  `ebitda_ttm` FLOAT,
  `gross_margin_ttm` FLOAT,
  `operating_margin_ttm` FLOAT,
  `net_margin_ttm` FLOAT,
  `ebitda_margin_ttm` FLOAT,
  `revenue_growth_yoy` FLOAT,
  `gross_profit_growth_yoy` FLOAT,
  `operating_income_growth_yoy` FLOAT,
  `net_income_growth_yoy` FLOAT,
  `eps_growth_yoy` FLOAT,
  `revenue_growth_qoq` FLOAT,
  `net_income_growth_qoq` FLOAT,
  PRIMARY KEY (`fiscal_quarter_end`, `ticker`),
  FOREIGN KEY (`ticker`) REFERENCES `stock_info_data`(`ticker`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

SELECT 'Created stock_income_stmt_quarterly' AS status;

-- Quarterly Balance Sheet
CREATE TABLE IF NOT EXISTS `stock_balancesheet_quarterly` (
  `fiscal_quarter_end` DATE NOT NULL,
  `fiscal_year` INT,
  `fiscal_quarter` INT,
  `date_published` DATE,
  `ticker` VARCHAR(255) NOT NULL,
  `total_assets` FLOAT,
  `current_assets` FLOAT,
  `cash_and_equivalents` FLOAT,
  `cash_and_investments` FLOAT,
  `accounts_receivable` FLOAT,
  `inventory` FLOAT,
  `goodwill` FLOAT,
  `intangible_assets` FLOAT,
  `total_liabilities` FLOAT,
  `current_liabilities` FLOAT,
  `accounts_payable` FLOAT,
  `total_debt` FLOAT,
  `short_term_debt` FLOAT,
  `long_term_debt` FLOAT,
  `total_equity` FLOAT,
  `retained_earnings` FLOAT,
  `current_ratio` FLOAT,
  `quick_ratio` FLOAT,
  `cash_ratio` FLOAT,
  `debt_to_equity` FLOAT,
  `debt_to_assets` FLOAT,
  `book_value_per_share` FLOAT,
  `tangible_book_per_share` FLOAT,
  `roa_ttm` FLOAT,
  `roe_ttm` FLOAT,
  `roic_ttm` FLOAT,
  `assets_growth_yoy` FLOAT,
  `equity_growth_yoy` FLOAT,
  `book_value_growth_yoy` FLOAT,
  PRIMARY KEY (`fiscal_quarter_end`, `ticker`),
  FOREIGN KEY (`ticker`) REFERENCES `stock_info_data`(`ticker`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

SELECT 'Created stock_balancesheet_quarterly' AS status;

-- Quarterly Cash Flow
CREATE TABLE IF NOT EXISTS `stock_cashflow_quarterly` (
  `fiscal_quarter_end` DATE NOT NULL,
  `fiscal_year` INT,
  `fiscal_quarter` INT,
  `date_published` DATE,
  `ticker` VARCHAR(255) NOT NULL,
  `operating_cash_flow_q` FLOAT,
  `capex_q` FLOAT,
  `free_cash_flow_q` FLOAT,
  `investing_cash_flow_q` FLOAT,
  `financing_cash_flow_q` FLOAT,
  `dividends_paid_q` FLOAT,
  `share_repurchases_q` FLOAT,
  `operating_cash_flow_ttm` FLOAT,
  `capex_ttm` FLOAT,
  `free_cash_flow_ttm` FLOAT,
  `dividends_paid_ttm` FLOAT,
  `share_repurchases_ttm` FLOAT,
  `fcf_per_share_ttm` FLOAT,
  `ocf_per_share_ttm` FLOAT,
  `fcf_margin_ttm` FLOAT,
  `capex_to_revenue_ttm` FLOAT,
  `fcf_conversion_ttm` FLOAT,
  `fcf_growth_yoy` FLOAT,
  `ocf_growth_yoy` FLOAT,
  PRIMARY KEY (`fiscal_quarter_end`, `ticker`),
  FOREIGN KEY (`ticker`) REFERENCES `stock_info_data`(`ticker`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

SELECT 'Created stock_cashflow_quarterly' AS status;

-- TTM-based Ratio Table (daily)
CREATE TABLE IF NOT EXISTS `stock_ratio_data_ttm` (
  `date` DATE NOT NULL,
  `ticker` VARCHAR(255) NOT NULL,
  `p_s_ttm` FLOAT,
  `p_e_ttm` FLOAT,
  `p_b` FLOAT,
  `p_fcf_ttm` FLOAT,
  `p_ocf_ttm` FLOAT,
  `ev_sales_ttm` FLOAT,
  `ev_ebitda_ttm` FLOAT,
  `ev_ebit_ttm` FLOAT,
  `ev_fcf_ttm` FLOAT,
  `peg_ratio` FLOAT,
  `pegf_ratio` FLOAT,
  `earnings_yield_ttm` FLOAT,
  `fcf_yield_ttm` FLOAT,
  `dividend_yield` FLOAT,
  `enterprise_value` FLOAT,
  `market_cap` FLOAT,
  PRIMARY KEY (`date`, `ticker`),
  FOREIGN KEY (`ticker`) REFERENCES `stock_info_data`(`ticker`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

SELECT 'Created stock_ratio_data_ttm' AS status;

-- Index Membership Tracking
CREATE TABLE IF NOT EXISTS `index_membership` (
  `ticker` VARCHAR(255) NOT NULL,
  `index_code` VARCHAR(50) NOT NULL,
  `index_name` VARCHAR(255),
  `exchange` VARCHAR(50),
  `date_added` DATE,
  `date_removed` DATE,
  `is_current` BOOLEAN DEFAULT TRUE,
  `last_updated` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`ticker`, `index_code`),
  FOREIGN KEY (`ticker`) REFERENCES `stock_info_data`(`ticker`),
  INDEX `idx_index_code` (`index_code`),
  INDEX `idx_is_current` (`is_current`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

SELECT 'Created index_membership' AS status;

-- ============================================
-- STEP 3: Verify new tables were created
-- ============================================

SELECT 'Migration complete! New table status:' AS status;

SELECT 
    TABLE_NAME,
    CREATE_TIME,
    TABLE_ROWS
FROM information_schema.TABLES 
WHERE TABLE_SCHEMA = 'stock_portefolio_builder'
  AND TABLE_NAME IN (
    'stock_income_stmt_quarterly',
    'stock_balancesheet_quarterly', 
    'stock_cashflow_quarterly',
    'stock_ratio_data_ttm',
    'index_membership'
  )
ORDER BY TABLE_NAME;

-- ============================================
-- NOTES:
-- ============================================
-- 
-- EXISTING TABLES PRESERVED:
--   - stock_info_data (parent table)
--   - stock_price_data (daily prices + technicals)
--   - stock_income_stmt_data (annual income statements)
--   - stock_balancesheet_data (annual balance sheets)
--   - stock_cash_flow_data (annual cash flow)
--   - stock_ratio_data (annual-based ratios)
--   - stock_prediction_data (predictions)
--
-- NEW TABLES ADDED:
--   - stock_income_stmt_quarterly (quarterly + TTM income)
--   - stock_balancesheet_quarterly (quarterly balance sheet)
--   - stock_cashflow_quarterly (quarterly + TTM cash flow)
--   - stock_ratio_data_ttm (TTM-based daily ratios)
--   - index_membership (tracks index constituents)
--
-- You can use BOTH annual and quarterly data side by side.
-- The quarterly tables provide more granular, up-to-date financials.
-- ============================================
