-- ============================================
-- MIGRATION: Add stock_beta_data table
-- ============================================
-- Stores rolling beta values for each stock against multiple market indices.
-- Beta measures a stock's sensitivity to market movements:
--   beta > 1 → more volatile than the market
--   beta = 1 → moves with the market
--   beta < 1 → less volatile than the market
--   beta < 0 → moves inversely to the market
--
-- Usage: Run this script against an existing stock_portefolio_builder database
--        to add beta tracking functionality.
-- ============================================

USE `stock_portefolio_builder`;

CREATE TABLE IF NOT EXISTS `stock_beta_data` (
  `date` DATE NOT NULL COMMENT 'Calculation date',
  `ticker` VARCHAR(255) NOT NULL COMMENT 'Stock ticker symbol',
  `index_code` VARCHAR(50) NOT NULL COMMENT 'Benchmark index code (e.g., SP500, C25, DAX40)',
  `index_symbol` VARCHAR(50) COMMENT 'yfinance symbol for the index (e.g., ^GSPC, ^GDAXI)',

  -- Beta values over different lookback windows
  `beta_60d` FLOAT COMMENT '60-day rolling beta (≈3 months)',
  `beta_120d` FLOAT COMMENT '120-day rolling beta (≈6 months)',
  `beta_252d` FLOAT COMMENT '252-day rolling beta (≈1 year)',

  -- Supporting statistics
  `correlation_252d` FLOAT COMMENT '252-day rolling correlation with index',
  `r_squared_252d` FLOAT COMMENT '252-day R-squared (proportion of variance explained by index)',

  `last_updated` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

  PRIMARY KEY (`date`, `ticker`, `index_code`),
  FOREIGN KEY (`ticker`) REFERENCES `stock_info_data`(`ticker`) ON DELETE CASCADE,
  INDEX `idx_beta_ticker` (`ticker`),
  INDEX `idx_beta_index` (`index_code`),
  INDEX `idx_beta_date` (`date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Rolling beta values for stocks vs market indices';

SELECT 'Created stock_beta_data table' AS status;

-- ============================================
-- Verify table was created
-- ============================================
SELECT
  TABLE_NAME,
  TABLE_COMMENT
FROM information_schema.TABLES
WHERE TABLE_SCHEMA = 'stock_portefolio_builder'
  AND TABLE_NAME = 'stock_beta_data';
