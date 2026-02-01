-- ============================================
-- MIGRATION: Add quarterly_fetch_metadata table
-- ============================================
-- This table tracks when quarterly data was last fetched for each ticker
-- Used to implement smart caching and reduce unnecessary API calls
--
-- Run this migration to add the table to an existing database
-- ============================================

USE `stock_portefolio_builder`;

-- Create the metadata table if it doesn't exist
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

-- Verify table was created
SELECT 'quarterly_fetch_metadata table created successfully' AS status;

-- Optional: Pre-populate metadata for tickers that already have quarterly data
-- This sets last_fetch_date to NULL so they will be fetched on next run
-- Uncomment to run:
/*
INSERT IGNORE INTO quarterly_fetch_metadata (ticker, last_fetch_date, last_quarter_end, quarters_count)
SELECT 
    ticker,
    CURDATE() - INTERVAL 31 DAY,  -- Set to 31 days ago to trigger refresh
    MAX(fiscal_quarter_end),
    COUNT(*)
FROM stock_income_stmt_quarterly
GROUP BY ticker;
*/
