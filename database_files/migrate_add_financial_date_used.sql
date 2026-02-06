-- ============================================
-- MIGRATION: Add financial_date_used column to stock_ratio_data
-- ============================================
-- This migration adds tracking for which financial statement date
-- was used to calculate each ratio row.
--
-- Run this script if you have an existing database without the
-- financial_date_used column in stock_ratio_data table.
--
-- Author: Stock Portfolio Builder
-- Date: 2026-02
-- ============================================

USE `stock_portefolio_builder`;

-- Check if column already exists before adding
SET @column_exists = (
    SELECT COUNT(*) 
    FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_SCHEMA = 'stock_portefolio_builder' 
    AND TABLE_NAME = 'stock_ratio_data' 
    AND COLUMN_NAME = 'financial_date_used'
);

-- Add column if it doesn't exist
SET @sql = IF(@column_exists = 0,
    'ALTER TABLE `stock_ratio_data` ADD COLUMN `financial_date_used` DATE COMMENT ''Fiscal period end date of the financial report used for calculation'' AFTER `p_fcf`',
    'SELECT ''Column financial_date_used already exists'' AS status'
);

PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Verify the change
DESCRIBE stock_ratio_data;

-- Show migration complete message
SELECT 'Migration complete: financial_date_used column added to stock_ratio_data' AS status;

-- ============================================
-- NOTES:
-- ============================================
-- After running this migration:
-- 1. Existing ratio data will have NULL in financial_date_used
-- 2. The orchestrator will detect this and trigger recalculation
--    when it processes each ticker
-- 3. New ratio calculations will populate the financial_date_used column
--
-- To force immediate recalculation for all tickers, you can either:
-- A) Delete all ratio data: TRUNCATE TABLE stock_ratio_data;
-- B) Or let the orchestrator handle it naturally on next run
--    (it will recalculate when it sees NULL in financial_date_used)
-- ============================================
