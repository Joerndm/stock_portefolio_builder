-- Migration Script: Add Extended Prediction and Monte Carlo Tables
-- This script adds new tables for storing ML prediction results and Monte Carlo simulations
--
-- Run this ONCE to upgrade your database schema
-- Safe to run multiple times (uses IF NOT EXISTS)
--
-- New tables:
--   - stock_prediction_extended: Detailed ML predictions with confidence intervals
--   - monte_carlo_results: Monte Carlo simulation results by year
--   - portfolio_runs: Track portfolio optimization runs
--   - portfolio_holdings: Individual stock holdings in each portfolio
--
-- Author: Stock Portfolio Builder
-- Date: February 2026

USE `stock_portefolio_builder`;

-- ============================================
-- STEP 1: Verify existing tables are intact
-- ============================================

SELECT 'Checking existing tables...' AS status;

SELECT 
    TABLE_NAME,
    TABLE_ROWS
FROM information_schema.TABLES 
WHERE TABLE_SCHEMA = 'stock_portefolio_builder'
ORDER BY TABLE_NAME;

-- ============================================
-- STEP 2: Add Extended Prediction Table
-- ============================================
-- Stores detailed ML predictions with uncertainty bounds

CREATE TABLE IF NOT EXISTS `stock_prediction_extended` (
  `prediction_id` INT AUTO_INCREMENT,
  `prediction_date` DATE NOT NULL COMMENT 'Date when prediction was made',
  `ticker` VARCHAR(255) NOT NULL COMMENT 'Stock ticker symbol',
  `prediction_horizon_days` INT NOT NULL COMMENT 'Days into the future (e.g., 30, 60, 90, 252)',
  `target_date` DATE COMMENT 'Target date for this prediction',
  
  -- Price predictions
  `predicted_price` FLOAT COMMENT 'Point estimate of predicted price',
  `current_price` FLOAT COMMENT 'Price at prediction time (for return calculation)',
  `predicted_return` FLOAT COMMENT 'Predicted return as decimal (e.g., 0.10 for 10%)',
  
  -- Confidence intervals
  `confidence_lower_5` FLOAT COMMENT '5th percentile price (95% confidence lower bound)',
  `confidence_lower_16` FLOAT COMMENT '16th percentile price (68% confidence lower bound)',
  `confidence_upper_84` FLOAT COMMENT '84th percentile price (68% confidence upper bound)',
  `confidence_upper_95` FLOAT COMMENT '95th percentile price (95% confidence upper bound)',
  
  -- Model information
  `model_type` VARCHAR(100) COMMENT 'Model type (e.g., ensemble, tcn, lstm, rf, xgb)',
  `model_version` VARCHAR(50) COMMENT 'Model version or run identifier',
  
  -- Uncertainty metrics
  `prediction_std` FLOAT COMMENT 'Standard deviation of prediction',
  `mc_dropout_used` BOOLEAN DEFAULT FALSE COMMENT 'Whether Monte Carlo Dropout was used',
  `mc_iterations` INT COMMENT 'Number of MC Dropout iterations if used',
  
  PRIMARY KEY (`prediction_id`),
  UNIQUE KEY `uk_prediction` (`prediction_date`, `ticker`, `prediction_horizon_days`),
  FOREIGN KEY (`ticker`) REFERENCES `stock_info_data`(`ticker`),
  INDEX `idx_ticker` (`ticker`),
  INDEX `idx_prediction_date` (`prediction_date`),
  INDEX `idx_target_date` (`target_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Extended ML predictions with confidence intervals';


-- ============================================
-- STEP 3: Add Monte Carlo Results Table
-- ============================================
-- Stores Monte Carlo simulation results by year

CREATE TABLE IF NOT EXISTS `monte_carlo_results` (
  `result_id` INT AUTO_INCREMENT,
  `simulation_date` DATE NOT NULL COMMENT 'Date when simulation was run',
  `ticker` VARCHAR(255) NOT NULL COMMENT 'Stock ticker symbol',
  `simulation_year` INT NOT NULL COMMENT 'Year number in simulation (1-10)',
  `num_simulations` INT COMMENT 'Number of Monte Carlo paths simulated',
  
  -- Price percentiles at end of year
  `percentile_5` FLOAT COMMENT '5th percentile price',
  `percentile_10` FLOAT COMMENT '10th percentile price',
  `percentile_16` FLOAT COMMENT '16th percentile price (~1 std below)',
  `percentile_25` FLOAT COMMENT '25th percentile price',
  `mean_price` FLOAT COMMENT 'Mean simulated price',
  `median_price` FLOAT COMMENT 'Median (50th percentile) price',
  `percentile_75` FLOAT COMMENT '75th percentile price',
  `percentile_84` FLOAT COMMENT '84th percentile price (~1 std above)',
  `percentile_90` FLOAT COMMENT '90th percentile price',
  `percentile_95` FLOAT COMMENT '95th percentile price',
  
  -- Return percentiles (more useful for portfolio construction)
  `return_percentile_5` FLOAT COMMENT '5th percentile annualized return',
  `return_mean` FLOAT COMMENT 'Mean annualized return',
  `return_percentile_95` FLOAT COMMENT '95th percentile annualized return',
  
  -- Risk metrics
  `volatility` FLOAT COMMENT 'Simulated volatility for this year',
  `var_95` FLOAT COMMENT 'Value at Risk (95%)',
  `cvar_95` FLOAT COMMENT 'Conditional VaR / Expected Shortfall (95%)',
  
  -- Simulation parameters
  `mu_used` FLOAT COMMENT 'Drift parameter used in simulation',
  `sigma_used` FLOAT COMMENT 'Volatility parameter used in simulation',
  `starting_price` FLOAT COMMENT 'Starting price for simulation',
  
  PRIMARY KEY (`result_id`),
  UNIQUE KEY `uk_mc_result` (`simulation_date`, `ticker`, `simulation_year`),
  FOREIGN KEY (`ticker`) REFERENCES `stock_info_data`(`ticker`),
  INDEX `idx_ticker` (`ticker`),
  INDEX `idx_simulation_date` (`simulation_date`),
  INDEX `idx_year` (`simulation_year`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Monte Carlo simulation results by year';


-- ============================================
-- STEP 4: Add Portfolio Runs Table
-- ============================================
-- Tracks each portfolio optimization run

CREATE TABLE IF NOT EXISTS `portfolio_runs` (
  `run_id` INT AUTO_INCREMENT,
  `run_date` TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'When the run was executed',
  `run_name` VARCHAR(255) COMMENT 'Optional name for this portfolio run',
  
  -- Investor profile settings
  `risk_level` ENUM('low', 'medium', 'high') NOT NULL DEFAULT 'medium',
  `investment_years` INT NOT NULL DEFAULT 5,
  `portfolio_size` INT NOT NULL DEFAULT 25,
  
  -- Filter criteria used
  `industries_filter` TEXT COMMENT 'JSON array of industries (null = all)',
  `countries_filter` TEXT COMMENT 'JSON array of countries (null = all)',
  `excluded_tickers` TEXT COMMENT 'JSON array of excluded tickers',
  
  -- Portfolio results
  `total_stocks_analyzed` INT COMMENT 'Number of stocks that were analyzed',
  `successful_predictions` INT COMMENT 'Number of successful predictions',
  `failed_predictions` INT COMMENT 'Number of failed predictions',
  
  -- Portfolio metrics
  `expected_return` FLOAT COMMENT 'Portfolio expected annual return',
  `expected_volatility` FLOAT COMMENT 'Portfolio expected volatility',
  `sharpe_ratio` FLOAT COMMENT 'Portfolio Sharpe ratio',
  
  -- Monte Carlo results for portfolio
  `mc_return_p5` FLOAT COMMENT 'Portfolio 5th percentile return',
  `mc_return_mean` FLOAT COMMENT 'Portfolio mean return',
  `mc_return_p95` FLOAT COMMENT 'Portfolio 95th percentile return',
  
  -- Status
  `status` ENUM('running', 'completed', 'failed') DEFAULT 'running',
  `error_message` TEXT COMMENT 'Error message if failed',
  `execution_time_seconds` FLOAT COMMENT 'Total execution time',
  
  PRIMARY KEY (`run_id`),
  INDEX `idx_run_date` (`run_date`),
  INDEX `idx_risk_level` (`risk_level`),
  INDEX `idx_status` (`status`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Portfolio optimization run tracking';


-- ============================================
-- STEP 5: Add Portfolio Holdings Table
-- ============================================
-- Stores individual stock holdings for each portfolio run

CREATE TABLE IF NOT EXISTS `portfolio_holdings` (
  `holding_id` INT AUTO_INCREMENT,
  `run_id` INT NOT NULL COMMENT 'Reference to portfolio_runs',
  `ticker` VARCHAR(255) NOT NULL COMMENT 'Stock ticker',
  
  -- Portfolio weight and ranking  
  `weight` FLOAT NOT NULL COMMENT 'Weight in portfolio (0-1)',
  `rank` INT COMMENT 'Rank in portfolio (1 = highest weight)',
  
  -- Individual stock metrics at time of selection
  `expected_return` FLOAT COMMENT 'Expected return for this stock',
  `volatility` FLOAT COMMENT 'Historical volatility',
  `sharpe_ratio` FLOAT COMMENT 'Individual Sharpe ratio',
  
  -- Correlation with portfolio
  `correlation_to_portfolio` FLOAT COMMENT 'Correlation with rest of portfolio',
  `marginal_contribution_to_risk` FLOAT COMMENT 'MCR to portfolio risk',
  
  -- Sector/Industry for diversification tracking
  `industry` VARCHAR(255) COMMENT 'Industry classification',
  `country` VARCHAR(100) COMMENT 'Country of domicile',
  
  PRIMARY KEY (`holding_id`),
  UNIQUE KEY `uk_holding` (`run_id`, `ticker`),
  FOREIGN KEY (`run_id`) REFERENCES `portfolio_runs`(`run_id`) ON DELETE CASCADE,
  FOREIGN KEY (`ticker`) REFERENCES `stock_info_data`(`ticker`),
  INDEX `idx_run_id` (`run_id`),
  INDEX `idx_ticker` (`ticker`),
  INDEX `idx_weight` (`weight`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Individual holdings in optimized portfolios';


-- ============================================
-- STEP 6: Add country column to stock_info_data if missing
-- ============================================
-- This prepares for future geographical filtering

-- Check if column exists before adding
SET @column_exists = (
    SELECT COUNT(*) 
    FROM information_schema.COLUMNS 
    WHERE TABLE_SCHEMA = 'stock_portefolio_builder' 
    AND TABLE_NAME = 'stock_info_data' 
    AND COLUMN_NAME = 'country'
);

-- Only add if not exists (MySQL doesn't support IF NOT EXISTS for ALTER)
-- You may need to run this manually if the column doesn't exist:
-- ALTER TABLE `stock_info_data` ADD COLUMN `country` VARCHAR(100) COMMENT 'Country of domicile' AFTER `industry`;

SELECT CASE 
    WHEN @column_exists > 0 THEN 'Country column already exists'
    ELSE 'NOTE: Run this command manually: ALTER TABLE stock_info_data ADD COLUMN country VARCHAR(100) AFTER industry'
END AS migration_note;


-- ============================================
-- STEP 7: Verify new tables were created
-- ============================================

SELECT 'New tables created:' AS status;

SELECT 
    TABLE_NAME,
    TABLE_COMMENT
FROM information_schema.TABLES 
WHERE TABLE_SCHEMA = 'stock_portefolio_builder'
AND TABLE_NAME IN (
    'stock_prediction_extended',
    'monte_carlo_results', 
    'portfolio_runs',
    'portfolio_holdings'
)
ORDER BY TABLE_NAME;

SELECT 'Migration complete!' AS status;
