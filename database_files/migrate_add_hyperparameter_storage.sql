-- Migration Script: Add Hyperparameter Storage Table
-- This script adds a table for storing best hyperparameters from tuning
-- to avoid re-tuning and reduce storage usage of tuning_dir
--
-- Run this ONCE to upgrade your database schema
-- Safe to run multiple times (uses IF NOT EXISTS)
--
-- Author: Stock Portfolio Builder
-- Date: February 2026

USE `stock_portefolio_builder`;

-- ============================================
-- HYPERPARAMETER STORAGE TABLE
-- ============================================
-- Stores best hyperparameters from tuning sessions
-- Allows skipping tuning if recent valid HPs exist

CREATE TABLE IF NOT EXISTS `model_hyperparameters` (
  `hp_id` INT AUTO_INCREMENT,
  `ticker` VARCHAR(255) NOT NULL COMMENT 'Stock ticker symbol',
  `model_type` ENUM('rf', 'xgb', 'lstm', 'tcn') NOT NULL COMMENT 'Model type',
  `tuning_date` TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'When tuning was performed',
  
  -- Hyperparameters stored as JSON for flexibility
  `hyperparameters` JSON NOT NULL COMMENT 'Best hyperparameters as JSON object',
  
  -- Tuning metadata
  `num_trials` INT COMMENT 'Number of trials in tuning session',
  `best_score` FLOAT COMMENT 'Best validation score achieved',
  `tuning_time_seconds` FLOAT COMMENT 'Time taken for tuning',
  
  -- Dataset characteristics (for validation)
  `training_samples` INT COMMENT 'Number of training samples used',
  `num_features` INT COMMENT 'Number of features used',
  `feature_hash` VARCHAR(64) COMMENT 'Hash of feature list for validation',
  
  -- Model performance metrics on validation set
  `val_mse` FLOAT COMMENT 'Validation MSE',
  `val_r2` FLOAT COMMENT 'Validation R2',
  `val_mae` FLOAT COMMENT 'Validation MAE',
  
  -- Flags
  `is_constrained` BOOLEAN DEFAULT FALSE COMMENT 'Whether overfitting constraints were applied',
  `is_valid` BOOLEAN DEFAULT TRUE COMMENT 'Whether these HPs are still valid for use',
  
  PRIMARY KEY (`hp_id`),
  UNIQUE KEY `uk_ticker_model` (`ticker`, `model_type`),  -- One set of HPs per ticker/model
  FOREIGN KEY (`ticker`) REFERENCES `stock_info_data`(`ticker`),
  INDEX `idx_ticker` (`ticker`),
  INDEX `idx_model_type` (`model_type`),
  INDEX `idx_tuning_date` (`tuning_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Best hyperparameters from model tuning';


-- ============================================
-- VERIFY TABLE CREATION
-- ============================================

SELECT 'Hyperparameter storage table created:' AS status;

SELECT 
    TABLE_NAME,
    TABLE_COMMENT
FROM information_schema.TABLES 
WHERE TABLE_SCHEMA = 'stock_portefolio_builder'
AND TABLE_NAME = 'model_hyperparameters';

SELECT 'Migration complete!' AS status;
