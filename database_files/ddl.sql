DROP DATABASE IF EXISTS `stock_portefolio_builder`;

CREATE DATABASE IF NOT EXISTS `stock_portefolio_builder`;

USE `stock_portefolio_builder`;

CREATE TABLE `stock_info_data` (
  `ticker` VARCHAR(255) NOT NULL,
  `company_Name` VARCHAR(255),
  `industry` VARCHAR(255),
  PRIMARY KEY (`ticker`),
  UNIQUE (`ticker`)
);

CREATE TABLE `stock_price_data` (
  `date` DATE NOT NULL,
  `ticker` VARCHAR(255) NOT NULL ,
  `currency` VARCHAR(255),
  `trade_Volume` FLOAT,
  `open_Price` FLOAT,
  `1D` FLOAT,
  `1M` FLOAT,
  `3M` FLOAT,
  `6M` FLOAT,
  `9M` FLOAT,
  `1Y` FLOAT,
  `2Y` FLOAT,
  `3Y` FLOAT,
  `4Y` FLOAT,
  `5Y` FLOAT,
  `sma_40` FLOAT,
  `sma_120` FLOAT,
  `ema_40` FLOAT,
  `ema_120` FLOAT,
  `std_Div_40` FLOAT,
  `std_Div_120` FLOAT,
  `bollinger_Band_40_2STD` FLOAT,
  `bollinger_Band_120_2STD` FLOAT,
  `momentum` FLOAT,
  CONSTRAINT `PK_stock_price` PRIMARY KEY (`date`, `ticker`),
  FOREIGN KEY (`ticker`) REFERENCES `stock_info_data`(`ticker`)
);

CREATE TABLE `stock_income_stmt_data` (
  `financial_Statement_Date` DATE NOT NULL,
  `date_published` DATE,
  `ticker` VARCHAR(255) NOT NULL,
  `revenue` FLOAT,
  `revenue_Growth` FLOAT,
  `gross_Profit` FLOAT,
  `gross_Profit_Growth` FLOAT,
  `gross_Margin` FLOAT,
  `gross_Margin_Growth` FLOAT,
  `operating_Earning` FLOAT,
  `operating_Earning_Growth` FLOAT,
  `operating_Earning_Margin` FLOAT,
  `operating_Earning_Margin_Growth` FLOAT,
  `net_Income` FLOAT,
  `net_Income_Growth` FLOAT,
  `net_Income_Margin` FLOAT,
  `net_Income_Margin_Growth` FLOAT,
  `eps` FLOAT,
  `eps_Growth` FLOAT,
  `average_shares` FLOAT,
  CONSTRAINT `PK_Stock_income_stmt` PRIMARY KEY (`financial_Statement_Date`, `ticker`),
  FOREIGN KEY (`ticker`) REFERENCES `stock_info_data`(`ticker`)
);

CREATE TABLE `stock_balancesheet_data` (
  `financial_Statement_Date` DATE NOT NULL,
  `date_Published` DATE,
  `ticker` VARCHAR(255) NOT NULL,
  `total_Assets` FLOAT,
  `total_Assets_Growth` FLOAT,
  `current_Assets` FLOAT,
  `current_Assets_Growth` FLOAT,
  `cash_And_Cash_Equivalents` FLOAT,
  `cash_And_Cash_Equivalents_Growth` FLOAT,
  `equity` FLOAT,
  `equity_Growth` FLOAT,
  `liabilities` FLOAT,
  `liabilities_Growth` FLOAT,
  `current_liabilities` FLOAT,
  `current_liabilities_Growth` FLOAT,
  `book_Value` FLOAT,
  `book_Value_Growth` FLOAT,
  `book_Value_Per_Share` FLOAT,
  `book_Value_Per_Share_Growth` FLOAT,
  `return_On_Assets` FLOAT,
  `return_On_Assets_Growth` FLOAT,
  `return_On_Equity` FLOAT,
  `return_On_Equity_Growth` FLOAT,
  `current_Ratio` FLOAT,
  `current_Ratio_Growth` FLOAT,
  `quick_Ratio` FLOAT,
  `quick_Ratio_Growth` FLOAT,
  `debt_To_Equity` FLOAT,
  `debt_To_Equity_Growth` FLOAT,
  CONSTRAINT `PK_dtock_balancesheet` PRIMARY KEY (`financial_Statement_Date`, `ticker`),
  FOREIGN KEY (`ticker`) REFERENCES `stock_info_data`(`ticker`)
);

CREATE TABLE `stock_cash_flow_data` (
  `financial_Statement_Date` DATE NOT NULL,
  `date_published` DATE,
  `ticker` VARCHAR(255) NOT NULL,
  `free_Cash_Flow` FLOAT,
  `free_Cash_Flow_Growth` FLOAT,
  `free_Cash_Flow_Per_Share` FLOAT,
  `free_Cash_Flow_Per_Share_Growth` FLOAT,
  CONSTRAINT `PK_stock_cash_flow` PRIMARY KEY (`financial_Statement_Date`, `ticker`),
  FOREIGN KEY (`ticker`) REFERENCES `stock_info_data`(`ticker`)
);

CREATE TABLE `stock_ratio_data` (
  `date` DATE NOT NULL,
  `ticker` VARCHAR(255) NOT NULL,
  `p_s` FLOAT,
  `p_e` FLOAT,
  `p_b` FLOAT,
  `p_fcf` FLOAT,
  CONSTRAINT `PK_stock_ratio_data` PRIMARY KEY (`date`, `ticker`),
  FOREIGN KEY (`ticker`) REFERENCES `stock_info_data`(`ticker`)
);

CREATE TABLE `stock_prediction_data` (
  `ticker` VARCHAR(255) NOT NULL,
  `model_Type` VARCHAR(255),
  `model_Equation` VARCHAR(255),
  `prediction_30_Days` FLOAT,
  `prediction_60_Days` FLOAT,
  `prediction_90_Days` FLOAT,
  PRIMARY KEY (`ticker`),
  FOREIGN KEY (`ticker`) REFERENCES `stock_info_data`(`ticker`)
);
