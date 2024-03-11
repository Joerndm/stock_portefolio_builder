DROP DATABASE IF EXISTS `stock_portefolio_builder`;

CREATE DATABASE IF NOT EXISTS `stock_portefolio_builder`;

USE `stock_portefolio_builder`;

CREATE TABLE `stock_Info_Data` (
  `ticker` VARCHAR(255) NOT NULL,
  `company_Name` VARCHAR(255),
  `industry` VARCHAR(255),
  PRIMARY KEY (`ticker`),
  UNIQUE (`ticker`)
);

CREATE TABLE `stock_Prediction` (
  `ticker` VARCHAR(255) NOT NULL,
  `model_Type` VARCHAR(255),
  `model_Equation` VARCHAR(255),
  `prediction_30_Days` FLOAT,
  `prediction_60_Days` FLOAT,
  `prediction_90_Days` FLOAT,
  PRIMARY KEY (`ticker`),
  FOREIGN KEY (`ticker`) REFERENCES `stock_Info_Data`(`ticker`)
);

CREATE TABLE `stock_Price_Data` (
  `date` DATE NOT NULL,
  `ticker` VARCHAR(255) NOT NULL ,
  `currency` VARCHAR(255),
  `opening_Price` FLOAT,
  `average_Price` FLOAT,
  `trade_Volumne` FLOAT,
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
  `bollinger_40_2STD` FLOAT,
  `sma_120` FLOAT,
  `bollinger_120_2STD` FLOAT,
  `ema_40` FLOAT,
  `ema_120` FLOAT,
  `beta` FLOAT,
  `rsi` FLOAT,
  CONSTRAINT `PK_Stock_Price` PRIMARY KEY (`date`, `ticker`),
  FOREIGN KEY (`ticker`) REFERENCES `stock_Info_Data`(`ticker`)
);

CREATE TABLE `stock_Income_Stmt_Data` (
  `date` DATE NOT NULL,
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
  CONSTRAINT `PK_Stock_Income_STMT` PRIMARY KEY (`date`, `ticker`),
  FOREIGN KEY (`ticker`) REFERENCES `stock_Info_Data`(`ticker`)
);

CREATE TABLE `stock_Balancesheet_Data` (
  `date` DATE NOT NULL,
  `date_published` DATE,
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
  CONSTRAINT `PK_Stock_Balancesheet` PRIMARY KEY (`date`, `ticker`),
  FOREIGN KEY (`ticker`) REFERENCES `stock_Info_Data`(`ticker`)
);

CREATE TABLE `stock_Cash_Flow_Data` (
  `date` DATE NOT NULL,
  `date_published` DATE,
  `ticker` VARCHAR(255) NOT NULL,
  `free_Cash_Flow` FLOAT,
  `free_Cash_Flow_Growth` FLOAT,
  `free_Cash_Flow_Per_Share` FLOAT,
  `free_Cash_Flow_Per_Share_Growth` FLOAT,
  CONSTRAINT `PK_Stock_Cash_Flow` PRIMARY KEY (`date`, `ticker`),
  FOREIGN KEY (`ticker`) REFERENCES `stock_Info_Data`(`ticker`)
);

