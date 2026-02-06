SELECT *
FROM stock_portefolio_builder.stock_info_data;

SELECT *
FROM stock_portefolio_builder.stock_price_data
ORDER BY date DESC;

SELECT *
FROM stock_portefolio_builder.stock_price_data
WHERE ticker = "DEMANT.CO" and 5Y IS NOT NULL
ORDER BY date DESC;

SELECT *
FROM stock_portefolio_builder.stock_price_data
WHERE ticker = "DEMANT.CO" AND `date` > "2024-03-27";

SELECT *
FROM (SELECT * FROM stock_portefolio_builder.stock_price_data
WHERE ticker = "ORSTED.CO"
ORDER BY `date` 
DESC LIMIT 1) AS temp
ORDER BY date ASC;

SELECT *
FROM stock_portefolio_builder.stock_income_stmt_data
WHERE ticker = "DEMANT.CO";

SELECT *
FROM stock_portefolio_builder.stock_income_stmt_data
WHERE gross_Profit_Growth IS NULL;

SELECT COUNT(financial_statement_date)
FROM stock_portefolio_builder.stock_income_stmt_data
WHERE ticker = "NOVO-B.CO";

SELECT financial_Statement_Date
FROM stock_portefolio_builder.stock_income_stmt_data
WHERE ticker = "NOVO-B.CO"
ORDER BY financial_Statement_Date DESC
LIMIT 1;

SELECT *
FROM stock_portefolio_builder.stock_income_stmt_quarterly
WHERE ticker= "DEMANT.CO";

SELECT *
FROM stock_portefolio_builder.stock_balancesheet_quarterly
WHERE ticker= "FDR.MC";

SELECT *
FROM stock_portefolio_builder.stock_cash_flow_data
WHERE ticker = "FDR.MC";

SELECT COUNT(financial_Statement_Date)
FROM stock_portefolio_builder.stock_cash_flow_data
WHERE ticker = "NOVO-B.CO";

SELECT *
FROM (SELECT *
FROM stock_portefolio_builder.stock_cash_flow_data
WHERE ticker = "NOVO-B.CO" 
ORDER BY financial_statement_date 
DESC LIMIT 1) AS temp
ORDER BY financial_statement_date ASC;

SELECT *
FROM stock_portefolio_builder.stock_ratio_data
ORDER BY date DESC;

SELECT *
FROM stock_portefolio_builder.stock_ratio_data
WHERE ticker = "NOVO-B.CO";


SELECT Count("ticker")
FROM stock_portefolio_builder.stock_ratio_data
WHERE ticker = "DEMANT.CO"
ORDER BY date ASC;

SELECT *
FROM stock_portefolio_builder.stock_ratio_data
WHERE ticker = "NOVO-B.CO"
ORDER BY date DESC;

SELECT *
FROM stock_portefolio_builder.stock_ratio_data_ttm
WHERE ticker = "NOVO-B.CO"
ORDER BY date DESC;

SELECT *
FROM stock_portefolio_builder.stock_ratio_data
WHERE ticker = "NOVO-B.CO" AND `date` > "2024-03-27";

SELECT *
FROM stock_portefolio_builder.stock_ratio_data
WHERE ticker = "NOVO-B.CO" AND `date` > "2024-03-21"
or ticker = "AMBU-B.CO" AND `date` > "2024-03-21";

SELECT COUNT(date)
FROM stock_portefolio_builder.stock_ratio_data
WHERE ticker = "NOVO-B.CO";
