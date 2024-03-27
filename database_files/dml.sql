SELECT *
FROM stock_portefolio_builder.stock_info_data;

SELECT *
FROM stock_portefolio_builder.stock_price_data;

SELECT *
FROM stock_portefolio_builder.stock_price_data
WHERE ticker = "AMBU-B.CO";

SELECT *
FROM stock_portefolio_builder.stock_price_data
WHERE ticker = "AMBU-B.CO" AND `date` = "2024-03-27";

SELECT *
FROM (SELECT * FROM stock_portefolio_builder.stock_price_data
WHERE ticker = "AMBU-B.CO"
ORDER BY `date` 
DESC LIMIT 1) AS temp
ORDER BY date ASC;

SELECT *
FROM stock_portefolio_builder.stock_income_stmt_data
WHERE ticker = "NOVO-B.CO";

SELECT COUNT(financial_statement_date)
FROM stock_portefolio_builder.stock_income_stmt_data
WHERE ticker = "NOVO-B.CO";

SELECT *
FROM (SELECT * FROM stock_cash_flow_data ORDER BY financial_statement_date DESC LIMIT 1) AS temp
WHERE ticker = "NOVO-B.CO"
ORDER BY financial_statement_date ASC;

SELECT *
FROM stock_portefolio_builder.stock_ratio_data
WHERE ticker = "NOVO-B.CO" AND `date` > "2024-03-21";

SELECT COUNT(date)
FROM stock_portefolio_builder.stock_ratio_data
WHERE ticker = "NOVO-B.CO";
