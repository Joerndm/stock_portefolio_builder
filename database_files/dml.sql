SELECT *
FROM stock_portefolio_builder.stock_info_data;

SELECT *
FROM stock_portefolio_builder.stock_price_data;

SELECT *
FROM stock_portefolio_builder.stock_price_data
WHERE ticker = "NOVO-B.CO" AND `date` = "2024-03-19";

SELECT *
FROM stock_portefolio_builder.stock_income_stmt_data
WHERE ticker = "NOVO-B.CO";

SELECT *
FROM
(SELECT * FROM stock_cash_flow_data
ORDER BY financial_statement_date DESC LIMIT 1
) AS temp
WHERE ticker = "NOVO-B.CO"
ORDER BY financial_statement_date ASC;

SELECT COUNT(financial_statement_date)
FROM stock_portefolio_builder.stock_income_stmt_data
WHERE ticker = "NOVO-B.CO";

SELECT *
FROM stock_portefolio_builder.stock_price_data
WHERE ticker = "NOVO-B.CO" AND date >= "2021-12-31";

SELECT *
FROM stock_portefolio_builder.stock_ratio_data
WHERE ticker = "NOVO-B.CO";

SELECT COUNT(date)
FROM stock_portefolio_builder.stock_ratio_data
WHERE ticker = "NOVO-B.CO";
