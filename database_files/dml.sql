SELECT * FROM stock_portefolio_builder.stock_info_data;

SELECT * FROM stock_portefolio_builder.stock_price_data;

SELECT * FROM stock_portefolio_builder.stock_price_data
WHERE ticker = "NOVO-B.CO" AND `date` = "2024-03-19";

SELECT * FROM stock_portefolio_builder.stock_income_stmt_data
WHERE ticker = "NOVO-B.CO";

SELECT * FROM
(SELECT * FROM stock_cash_flow_data
ORDER BY date DESC LIMIT 1
) AS temp
WHERE ticker = "NOVO-B.CO"
ORDER BY date ASC;