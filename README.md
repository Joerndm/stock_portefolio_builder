# Stock Portfolio Builder

This repository contains scripts that fetch stock data and use it to build a machine learning model for predicting stock prices.

## Prerequisites

- Python 3.12.1 installed on your machine
- Download the entire folder to your local machine

## Installation

1. Open a terminal or command prompt.
2. Navigate to the project directory.
3. Run the following command to install the required Python libraries:

    ```
    pip install -r requirements.txt
    ```

## Running the Scripts

### Windows

1. Open a terminal or command prompt.
2. Navigate to the project directory.

#### Predicting the Price of a Single Stock

1. To predict the price of a single stock, run the following command:

    ```
    python3 ml_builder.py
    ```

    Note: Make sure you have the "index_symbol_list_single_stock.csv" file in the same directory as "ml_builder.py".

#### Predicting the Price of Multiple Stocks

1. To predict the price of multiple stocks, run the following command:

    ```
    python3 stock_analyzer.py
    ```

    Note: Make sure you have the "index_symbol_list_multiple_stocks.csv" file in the same directory as "stock_analyzer.py".

### iOS

1. Open a terminal.
2. Navigate to the project directory.

#### Predicting the Price of a Single Stock

1. To predict the price of a single stock, run the following command:

    ```shell
    python3 ml_builder.py
    ```

    Note: Make sure you have the "index_symbol_list_single_stock.csv" file in the same directory as "ml_builder.py".

#### Predicting the Price of Multiple Stocks

1. To predict the price of multiple stocks, run the following command:

    ```shell
    python3 stock_analyzer.py
    ```

    Note: Make sure you have the "index_symbol_list_multiple_stocks.csv" file in the same directory as "stock_analyzer.py".

## Changing the Stock Ticker

To change the stock ticker that "ml_builder.py" predicts the price of, follow these steps:

1. Open the "index_symbol_list_single_stock.csv" file in a text editor.
2. Change the ticker to the desired stock symbol.
3. Save the file.

## Additional Requirements

- "index_symbol_list_single_stock.csv" is required to run "ml_builder.py".
- "index_symbol_list_multiple_stocks.csv" is required to run "stock_analyzer.py".
