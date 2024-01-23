# Stock Picker

## Description
This repository contains scripts for fetching stock data and building a Machine Learning model.

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

## Running the Scripts (Windows)
1. Open a terminal or command prompt.
2. Navigate to the project directory.
3. To build datasets, run the following command:
    ```
    python3 stock_data_fetch.py
    ```
    Note: Make sure you have the "index_symbol_list_single_stock.csv" file in the same directory as "stock_data_fetch.py".
4. To build the Machine Learning model, run the following command:
    ```
    python3 ml_builder.py
    ```
    Note: Make sure you have the "stock_data_test_single_mod.csv" file in the same directory as "ml_builder.py".

## Running the Scripts (iOS)
1. Open a terminal.
2. Navigate to the project directory.
3. To build datasets, run the following command:
    ```
    python3 stock_data_fetch.py
    ```
    Note: Make sure you have the "index_symbol_list_single_stock.csv" file in the same directory as "stock_data_fetch.py".
4. To build the Machine Learning model, run the following command:
    ```
    python3 ml_builder.py
    ```
    Note: Make sure you have the "stock_data_test_single_mod.csv" file in the same directory as "ml_builder.py".

## Running "stock_data_fetch.py" with Multiple Stocks
To run "stock_data_fetch.py" with multiple stocks, follow these steps:
1. Open "stock_data_fetch.py" in a text editor.
2. Go to line 1760.
3. Change the file name to "index_symbol_list_multiple_stocks.csv".
4. Save the file.
