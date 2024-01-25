import os
import pandas as pd
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

import stock_data_fetch
import import_csv_file
import split_dataset
import pca_dataset_analysis

def predict_price(traning_dataset, test_dataset, prediction_dataset, stock_df):
    x_training_df = traning_dataset.drop(["Price"], axis=1)
    y_training_df = traning_dataset["Price"]
    # Convert the DataFrame to a numpy array
    x_training = np.array(x_training_df)
    y_training = np.array(y_training_df)
    x_test_df = test_dataset.drop(["Price"], axis=1)
    y_test_df = test_dataset["Price"]
    # Convert the DataFrame to a numpy array
    x_test = np.array(x_test_df)
    y_test = np.array(y_test_df)
    # Convert the DataFrame to a numpy array
    x_prediction = np.array(prediction_dataset)
    lr = LinearRegression()
    lr.fit(x_training, y_training)
    lr_confidence = lr.score(x_test, y_test)
    lr_mean_absolute_error = mean_absolute_error(y_test, lr.predict(x_test))
    rf = RandomForestRegressor()
    rf.fit(x_test, y_test)
    rf_confidence = rf.score(x_test, y_test)
    rf_mean_absolute_error = mean_absolute_error(y_test, rf.predict(x_test))
    rg = Ridge()
    rg.fit(x_test, y_test)
    # Print the fitted model's weight and intercept
    # print(f"Weight: {rg.coef_}")
    # print(f"Intercept: {rg.intercept_}")
    # # Print the model's equation
    # print("The equation of the price prediction model is: ")
    # print("y = ")
    # for feature in range(len(rg.coef_)):
    #     print(f"{rg.coef_[feature]} * {train_data_df.drop(["Price", "Prediction"], axis=1).columns[feature]} + ")
        

    # print(f"{rg.intercept_}")
    rg_confidence = rg.score(x_test, y_test)
    rg_mean_absolute_error = mean_absolute_error(y_test, rg.predict(x_test))
    predict_precision_df = pd.DataFrame({"Model": ["Linear Regression", "Random Forest", "Ridge"],
        "Confidence": [lr_confidence, rf_confidence, rg_confidence],
        "Mean Absolute Error": [lr_mean_absolute_error, rf_mean_absolute_error, rg_mean_absolute_error]
    })
    print(predict_precision_df)
    forecast_set_lr = lr.predict(x_prediction)
    forecast_set_rf = rf.predict(x_prediction)
    forecast_set_rg = rg.predict(x_prediction)
    date_list = []
    for i in range(len(x_prediction)):
        x = len(x_prediction) - i
        date = stock_df["Date"].values[-x]
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
        date_list.append(date)


    forecast_dict = {"Date":date_list, "Price_lr":forecast_set_lr,
                    "Price_rf":forecast_set_rf, "Price_rg":forecast_set_rg,
    }
    forecast_df = pd.DataFrame(forecast_dict)
    print(forecast_df)
    predicted_return = ((forecast_df.iloc[-1]["Price_rg"] / forecast_df.iloc[0]["Price_rg"]) - 1) * 100
    if predicted_return > 0:
        print(f"The prediction expects a profitable return on: {predicted_return}%, over the next {len(forecast_df)} days.")
    elif predicted_return < 0:
        print(f"The prediction expects a loss of: {predicted_return}%, over the next {len(forecast_df)} days.")
    else:
        print(f"The prediction expects no return over the next {len(forecast_df)} days.")


    return forecast_df


def plot_graph(stock_data_df, forecast_data_df):
    """
    Plots a graph of the stock data.

    Parameters:
    - stock_data_df (pandas.DataFrame): A DataFrame containing the stock data.
    """

    # Plot the graph
    plt.figure(figsize=(18, 8))
    stock_data_df["Date"] = stock_data_df["Date"].astype('datetime64[ns]')
    stock_data_df = stock_data_df.set_index("Date")
    forecast_data_df["Date"] = forecast_data_df["Date"].astype('datetime64[ns]')
    forecast_data_df = forecast_data_df.set_index("Date")
    stock_data_df["Price"].plot()
    forecast_data_df["Price_lr"].plot()
    forecast_data_df["Price_rf"].plot()
    forecast_data_df["Price_rg"].plot()
    plt.legend([
        "Stock Price", "Linear Regression",
        "Random Forest", "Ridge"
        ],
        loc="best"
    )
    # plt.legend([
    #     "Stock Price", "Ridge"
    #     ],
    #     loc="upper right"
    # )
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"Stock Price Prediction of {stock_data_df.iloc[0]["Name"]}")
    # Change " " in stock_data_df.iloc[0]["Name"] to "_" to avoid error when saving the graph
    stock_data_df = stock_data_df.replace({"Name": [" ", "/"]}, {"Name": "_"}, regex=True)
    stock_name = stock_data_df.iloc[0]["Name"]
    graph_name = str(f"stock_prediction_of_{stock_name}.png")
    my_path = os.path.abspath(__file__)
    path = os.path.dirname(my_path)
    # Save the graph
    # try:
    #     plt.savefig(os.path.join(path, "generated_graphs", graph_name), bbox_inches="tight", pad_inches=0.5, transparent=False, format="png")


    # Show the graph
    plt.show()

    # except FileNotFoundError:
    #     raise FileNotFoundError("The graph could not be saved. Please check the file name or path.")


if __name__ == "__main__":
    start_time = time.time()
    stock_symbols_df = stock_data_fetch.import_stock_symbols('index_symbol_list_single_stock.csv')
    stock_symbols_list = stock_symbols_df['Symbol'].tolist()
    stock_symbol = stock_symbols_list[0]
    print(stock_symbol)
    # Fetch stock data for the imported stock symbols
    stock_data_df = stock_data_fetch.fetch_stock_price_data(stock_symbol)
    # print(stock_data_df)
    # Fetch stock data for the imported stock symbols
    full_stock_financial_data_df = stock_data_fetch.fetch_stock_financial_data(stock_symbol)
    # print(full_stock_financial_data_df)
    # Combine stock data with stock financial data
    combined_stock_data_df = stock_data_fetch.combine_stock_data(stock_data_df, full_stock_financial_data_df)
    # print(combined_stock_data_df)
    # Calculate ratios
    combined_stock_data_df = stock_data_fetch.calculate_ratios(combined_stock_data_df)
    # print(combined_stock_data_df)
    # Create a dictionary of dataframes to export to Excel
    dataframes = {
        # "Stock Data": stock_data_df,
        # "Full Stock Financial Data": full_stock_financial_data_df,
        "Combined Stock Data": combined_stock_data_df
    }
    # Export the dataframes to an Excel file
    stock_data_fetch.export_to_excel(dataframes, 'stock_data_single_v2.xlsx')
    # Import the stock data from an Excel file
    dataframes = stock_data_fetch.import_excel("stock_data_single_v2.xlsx")
    for key, value in dataframes.items():
        dataframe = value


    # Export the stock data to a CSV file
    stock_data_fetch.convert_excel_to_csv(dataframe, "stock_data_single_v2")
    stock_data_df = import_csv_file.import_as_df('stock_data_single_v2.csv')
    # Split the dataset into traning, test data and prediction data
    x_training_data, x_test_data, y_training_data, y_test_data, prediction_data = split_dataset.dataset_train_test_split(stock_data_df, 0.20, 1)
    # Reduce the dataset dimensions with PCA
    x_training_dataset, x_test_dataset, x_prediction_dataset = pca_dataset_analysis.pca_dataset_transformation(x_training_data, x_test_data, prediction_data, 10)
    # Combine the reduced dataset with the stock price
    x_training_dataset_df = pd.DataFrame(x_training_dataset)
    y_training_data_df = pd.DataFrame(y_training_data, columns=["Price"])
    traning_dataset_df = x_training_dataset_df.join(y_training_data_df)
    x_test_dataset_df = pd.DataFrame(x_test_dataset)
    y_test_data_df = pd.DataFrame(y_test_data, columns=["Price"])
    test_dataset_df = x_test_dataset_df.join(y_test_data_df)
    x_prediction_dataset_df = pd.DataFrame(x_prediction_dataset)
    # Predict the stock price
    forecast_df = predict_price(traning_dataset_df, test_dataset_df, x_prediction_dataset_df, stock_data_df)
    # Calculate the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds to build dataset and ML models.")
    # Plot the graph
    plot_graph(stock_data_df, forecast_df)
