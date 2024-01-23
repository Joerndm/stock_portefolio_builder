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

# start_time = time.time()

# import stock_data_fetch

# Import stock symbols from a CSV file
def import_stock_data(csv_file):
    """
    Imports stock symbols from a CSV file and returns a pandas DataFrame.

    The CSV file should have a column named 'Symbol' containing the stock symbols.

    Parameters:
    - csv_file (str): The path to the CSV file.

    Returns:
    pandas.DataFrame: A DataFrame containing the imported stock symbols.

    Raises:
    - FileNotFoundError: If the specified CSV file does not exist.
    - KeyError: If the CSV file does not have a column named 'Symbol'.
    """

    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)

        # Check if the 'Symbol' column exists in the DataFrame
        if 'Date' not in df.columns:
            raise KeyError("CSV file does not have a column named 'Date'.")

        # Return the DataFrame with stock symbols
        return df

    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file '{csv_file}' does not exist.")

# Predict the stock price
def predict_price(stock_df):
    train_data_df = stock_df[[
        "Date", "Name", "Ticker", "Currency", "Price", "2Y", "3Y", "4Y", "5Y",
        "SMA_40", "SMA_120", "EMA_40", "EMA_120", "Revenue", "Revenue growth",
        "P/S", "Current Ratio", "Current Ratio growth", "P/B", "P/FCF"
    ]]
    train_data_df = train_data_df.drop(["Date", "Name", "Ticker", "Currency"], axis=1)
    forecast_out = int(math.ceil(0.05 * len(train_data_df)))
    prediction_data_dict = {"Prediction": train_data_df["Price"].shift(-forecast_out).values}
    prediction_data_df = pd.DataFrame(prediction_data_dict)
    train_data_df = train_data_df.join(prediction_data_df)
    scaler = MinMaxScaler()
    x = np.array(train_data_df.drop(["Price", "Prediction"], axis=1))
    scaled_x = scaler.fit_transform(x)
    x_Predictions = scaled_x[-forecast_out:]
    print(len(x_Predictions))
    x = scaled_x[:-forecast_out]
    print(len(x))
    train_data_df = train_data_df.dropna(axis=0, how="any")
    y = np.array(train_data_df["Prediction"])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    lr_confidence = lr.score(x_test, y_test)
    lr_mean_absolute_error = mean_absolute_error(y_test, lr.predict(x_test))
    rf = RandomForestRegressor()
    rf.fit(x_train, y_train)
    rf_confidence = rf.score(x_test, y_test)
    rf_mean_absolute_error = mean_absolute_error(y_test, rf.predict(x_test))
    rg = Ridge()
    rg.fit(x_train, y_train)
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
    svm = SVR()
    svm.fit(x_train, y_train)
    svm_confidence = svm.score(x_test, y_test)
    svm_mean_absolute_error = mean_absolute_error(y_test, svm.predict(x_test))
    predict_precision_df = pd.DataFrame({"Model": ["Linear Regression", "Random Forest", "Ridge", "Support Vector Regression"],
        "Confidence": [lr_confidence, rf_confidence, rg_confidence, svm_confidence],
        "Mean Absolute Error": [lr_mean_absolute_error, rf_mean_absolute_error, rg_mean_absolute_error, svm_mean_absolute_error]
    })
    print(predict_precision_df)
    forecast_set_lr = lr.predict(x_Predictions)
    forecast_set_rf = rf.predict(x_Predictions)
    forecast_set_rg = rg.predict(x_Predictions)
    forecast_set_svm = svm.predict(x_Predictions)
    date_list = []
    for i in range(len(x_Predictions)):
        x = len(x_Predictions) - i
        date = stock_df["Date"].values[-x]
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
        date_list.append(date)


    forecast_dict = {"Date":date_list, "Price_lr":forecast_set_lr,
                    "Price_rf":forecast_set_rf, "Price_rg":forecast_set_rg,
                    "Price_svm":forecast_set_svm
    }
    forecast_df = pd.DataFrame(forecast_dict)
    # print(forecast_df)
    predicted_return = ((forecast_df.iloc[-1]["Price_rg"] / forecast_df.iloc[0]["Price_rg"]) - 1) * 100
    if predicted_return > 0:
        print(f"The prediction expects a profitable return on: {predicted_return}%, over the next {len(forecast_df)} days.")
    elif predicted_return < 0:
        print(f"The prediction expects a loss of: {predicted_return}%, over the next {len(forecast_df)} days.")
    else:
        print(f"The prediction expects no return over the next {len(forecast_df)} days.")


    return forecast_df

# Plot the graph
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
    forecast_data_df["Price_svm"].plot()
    plt.legend([
        "Stock Price", "Linear Regression", "Random Forest",
        "Ridge", "Support Vector Machine"
        ],
        loc="upper right"
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
    try:
        plt.savefig(os.path.join(path, "generated_graphs", graph_name), bbox_inches="tight", pad_inches=0.5, transparent=False, format="png")


    except FileNotFoundError:
        raise FileNotFoundError("The graph could not be saved. Please check the file name or path.")
    # Show the graph
    # plt.show()


# # Import stock data from a CSV file
# stock_data_df = import_stock_data('stock_data_single_v2.csv')
# # Predict the stock price
# forecast_df = predict_price(stock_data_df)
# # end_time = time.time()
# # execution_time = end_time - start_time
# # print(f"Execution time: {execution_time} seconds to build dataset and ML models.")
# # Plot the graph
# plot_graph(stock_data_df, forecast_df)
