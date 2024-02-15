import os
import pandas as pd
import numpy
import matplotlib.pyplot as pyplot
import datetime
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn import tree
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import time

import stock_data_fetch
import import_csv_file
import split_dataset
import dimension_reduction
import monte_carlo_sim

# Fits a linear regression model to the traning dataset and predicts the stock price
def linear_model(traning_dataset, test_dataset, prediction_dataset, stock_df):
    """
    Predicts the stock price using a linear regression model.
    
    Parameters:
    - traning_dataset (pandas.DataFrame): A DataFrame containing the traning dataset.
    - test_dataset (pandas.DataFrame): A DataFrame containing the test dataset.
    - prediction_dataset (pandas.DataFrame): A DataFrame containing the prediction dataset.
    - stock_df (pandas.DataFrame): A DataFrame containing the stock data.
    
    Returns:
    pandas.DataFrame: A DataFrame containing the predicted stock prices.
    
    Raises:
    None
    """
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
    lr_mean_squared_error = mean_squared_error(y_test, lr.predict(x_test))
    predict_precision_df = pd.DataFrame({"Model": ["Linear Regression"],
        "Confidence": [lr_confidence],
        "Mean Absolute Error": [lr_mean_absolute_error],
        "Mean Squared Error": [lr_mean_squared_error]
    })
    print(predict_precision_df)
    # Print the fitted model's weight and intercept
    # print(f"Weight: {lr.coef_}")
    # print(f"Intercept: {lr.intercept_}")
    # # Print the model's equation
    # print("The equation of the price prediction model is: ")
    # print("y = ")
    # for feature in range(len(lr.coef_)):
    #     print(f"{lr.coef_[feature]} * pca component number {x_training_df.columns[feature]+1} + ")
        

    # print(f"{lr.intercept_}")
    forecast_set_lr = lr.predict(x_prediction)
    date_list = []
    for i in range(len(x_prediction)):
        x = len(x_prediction) - i
        date = stock_df["Date"].values[-x]
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
        date_list.append(date)


    forecast_dict = {"Date":date_list, "Price_lr":forecast_set_lr
    }
    forecast_df = pd.DataFrame(forecast_dict)
    forecast_df = forecast_df.set_index("Date")
    # print(forecast_df)
    predicted_return = ((forecast_df.iloc[-1]["Price_lr"] / forecast_df.iloc[0]["Price_lr"]) - 1) * 100
    if predicted_return > 0:
        print(f"The prediction expects a profitable return on: {predicted_return}%, over the next {len(forecast_df)} days.")
    elif predicted_return < 0:
        print(f"The prediction expects a loss of: {predicted_return}%, over the next {len(forecast_df)} days.")
    else:
        print(f"The prediction expects no return over the next {len(forecast_df)} days.")


    return forecast_df

# Fits a random forest regressor model to the traning dataset and predicts the stock price
def random_forest_regressor_model(traning_dataset, test_dataset, prediction_dataset, stock_df):
    """
    Predicts the stock price using a random forest regressor model.

    Parameters:
    - traning_dataset (pandas.DataFrame): A DataFrame containing the traning dataset.
    - test_dataset (pandas.DataFrame): A DataFrame containing the test dataset.
    - prediction_dataset (pandas.DataFrame): A DataFrame containing the prediction dataset.
    - stock_df (pandas.DataFrame): A DataFrame containing the stock data.

    Returns:
    pandas.DataFrame: A DataFrame containing the predicted stock prices.

    Raises:
    None    
    """
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
    rf = RandomForestRegressor()
    rf.fit(x_training, y_training)
    rf_confidence = rf.score(x_test, y_test)
    rf_mean_absolute_error = mean_absolute_error(y_test, rf.predict(x_test))
    # Print the fitted model's weight and intercept
    rf_confidence = rf.score(x_test, y_test)
    rf_mean_absolute_error = mean_absolute_error(y_test, rf.predict(x_test))
    rf_mean_squared_error = mean_squared_error(y_test, rf.predict(x_test))
    predict_precision_df = pd.DataFrame({"Model": ["Random Forest"],
        "Confidence": [rf_confidence],
        "Mean Absolute Error": [rf_mean_absolute_error],
        "Mean Squared Error": [rf_mean_squared_error]
    })
    print(predict_precision_df)
    forecast_set_rf = rf.predict(x_prediction)
    date_list = []
    for i in range(len(x_prediction)):
        x = len(x_prediction) - i
        date = stock_df["Date"].values[-x]
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
        date_list.append(date)


    forecast_dict = {"Date":date_list, "Price_rf":forecast_set_rf
    }
    forecast_df = pd.DataFrame(forecast_dict)
    forecast_df = forecast_df.set_index("Date")
    # print(forecast_df)
    predicted_return = ((forecast_df.iloc[-1]["Price_rf"] / forecast_df.iloc[0]["Price_rf"]) - 1) * 100
    if predicted_return > 0:
        print(f"The prediction expects a profitable return on: {predicted_return}%, over the next {len(forecast_df)} days.")
    elif predicted_return < 0:
        print(f"The prediction expects a loss of: {predicted_return}%, over the next {len(forecast_df)} days.")
    else:
        print(f"The prediction expects no return over the next {len(forecast_df)} days.")


    return forecast_df

# Fits a ridge model to the traning dataset and predicts the stock price
def ridge_model(traning_dataset, test_dataset, prediction_dataset, stock_df):
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
    rg = Ridge()
    rg.fit(x_training, y_training)
    rg_confidence = rg.score(x_test, y_test)
    rg_mean_absolute_error = mean_absolute_error(y_test, rg.predict(x_test))
    rg_mean_squared_error = mean_squared_error(y_test, rg.predict(x_test))
    predict_precision_df = pd.DataFrame({"Model": ["Ridge Regression"],
        "Confidence": [rg_confidence],
        "Mean Absolute Error": [rg_mean_absolute_error],
        "Mean Squared Error": [rg_mean_squared_error]
    })
    print(predict_precision_df)
    # Print the fitted model's weight and intercept
    # print(f"Weight: {rg.coef_}")
    # print(f"Intercept: {rg.intercept_}")
    # # Print the model's equation
    # print("The equation of the price prediction model is: ")
    # print("y = ")
    # for feature in range(len(rg.coef_)):
    #     print(f"{rg.coef_[feature]} * pca component number {x_training_df.columns[feature]+1} + ")
        

    # print(f"{rg.intercept_}")
    forecast_set_rg = rg.predict(x_prediction)
    date_list = []
    for i in range(len(x_prediction)):
        x = len(x_prediction) - i
        date = stock_df["Date"].values[-x]
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
        date_list.append(date)


    forecast_dict = {"Date":date_list, "Price_rg":forecast_set_rg
    }
    forecast_df = pd.DataFrame(forecast_dict)
    forecast_df = forecast_df.set_index("Date")
    # print(forecast_df)
    predicted_return = ((forecast_df.iloc[-1]["Price_rg"] / forecast_df.iloc[0]["Price_rg"]) - 1) * 100
    if predicted_return > 0:
        print(f"The prediction expects a profitable return on: {predicted_return}%, over the next {len(forecast_df)} days.")
    elif predicted_return < 0:
        print(f"The prediction expects a loss of: {predicted_return}%, over the next {len(forecast_df)} days.")
    else:
        print(f"The prediction expects no return over the next {len(forecast_df)} days.")


    return forecast_df

# Fits a support vector machine model to the traning dataset and predicts the stock price
def svm_model(traning_dataset, test_dataset, prediction_dataset, stock_df, kernel_type):
    """
    Predicts the stock price using a support vector machine model.

    Parameters:
    - traning_dataset (pandas.DataFrame): A DataFrame containing the traning dataset.
    - test_dataset (pandas.DataFrame): A DataFrame containing the test dataset.
    - prediction_dataset (pandas.DataFrame): A DataFrame containing the prediction dataset.
    - stock_df (pandas.DataFrame): A DataFrame containing the stock data.
    - kernel_type (str): The kernel type.

    Returns:
    pandas.DataFrame: A DataFrame containing the predicted stock prices.

    Raises:
    None    
    """
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
    svm = SVR(kernel=kernel_type)
    svm.fit(x_training, y_training)
    svm_confidence = svm.score(x_test, y_test)
    svm_mean_absolute_error = mean_absolute_error(y_test, svm.predict(x_test))
    svm_mean_squared_error = mean_squared_error(y_test, svm.predict(x_test))
    predict_precision_df = pd.DataFrame({"Model": ["Support Vector Machine"],
        "Confidence": [svm_confidence],
        "Mean Absolute Error": [svm_mean_absolute_error],
        "Mean Squared Error": [svm_mean_squared_error]
    })
    print(predict_precision_df)
    forecast_set_svm = svm.predict(x_prediction)
    date_list = []
    for i in range(len(x_prediction)):
        x = len(x_prediction) - i
        date = stock_df["Date"].values[-x]
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
        date_list.append(date)


    forecast_dict = {"Date":date_list, "Price_svm":forecast_set_svm
    }
    forecast_df = pd.DataFrame(forecast_dict)
    forecast_df = forecast_df.set_index("Date")
    # print(forecast_df)
    predicted_return = ((forecast_df.iloc[-1]["Price_svm"] / forecast_df.iloc[0]["Price_svm"]) - 1) * 100
    if predicted_return > 0:
        print(f"The prediction expects a profitable return on: {predicted_return}%, over the next {len(forecast_df)} days.")
    elif predicted_return < 0:
        print(f"The prediction expects a loss of: {predicted_return}%, over the next {len(forecast_df)} days.")
    else:
        print(f"The prediction expects no return over the next {len(forecast_df)} days.")


    return forecast_df

# Fits a decision tree model to the traning dataset and predicts the stock price
def decision_tree_model(traning_dataset, test_dataset, prediction_dataset, stock_df):
    """
    Predicts the stock price using a decision tree model.

    Parameters:
    - traning_dataset (pandas.DataFrame): A DataFrame containing the traning dataset.
    - test_dataset (pandas.DataFrame): A DataFrame containing the test dataset.
    - prediction_dataset (pandas.DataFrame): A DataFrame containing the prediction dataset.
    - stock_df (pandas.DataFrame): A DataFrame containing the stock data.

    Returns:
    pandas.DataFrame: A DataFrame containing the predicted stock prices.

    Raises:
    None
    """
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
    dt = tree.DecisionTreeRegressor(max_depth=10)
    dt.fit(x_training, y_training)
    dt_confidence = dt.score(x_test, y_test)
    dt_mean_absolute_error = mean_absolute_error(y_test, dt.predict(x_test))
    dt_mean_squared_error = mean_squared_error(y_test, dt.predict(x_test))
    predict_precision_df = pd.DataFrame({"Model": ["Decision Tree"],
        "Confidence": [dt_confidence],
        "Mean Absolute Error": [dt_mean_absolute_error],
        "Mean Squared Error": [dt_mean_squared_error]
    })
    print(predict_precision_df)
    forecast_set_dt = dt.predict(x_prediction)
    date_list = []
    for i in range(len(x_prediction)):
        x = len(x_prediction) - i
        date = stock_df["Date"].values[-x]
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
        date_list.append(date)


    forecast_dict = {"Date":date_list, "Price_dt":forecast_set_dt
    }
    forecast_df = pd.DataFrame(forecast_dict)
    forecast_df = forecast_df.set_index("Date")
    # print(forecast_df)
    predicted_return = ((forecast_df.iloc[-1]["Price_dt"] / forecast_df.iloc[0]["Price_dt"]) - 1) * 100
    if predicted_return > 0:
        print(f"The prediction expects a profitable return on: {predicted_return}%, over the next {len(forecast_df)} days.")
    elif predicted_return < 0:
        print(f"The prediction expects a loss of: {predicted_return}%, over the next {len(forecast_df)} days.")
    else:
        print(f"The prediction expects no return over the next {len(forecast_df)} days.")


    return forecast_df

# Fits a neural network model to the traning dataset and predicts the stock price
def neural_network_model(traning_dataset, test_dataset, prediction_dataset, hiddenLayer_1, hiddenLayer_2, hiddenLayer_3, hiddenLayer_4, iterations, randomState, stock_df):
    """
    Predicts the stock price using a neural network model.

    Parameters:
    - traning_dataset (pandas.DataFrame): A DataFrame containing the traning dataset.
    - test_dataset (pandas.DataFrame): A DataFrame containing the test dataset.
    - prediction_dataset (pandas.DataFrame): A DataFrame containing the prediction dataset.
    - hiddenLayer_1 (int): The number of neurons in the first hidden layer.
    - hiddenLayer_2 (int): The number of neurons in the second hidden layer.
    - hiddenLayer_3 (int): The number of neurons in the third hidden layer.
    - hiddenLayer_4 (int): The number of neurons in the fourth hidden layer.
    - iterations (int): The number of iterations.
    - randomState (int): The random state.

    Returns:
    pandas.DataFrame: A DataFrame containing the predicted stock prices.

    Raises:
    - ValueError: If the prediction could not be completed.
    """

    try:
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
        nn = MLPRegressor(alpha=1e-5, hidden_layer_sizes=(hiddenLayer_1, hiddenLayer_2, hiddenLayer_3, hiddenLayer_4), max_iter=iterations, random_state=randomState,)
        nn.fit(x_training, y_training)
        nn_confidence = nn.score(x_test, y_test)
        nn_mean_absolute_error = mean_absolute_error(y_test, nn.predict(x_test))
        nn_mean_squared_error = mean_squared_error(y_test, nn.predict(x_test))
        predict_precision_df = pd.DataFrame({"Model": ["Neural Network"],
            "Confidence": [nn_confidence],
            "Mean Absolute Error": [nn_mean_absolute_error],
            "Mean Squared Error": [nn_mean_squared_error]
        })
        print(predict_precision_df)
        # ['1M', '3M', '6M', '9M', '1Y', '2Y', '3Y', '4Y',
        # '5Y', 'SMA_40', 'SMA_120', 'EMA_40', 'EMA_120',
        # 'EPS', 'EPS growth', 'P/S', 'P/E', 'P/B', 'P/FCF'
        # ]
        # Loop through each row in stock_price_data_df
        forecast_set_nn = []
        date_list = []
        for index, row in prediction_dataset.iterrows():
            if index == 0:
                continue    
            elif index > 0 and index != len(prediction_dataset) - 1:
                # Convert the DataFrame to a numpy array
                x_prediction = np.array(prediction_dataset.iloc[index-1:index+1])
                forecast = nn.predict(x_prediction)
                forecast_price = forecast[0]
                forecast_set_nn.append(forecast_price)
                date = stock_df.loc[index, "Date"]
                date = datetime.datetime.strptime(date, '%Y-%m-%d')
                date_list.append(date)
            elif index == len(prediction_dataset) - 1:
                x_prediction = np.array(prediction_dataset.iloc[index-1:index+1])
                forecast = nn.predict(x_prediction)
                for price in range(len(forecast)):
                    forecast_price = forecast[price]
                    forecast_set_nn.append(forecast_price)


        i = len(prediction_dataset)
        date_df = stock_df.iloc[-i:]["Date"].copy()
        # Reset the index of the DataFrame
        date_df = date_df.reset_index(drop=True)
        # Convert values in Date column to datetime
        date_df = date_df.astype('datetime64[ns]')
        forecast_dict = {"Price_nn":forecast_set_nn
        }
        forecast_df = pd.DataFrame(forecast_dict)
        forecast_df = forecast_df.join(date_df)
        forecast_df = forecast_df.set_index("Date")
        # Create variable storing the model's equation
        # model_equation = "y = "
        # for feature in range(len(nn.coefs_[0])):
        #     model_equation += f"{nn.coefs_[0][feature]} * {x_training_df.columns[feature]} + "
        # model_equation += f"{nn.intercepts_[0]}"
        # # Print the model's equation
        # print("The equation of the price prediction model is: ")
        # print(model_equation)
        predicted_return = ((forecast_df.iloc[-1]["Price_nn"] / forecast_df.iloc[0]["Price_nn"]) - 1) * 100
        if predicted_return > 0:
            print(f"The prediction expects a profitable return on: {predicted_return}%, over the next {len(forecast_df)} days.")
        elif predicted_return < 0:
            print(f"The prediction expects a loss of: {predicted_return}%, over the next {len(forecast_df)} days.")
        else:
            print(f"The prediction expects no return over the next {len(forecast_df)} days.")


        return forecast_df
    

    except ValueError:
        raise ValueError("The prediction could not be completed. Please check the input data.")

# Combines the predicted stock prices from different models
def predict_price(traning_dataset, test_dataset, prediction_dataset, stock_df):
    """
    Predicts the stock price using different models.

    Parameters:
    - traning_dataset (pandas.DataFrame): A DataFrame containing the traning dataset.
    - test_dataset (pandas.DataFrame): A DataFrame containing the test dataset.
    - prediction_dataset (pandas.DataFrame): A DataFrame containing the prediction dataset.
    - stock_df (pandas.DataFrame): A DataFrame containing the stock data.

    Returns:
    pandas.DataFrame: A DataFrame containing the predicted stock prices.

    Raises:
    ValueError: If the prediction could not be completed.
    """

    try:
        # lr_forecast = linear_model(traning_dataset, test_dataset, prediction_dataset, stock_df)
        # print(lr_forecast)
        # rf_forecast = random_forest_regressor_model(traning_dataset, test_dataset, prediction_dataset, stock_df)
        # # print(rf_forecast)
        # forecast_df = lr_forecast.join(rf_forecast)
        # rd_forecast = ridge_model(traning_dataset, test_dataset, prediction_dataset, stock_df)
        # # print(rd_forecast)
        # forecast_df = forecast_df.join(rd_forecast)
        # print(forecast_df)
        # kernel_list = ["linear"]
        # kernel_list = ["poly"]
        # # kernel_list = ["linear", "poly"]
        # for kernel_type in kernel_list:
        #     svm_forecast = svm_model(traning_dataset, test_dataset, prediction_dataset, stock_df, kernel_type)
        #     # print(svm_forecast)
        #     forecast_df = lr_forecast.join(svm_forecast)
        #     # forecast_df = forecast_df.join(svm_forecast)
        #     forecast_df.rename(columns={"Price_svm":f"Price_svm_{kernel_type}"}, inplace=True)


        # dt_forecast = decision_tree_model(traning_dataset, test_dataset, prediction_dataset, stock_df)
        # # print(dt_forecast)
        # forecast_df = forecast_df.join(dt_forecast)
        nn_forecast = neural_network_model(traning_dataset, test_dataset, prediction_dataset, 70, 60, 50, 40, 100, 1, stock_df)
        # print(nn_forecast)
        forecast_df = nn_forecast
        # forecast_df = forecast_df.join(nn_forecast)
        print(forecast_df)
        forecast_df = forecast_df.reset_index()
        return forecast_df
    
    except ValueError:
        raise ValueError("The prediction could not be completed. Please check the input data.")

# Plots a graph of the stock data
def plot_graph(stock_data_df, forecast_data_df):
    """
    Plots a graph of the stock data.

    Parameters:
    - stock_data_df (pandas.DataFrame): A DataFrame containing the stock data.

    Returns:
    None

    Raises:
    - FileNotFoundError: If the graph could not be saved.
    """

    # Plot the graph
    plt.figure(figsize=(18, 8))
    stock_data_df["Date"] = stock_data_df["Date"].astype('datetime64[ns]')
    stock_data_df = stock_data_df.set_index("Date")
    forecast_data_df["Date"] = forecast_data_df["Date"].astype('datetime64[ns]')
    forecast_data_df = forecast_data_df.set_index("Date")
    stock_data_df["Price"].plot()
    for column in forecast_data_df.columns:
        forecast_data_df[column].plot()


    legend_list = ["Stock Price"]
    for column in forecast_data_df.columns:
        legend_list.append(column)


    plt.legend(legend_list,
        loc="best"
    )
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
        plt.clf()
        plt.close("all")


    except FileNotFoundError:
        raise FileNotFoundError("The graph could not be saved. Please check the file name or path.")
    

    # Show the graph
    # plt.show()

# Run the main function
if __name__ == "__main__":
    start_time = time.time()
    stock_symbols_df = stock_data_fetch.import_symbols('index_symbol_list_single_stock.csv')
    stock_symbols_list = stock_symbols_df['Symbol'].tolist()
    stock_symbol = stock_symbols_list[0]
    print(stock_symbol)
    # # Fetch stock data for the imported stock symbols
    # stock_price_data_df = stock_data_fetch.fetch_stock_price_data(stock_symbol)
    # stock_price_data_df = stock_data_fetch.calculate_period_returns(stock_price_data_df)
    # stock_price_data_df = stock_data_fetch.calculate_moving_averages(stock_price_data_df)
    # stock_price_data_df = stock_data_fetch.calculate_standard_diviation_value(stock_price_data_df)
    # # Fetch stock data for the imported stock symbols
    # full_stock_financial_data_df = stock_data_fetch.fetch_stock_financial_data(stock_symbol)
    # # Combine stock data with stock financial data
    # combined_stock_data_df = stock_data_fetch.combine_stock_data(stock_price_data_df, full_stock_financial_data_df)
    # # Calculate ratios
    # combined_stock_data_df = stock_data_fetch.calculate_ratios(combined_stock_data_df)
    # # Create a dictionary of dataframes to export to Excel
    # dataframes = {
    #     "Combined Stock Data": combined_stock_data_df
    # }
    # # Export the dataframes to an Excel file
    # stock_data_fetch.export_to_excel(dataframes, 'stock_data_single.xlsx')
    # # Import the stock data from an Excel file
    # dataframes = stock_data_fetch.import_excel("stock_data_single.xlsx")
    # for key, value in dataframes.items():
    #     dataframe = value


    # # Export the stock data to a CSV file
    # stock_data_fetch.convert_excel_to_csv(dataframe, "stock_data_single")
    stock_data_df = import_csv_file.import_as_df('stock_data_single.csv')
    # Split the dataset into traning, test data and prediction data
    x_training_data, x_test_data, y_training_data, y_test_data, prediction_data = split_dataset.dataset_train_test_split(stock_data_df, 0.20, 1)
    # Feature selection
    x_training_dataset, x_test_dataset, x_prediction_dataset, selected_features_list = dimension_reduction.feature_selection(15, x_training_data, x_test_data, y_training_data, y_test_data, prediction_data, stock_data_df)
    # Combine the reduced dataset with the stock price
    x_training_dataset_df = pd.DataFrame(x_training_dataset, columns=selected_features_list)
    y_training_data_df = pd.DataFrame(y_training_data, columns=["Price"])
    traning_dataset_df = x_training_dataset_df.join(y_training_data_df)
    x_test_dataset_df = pd.DataFrame(x_test_dataset, columns=selected_features_list)
    y_test_data_df = pd.DataFrame(y_test_data, columns=["Price"])
    test_dataset_df = x_test_dataset_df.join(y_test_data_df)
    x_prediction_dataset_df = pd.DataFrame(x_prediction_dataset, columns=selected_features_list)
    # Predict the stock price
    forecast_df = predict_price(traning_dataset_df, test_dataset_df, x_prediction_dataset_df, stock_data_df)
    # Plot the graph
    plot_graph(stock_data_df, forecast_df)
    # Calculate the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds to build dataset and ML models.")
    monte_carlo_df = monte_carlo_sim.monte_carlo_analysis_3(0, stock_data_df, forecast_df, 5, 1000)
    print(monte_carlo_df)
    print(monte_carlo_df.describe())
    print(monte_carlo_df.info())

