import os
import pandas as pd
import time
import datetime
from dateutil.relativedelta import relativedelta
import yfinance as yf
import numpy as np
import matplotlib
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

matplotlib.use('Agg')

import stock_data_fetch
import import_stock_data
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


    forecast_dict = {"Date":date_list, "Prediction_lr":forecast_set_lr
    }
    forecast_df = pd.DataFrame(forecast_dict)
    forecast_df = forecast_df.set_index("Date")
    # print(forecast_df)
    predicted_return = ((forecast_df.iloc[-1]["Prediction_lr"] / forecast_df.iloc[0]["Prediction_lr"]) - 1) * 100
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


    forecast_dict = {"Date":date_list, "Prediction_rf":forecast_set_rf
    }
    forecast_df = pd.DataFrame(forecast_dict)
    forecast_df = forecast_df.set_index("Date")
    # print(forecast_df)
    predicted_return = ((forecast_df.iloc[-1]["Prediction_rf"] / forecast_df.iloc[0]["Prediction_rf"]) - 1) * 100
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


    forecast_dict = {"Date":date_list, "Prediction_rg":forecast_set_rg
    }
    forecast_df = pd.DataFrame(forecast_dict)
    forecast_df = forecast_df.set_index("Date")
    # print(forecast_df)
    predicted_return = ((forecast_df.iloc[-1]["Prediction_rg"] / forecast_df.iloc[0]["Prediction_rg"]) - 1) * 100
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


    forecast_dict = {"Date":date_list, "Prediction_svm":forecast_set_svm
    }
    forecast_df = pd.DataFrame(forecast_dict)
    forecast_df = forecast_df.set_index("Date")
    # print(forecast_df)
    predicted_return = ((forecast_df.iloc[-1]["Prediction_svm"] / forecast_df.iloc[0]["Prediction_svm"]) - 1) * 100
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


    forecast_dict = {"Date":date_list, "Prediction_dt":forecast_set_dt
    }
    forecast_df = pd.DataFrame(forecast_dict)
    forecast_df = forecast_df.set_index("Date")
    # print(forecast_df)
    predicted_return = ((forecast_df.iloc[-1]["Prediction_dt"] / forecast_df.iloc[0]["Prediction_dt"]) - 1) * 100
    if predicted_return > 0:
        print(f"The prediction expects a profitable return on: {predicted_return}%, over the next {len(forecast_df)} days.")
    elif predicted_return < 0:
        print(f"The prediction expects a loss of: {predicted_return}%, over the next {len(forecast_df)} days.")
    else:
        print(f"The prediction expects no return over the next {len(forecast_df)} days.")


    return forecast_df

# Fits a neural network model to the traning dataset and predicts the stock price
def neural_network_model(traning_dataset, test_dataset, hiddenLayer_1, hiddenLayer_2, hiddenLayer_3, hiddenLayer_4, iterations, randomState):
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
    - ValueError: If the model could not be generated.
    """

    try:
        x_training_df = traning_dataset.drop(["Prediction"], axis=1)
        y_training_df = traning_dataset["Prediction"]
        # Convert the DataFrame to a numpy array
        x_training = np.array(x_training_df)
        y_training = np.array(y_training_df)
        x_test_df = test_dataset.drop(["Prediction"], axis=1)
        y_test_df = test_dataset["Prediction"]
        # Convert the DataFrame to a numpy array
        x_test = np.array(x_test_df)
        y_test = np.array(y_test_df)
        nn = MLPRegressor(alpha=1e-5, hidden_layer_sizes=(hiddenLayer_1, hiddenLayer_2, hiddenLayer_3, hiddenLayer_4), max_iter=iterations, random_state=randomState,)
        nn.fit(x_training, y_training)
        nn_confidence = nn.score(x_test, y_test)
        nn_mean_absolute_error = mean_absolute_error(y_test, nn.predict(x_test))
        nn_mean_squared_error = mean_squared_error(y_test, nn.predict(x_test))
        nn_r2_score = r2_score(y_test, nn.predict(x_test))
        predict_precision_df = pd.DataFrame({"Model": ["Neural Network"],
            "Confidence": [nn_confidence],
            "Mean Absolute Error": [nn_mean_absolute_error],
            "Mean Squared Error": [nn_mean_squared_error],
            "R2 Score": [nn_r2_score]
        }).transpose()
        print(predict_precision_df)
        # # Create variable storing the model's equation
        # model_equation = "y = "
        # for feature in range(len(nn.coefs_[0])):
        #     model_equation += f"{nn.coefs_[0][feature]} * {x_training_df.columns[feature]} + "
        # model_equation += f"{nn.intercepts_[0]}"
        # # Print the model's equation
        # print("The equation of the price prediction model is: ")
        # print(model_equation)
        return nn
    

    except ValueError:
        raise ValueError("The model could not be generated. Please check the input data.")

# Predicts the future stock price
def predict_future_price_changes(ticker, scaler, model, selected_features_list, stock_df, prediction_days):
    """
    Predicts the future stock price.

    Parameters:
    - ticker (str): The stock ticker.
    - scaler (object): The scaler object.
    - model (object): The model object.
    - selected_features_list (list): The list of selected features.
    - stock_df (pandas.DataFrame): A DataFrame containing the stock data.
    - prediction_days (int): The number of days to predict.

    Returns:
    pandas.DataFrame: A DataFrame containing the predicted stock prices.

    Raises:
    - ValueError: If the prediction could not be completed.
    """
    try:
        # Predict the future stock price
        short_term_dynamic_list = [
            '1M', '3M', '6M', '9M', '1Y', '2Y', '3Y', '4Y',
            '5Y', 'SMA_40', 'SMA_120', 'EMA_40', 'EMA_120',
            "STD_Div_40", "STD_Div_120", "Bollinger_Band_40",
            "Bollinger_Band_120", 'P/S', 'P/E', 'P/B',
            'P/FCF', "Momentum"
        ]
        print(f"selected_features_list: {selected_features_list}")
        features_list = selected_features_list.copy()
        # Append "Date" to the selected_features_list in position 0
        features_list.insert(0, "Date")
        features_list.insert(1, "Price")
        features_list.insert(2, "1D")
        stock_mod_df = stock_df.copy()
        for run in range(prediction_days):
            future_df = stock_mod_df.iloc[-1].copy().to_frame().transpose()
            for feature in range(len(short_term_dynamic_list)):
                if short_term_dynamic_list[feature] in features_list:
                    future_day = pd.to_datetime(stock_mod_df.iloc[-1]["Date"]) + relativedelta(days=1)
                    if future_day.weekday() == 5:
                        future_day = future_day + datetime.timedelta(days=2)
                        future_day = future_day.strftime("%Y-%m-%d")
                        future_df["Date"] = str(future_day)
                    elif future_day.weekday() == 6:
                        future_day = future_day + datetime.timedelta(days=1)
                        future_day = future_day.strftime("%Y-%m-%d")
                        future_df["Date"] = str(future_day)
                    else:
                        future_day = future_day.strftime("%Y-%m-%d")
                        future_df["Date"] = str(future_day)

                    # Calculate the return for the last 1 month
                    if short_term_dynamic_list[feature] == "1M":
                        if len(stock_mod_df) <= 21:
                            hist_df = pd.DataFrame(yf.download(ticker, period="6y"))["Open"].reset_index()
                            hist_df = hist_df.loc[hist_df["Date"] < stock_mod_df.iloc[0]["Date"]]
                            hist_df = hist_df.rename(columns={"Open": "Price"})
                            hist_df = pd.concat([hist_df, stock_mod_df[["Date", "Price"]]], axis=0)
                            return_1M = hist_df.iloc[-22:]["Price"].pct_change(21).iloc[-1]
                            future_df["1M"] = return_1M
                        elif len(stock_mod_df) > 21:
                            return_1M = stock_mod_df.iloc[-22:]["Price"].pct_change(21).iloc[-1]
                            future_df["1M"] = return_1M

                    # Calculate the return for the last 3 months
                    if short_term_dynamic_list[feature] == "3M":
                        if len(stock_mod_df) <= 63:
                            hist_df = pd.DataFrame(yf.download(ticker, period="6y"))["Open"].reset_index()
                            hist_df = hist_df.loc[hist_df["Date"] < stock_mod_df.iloc[0]["Date"]]
                            hist_df = hist_df.rename(columns={"Open": "Price"})
                            hist_df = pd.concat([hist_df, stock_mod_df[["Date", "Price"]]], axis=0)
                            return_3M = hist_df.iloc[-64:]["Price"].pct_change(63).iloc[-1]
                            future_df["3M"] = return_3M
                        elif len(stock_mod_df) > 63:
                            return_3M = stock_mod_df.iloc[-64:]["Price"].pct_change(63).iloc[-1]
                            future_df["3M"] = return_3M

                    # Calculate the return for the last 6 months
                    if short_term_dynamic_list[feature] == "6M":
                        if len(stock_mod_df) <= 126:
                            hist_df = pd.DataFrame(yf.download(ticker, period="6y"))["Open"].reset_index()
                            hist_df = hist_df.loc[hist_df["Date"] < stock_mod_df.iloc[0]["Date"]]
                            hist_df = hist_df.rename(columns={"Open": "Price"})
                            hist_df = pd.concat([hist_df, stock_mod_df[["Date", "Price"]]], axis=0)
                            return_6M = hist_df.iloc[-127:]["Price"].pct_change(126).iloc[-1]
                            future_df["6M"] = return_6M
                        elif len(stock_mod_df) > 126:
                            return_6M = stock_mod_df.iloc[-127:]["Price"].pct_change(126).iloc[-1]
                            future_df["6M"] = return_6M

                    # Calculate the return for the last 9 months
                    if short_term_dynamic_list[feature] == "9M":
                        if len(stock_mod_df) <= 189:
                            hist_df = pd.DataFrame(yf.download(ticker, period="6y"))["Open"].reset_index()
                            hist_df = hist_df.loc[hist_df["Date"] < stock_mod_df.iloc[0]["Date"]]
                            hist_df = hist_df.rename(columns={"Open": "Price"})
                            hist_df = pd.concat([hist_df, stock_mod_df[["Date", "Price"]]], axis=0)
                            return_9M = hist_df.iloc[-190:]["Price"].pct_change(189).iloc[-1]
                            future_df["9M"] = return_9M
                        elif len(stock_mod_df) > 189:
                            return_9M = stock_mod_df.iloc[-190:]["Price"].pct_change(189).iloc[-1]
                            future_df["9M"] = return_9M

                    # Calculate the return for the last 1 year
                    if short_term_dynamic_list[feature] == "1Y":
                        if len(stock_mod_df) <= 252:
                            hist_df = pd.DataFrame(yf.download(ticker, period="6y"))["Open"].reset_index()
                            hist_df = hist_df.loc[hist_df["Date"] < stock_mod_df.iloc[0]["Date"]]
                            hist_df = hist_df.rename(columns={"Open": "Price"})
                            hist_df = pd.concat([hist_df, stock_mod_df[["Date", "Price"]]], axis=0)
                            return_1Y = hist_df.iloc[-253:]["Price"].pct_change(252).iloc[-1]
                            future_df["1Y"] = return_1Y
                        elif len(stock_mod_df) > 252:
                            return_1Y = stock_mod_df.iloc[-253:]["Price"].pct_change(252).iloc[-1]
                            future_df["1Y"] = return_1Y

                    # Calculate the return for the last 2 years
                    if short_term_dynamic_list[feature] == "2Y":
                        if len(stock_mod_df) <= 504:
                            hist_df = pd.DataFrame(yf.download(ticker, period="6y"))["Open"].reset_index()
                            hist_df = hist_df.loc[hist_df["Date"] < stock_mod_df.iloc[0]["Date"]]
                            hist_df = hist_df.rename(columns={"Open": "Price"})
                            hist_df = pd.concat([hist_df, stock_mod_df[["Date", "Price"]]], axis=0)
                            return_2Y = hist_df.iloc[-505:]["Price"].pct_change(504).iloc[-1]
                            future_df["2Y"] = return_2Y
                        elif len(stock_mod_df) > 504:
                            return_2Y = stock_mod_df.iloc[-505:]["Price"].pct_change(504).iloc[-1]
                            future_df["2Y"] = return_2Y

                    # Calculate the return for the last 3 years
                    if short_term_dynamic_list[feature] == "3Y":
                        if len(stock_mod_df) <= 756:
                            hist_df = pd.DataFrame(yf.download(ticker, period="6y"))["Open"].reset_index()
                            hist_df = hist_df.loc[hist_df["Date"] < stock_mod_df.iloc[0]["Date"]]
                            hist_df = hist_df.rename(columns={"Open": "Price"})
                            hist_df = pd.concat([hist_df, stock_mod_df[["Date", "Price"]]], axis=0)
                            return_3Y = hist_df.iloc[-757:]["Price"].pct_change(756).iloc[-1]
                            future_df["3Y"] = return_3Y
                        elif len(stock_mod_df) > 756:
                            return_3Y = stock_mod_df.iloc[-757:]["Price"].pct_change(756).iloc[-1]
                            future_df["3Y"] = return_3Y

                    # Calculate the return for the last 4 years
                    if short_term_dynamic_list[feature] == "4Y":
                        if len(stock_mod_df) <= 1008:
                            hist_df = pd.DataFrame(yf.download(ticker, period="6y"))["Open"].reset_index()
                            hist_df = hist_df.loc[hist_df["Date"] < stock_mod_df.iloc[0]["Date"]]
                            hist_df = hist_df.rename(columns={"Open": "Price"})
                            hist_df = pd.concat([hist_df, stock_mod_df[["Date", "Price"]]], axis=0)
                            return_4Y = hist_df.iloc[-1009:]["Price"].pct_change(1008).iloc[-1]
                            future_df["4Y"] = return_4Y
                        elif len(stock_mod_df) > 1008:
                            return_4Y = stock_mod_df.iloc[-1009:]["Price"].pct_change(1008).iloc[-1]
                            future_df["4Y"] = return_4Y

                    # Calculate the return for the last 5 years
                    if short_term_dynamic_list[feature] == "5Y":
                        if len(stock_mod_df) <= 1260:
                            hist_df = pd.DataFrame(yf.download(ticker, period="6y"))["Open"].reset_index()
                            hist_df = hist_df.loc[hist_df["Date"] < stock_mod_df.iloc[0]["Date"]]
                            hist_df = hist_df.rename(columns={"Open": "Price"})
                            hist_df = pd.concat([hist_df, stock_mod_df[["Date", "Price"]]], axis=0)
                            return_5Y = hist_df.iloc[-1261:]["Price"].pct_change(1260).iloc[-1]
                            future_df["5Y"] = return_5Y
                        elif len(stock_mod_df) > 1260:
                            return_5Y = stock_mod_df.iloc[-1261:]["Price"].pct_change(1260).iloc[-1]
                            future_df["5Y"] = return_5Y

                    # Calculate the 40 days simple moving average
                    if short_term_dynamic_list[feature] == "SMA_40":
                        sma_40 = stock_mod_df.iloc[-40:]["Price"].mean()
                        future_df["SMA_40"] = sma_40

                    # Calculate the 120 days simple moving average
                    if short_term_dynamic_list[feature] == "SMA_120":
                        sma_120 = stock_mod_df.iloc[-120:]["Price"].mean()
                        future_df["SMA_120"] = sma_120

                    # Calculate the 40 days exponential moving average
                    if short_term_dynamic_list[feature] == "EMA_40":
                        ema_40 = stock_mod_df.iloc[-40:]["Price"].ewm(span=40).mean().iloc[-1]
                        future_df["EMA_40"] = ema_40

                    # Calculate the 120 days exponential moving average
                    if short_term_dynamic_list[feature] == "EMA_120":
                        ema_120 = stock_mod_df.iloc[-120:]["Price"].ewm(span=120).mean().iloc[-1]
                        future_df["EMA_120"] = ema_120

                    # Calculate the 40 days standard deviation
                    if short_term_dynamic_list[feature] == "STD_Div_40":
                        std_div_40 = stock_mod_df.iloc[-40:]["Price"].std()
                        future_df["STD_Div_40"] = std_div_40

                    # Calculate the 120 days standard deviation
                    if short_term_dynamic_list[feature] == "STD_Div_120":
                        std_div_120 = stock_mod_df.iloc[-120:]["Price"].std()
                        future_df["STD_Div_120"] = std_div_120

                    # Calculate the Bollinger Band for the last 40 days
                    if short_term_dynamic_list[feature] == "Bollinger_Band_40":
                        std_div_40 = stock_mod_df.iloc[-40:]["Price"].std()
                        sma_40 = stock_mod_df.iloc[-40:]["Price"].mean()
                        bollinger_Band_40_Upper = (sma_40 + (std_div_40 * 2))
                        bollinger_Band_40_Lower = (sma_40 - (std_div_40 * 2))
                        bollinger_band_40 = bollinger_Band_40_Upper - bollinger_Band_40_Lower
                        future_df["Bollinger_Band_40"] = bollinger_band_40

                    # Calculate the Bollinger Band for the last 120 days
                    if short_term_dynamic_list[feature] == "Bollinger_Band_120":
                        std_div_120 = stock_mod_df.iloc[-120:]["Price"].std()
                        sma_120 = stock_mod_df.iloc[-120:]["Price"].mean()
                        bollinger_Band_120_Upper = (sma_120 + (std_div_120 * 2))
                        bollinger_Band_120_Lower = (sma_120 - (std_div_120 * 2))
                        bollinger_band_120 = bollinger_Band_120_Upper - bollinger_Band_120_Lower
                        future_df["Bollinger_Band_120"] = bollinger_band_120
                        
                    # Calculate the P/S ratio
                    if short_term_dynamic_list[feature] == "P/S":
                        p_s = stock_mod_df.iloc[-1]["Price"] / (stock_mod_df.iloc[-1]["Revenue"] / stock_mod_df.iloc[-1]["Amount of stocks"])
                        future_df["P/S"] = p_s

                    # Calculate the P/E ratio
                    if short_term_dynamic_list[feature] == "P/E":
                        p_e = stock_mod_df.iloc[-1]["Price"] / stock_mod_df.iloc[-1]["EPS"]
                        future_df["P/E"] = p_e

                    # Calculate the P/B ratio
                    if short_term_dynamic_list[feature] == "P/B":
                        p_b = stock_mod_df.iloc[-1]["Price"] / stock_mod_df.iloc[-1]["Book Value per share"]
                        future_df["P/B"] = p_b

                    # Calculate the P/FCF ratio
                    if short_term_dynamic_list[feature] == "P/FCF":
                        p_fcf = stock_mod_df.iloc[-1]["Price"] / stock_mod_df.iloc[-1]["Free Cash Flow per share growth"]
                        future_df["P/FCF"] = p_fcf

                    # Calculate the Momentum
                    if short_term_dynamic_list[feature] == "Momentum":
                        if stock_mod_df.iloc[-1]["Price"] >= stock_mod_df.iloc[-2]["Price"]:
                            if stock_mod_df.iloc[-1]["Momentum"] <= 0:
                                momentum = 1
                                # Update the Momentum column with the calculated value
                                future_df["Momentum"] = momentum
                            elif stock_mod_df.iloc[-1]["Momentum"] > 0:
                                momentum = stock_mod_df.iloc[-1]["Momentum"] + 1
                                # Update the Momentum column with the calculated value
                                future_df["Momentum"] = momentum
                        elif stock_mod_df.iloc[-1]["Price"] < stock_mod_df.iloc[-2]["Price"]:
                            if stock_mod_df.iloc[-1]["Momentum"] >= 0:
                                momentum = -1
                                # Update the Momentum column with the calculated value
                                future_df["Momentum"] = momentum
                            elif stock_mod_df.iloc[-1]["Momentum"] < 0:
                                momentum = stock_mod_df.iloc[-1]["Momentum"] - 1
                                # Update the Momentum column with the calculated value
                                future_df["Momentum"] = momentum
                    

            print(future_df)
            # Concat the pandas series as new line to the stock_mod_df
            prediction_df = pd.concat([stock_mod_df.iloc[-1].to_frame().transpose(), future_df], axis=0).reset_index(drop=True)
            prediction_df = prediction_df.drop(["Date", "Name", "Ticker", "Currency", "Price", "1D", "Trade volume", "Amount of stocks"], axis=1)
            # Concat the future_df to the stock_mod_df
            stock_mod_df = pd.concat([stock_mod_df, future_df], axis=0).reset_index(drop=True)
            # Scale the prediction_df
            scaled_prediction_df = scaler.transform(prediction_df)
            print(scaled_prediction_df)
            scaled_prediction_df = np.array(scaled_prediction_df[selected_features_list])
            # Predict the future stock price
            forecast = model.predict(scaled_prediction_df)
            forecast_price_change = forecast[1]
            # Update the 1D column with the calculated value
            stock_mod_df.loc[len(stock_mod_df)-1, "1D"] = forecast_price_change
            # Update the Price column with the calculated value
            stock_mod_df.loc[len(stock_mod_df)-1, "Price"] = stock_mod_df.loc[len(stock_mod_df)-2, "Price"] * (1 + forecast_price_change)


        stock_mod_df = stock_mod_df[features_list]
        print(stock_mod_df)
        return stock_mod_df


    except ValueError:
        raise ValueError("The prediction could not be completed. Please check the input data.")
    
# Combines the predicted stock prices from different models
def calculate_predicted_profit(forecast_df, prediction_days):
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
        predicted_return_df = forecast_df.loc[len(forecast_df)-prediction_days:, "Price"]
        print(predicted_return_df)
        predicted_return = ((predicted_return_df.iloc[-1] / predicted_return_df.iloc[0]) - 1) * 100
        print(predicted_return)
        if predicted_return > 0:
            print(f"The prediction expects a profitable return on: {round(predicted_return, 2)}%, over the next {prediction_days} days.")
        elif predicted_return < 0:
            print(f"The prediction expects a loss of: {round(predicted_return, 2)}%, over the next {prediction_days} days.")
        else:
            print(f"The prediction expects no return over the next {prediction_days} days.")
  

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
    forecast_data_df["Date"] = forecast_data_df["Date"].astype('datetime64[ns]')
    forecast_data_df = forecast_data_df.loc[forecast_data_df["Date"] >= stock_data_df.iloc[-1]["Date"]]
    stock_data_df = stock_data_df.set_index("Date")
    forecast_data_df = forecast_data_df.set_index("Date")
    stock_data_df["Price"].plot()
    forecast_data_df["Price"].plot()
    legend_list = ["Stock Price", "Predicted Stock Price"]
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
    # Import stock symbols from a CSV file
    stock_symbols_df = stock_data_fetch.import_symbols("index_symbol_list_single_stock.csv")
    stock_symbols_list = stock_symbols_df["Symbol"].tolist()
    stock_symbol = stock_symbols_list[0]
    print(stock_symbol)
    # Fetch stock data for the imported stock symbols
    stock_price_data_df = stock_data_fetch.fetch_stock_price_data(stock_symbol)
    stock_price_data_df = stock_data_fetch.calculate_period_returns(stock_price_data_df)
    stock_price_data_df = stock_data_fetch.calculate_moving_averages(stock_price_data_df)
    stock_price_data_df = stock_data_fetch.calculate_standard_diviation_value(stock_price_data_df)
    stock_price_data_df = stock_data_fetch.calculate_bollinger_bands(stock_price_data_df)
    # Fetch stock data for the imported stock symbols
    full_stock_financial_data_df = stock_data_fetch.fetch_stock_financial_data(stock_symbol)
    # Combine stock data with stock financial data
    combined_stock_data_df = stock_data_fetch.combine_stock_data(stock_price_data_df, full_stock_financial_data_df)
    # Calculate ratios
    combined_stock_data_df = stock_data_fetch.calculate_ratios(combined_stock_data_df)
    combined_stock_data_df = stock_data_fetch.calculate_momentum(combined_stock_data_df)
    combined_stock_data_df = stock_data_fetch.drop_nan_values(combined_stock_data_df)
    # Create a dictionary of dataframes to export to Excel
    dataframes = {
        "Combined Stock Data": combined_stock_data_df
    }
    # Export the dataframes to an Excel file
    stock_data_fetch.export_to_excel(dataframes, 'stock_data_single.xlsx')
    # Import the stock data from an Excel file
    dataframes = stock_data_fetch.import_excel("stock_data_single.xlsx")
    for key, value in dataframes.items():
        dataframe = value


    # Export the stock data to a CSV file
    stock_data_fetch.convert_excel_to_csv(dataframe, "stock_data_single")
    stock_data_df = import_stock_data.import_as_df_from_csv('stock_data_single.csv')
    # Split the dataset into traning, test data and prediction data
    test_size = 0.20
    scaler, x_training_data, x_test_data, y_training_data_df, y_test_data_df, prediction_data = split_dataset.dataset_train_test_split(stock_data_df, test_size, 1)
    # Feature selection
    feature_amount = 30
    x_training_dataset, x_test_dataset, x_prediction_dataset, selected_features_model, selected_features_list = dimension_reduction.feature_selection(feature_amount, x_training_data, x_test_data, y_training_data_df, y_test_data_df, prediction_data, stock_data_df)
    # Combine the reduced dataset with the stock price
    x_training_dataset_df = pd.DataFrame(x_training_dataset, columns=selected_features_list)
    y_training_data_df = y_training_data_df.reset_index(drop=True)
    traning_dataset_df = x_training_dataset_df.join(y_training_data_df)
    x_test_dataset_df = pd.DataFrame(x_test_dataset, columns=selected_features_list)
    y_test_data_df = y_test_data_df.reset_index(drop=True)
    test_dataset_df = x_test_dataset_df.join(y_test_data_df)
    x_prediction_dataset_df = pd.DataFrame(x_prediction_dataset, columns=selected_features_list)
    # Predict the stock price
    iterations = 7500
    nn_model = neural_network_model(traning_dataset_df, test_dataset_df, feature_amount*16, feature_amount*12, feature_amount*20, feature_amount*16, iterations, 1)
    amount_of_days = 25
    forecast_df = predict_future_price_changes(stock_symbol, scaler, nn_model, selected_features_list, stock_data_df, amount_of_days)
    # Create a dictionary of dataframes to export to Excel
    dataframes = {
        "Forecast Data": forecast_df
    }
    # Export the dataframes to an Excel file
    stock_data_fetch.export_to_excel(dataframes, "stock_data_mod_single.xlsx")
    calculate_predicted_profit(forecast_df, amount_of_days)
    # Plot the graph
    plot_graph(stock_data_df, forecast_df)
    # Run a Monte Carlo simulation
    year_amount = 20
    sim_amount = 1000
    monte_carlo_df = monte_carlo_sim.monte_carlo_analysis(0, stock_data_df, forecast_df, year_amount, sim_amount)
    # Calculate the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds to build dataset and ML models.")