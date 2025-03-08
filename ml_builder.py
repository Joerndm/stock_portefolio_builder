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
    x_training_df = traning_dataset.drop(["price"], axis=1)
    y_training_df = traning_dataset["price"]
    # Convert the DataFrame to a numpy array
    x_training = np.array(x_training_df)
    y_training = np.array(y_training_df)
    x_test_df = test_dataset.drop(["price"], axis=1)
    y_test_df = test_dataset["price"]
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
        date = stock_df["date"].values[-x]
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
        date_list.append(date)


    forecast_dict = {"date":date_list, "prediction_dt":forecast_set_dt
    }
    forecast_df = pd.DataFrame(forecast_dict)
    forecast_df = forecast_df.set_index("date")
    # print(forecast_df)
    predicted_return = ((forecast_df.iloc[-1]["prediction_dt"] / forecast_df.iloc[0]["prediction_dt"]) - 1) * 100
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
        x_training_df = traning_dataset.drop(["prediction"], axis=1)
        y_training_df = traning_dataset["prediction"]
        # Convert the DataFrame to a numpy array
        x_training = np.array(x_training_df)
        y_training = np.array(y_training_df)
        x_test_df = test_dataset.drop(["prediction"], axis=1)
        y_test_df = test_dataset["prediction"]
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

    except ValueError as e:
        raise ValueError("The model could not be generated. Please check the input data.") from e

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
        # Predict the future stock open_Price
        short_term_dynamic_list = [
            '1M', '3M', '6M', '9M', '1Y', '2Y', '3Y', '4Y',
            '5Y', 'sma_40', 'sma_120', 'ema_40', 'ema_120',
            "std_Div_40", "std_Div_120", "bollinger_Band_40_2STD",
            "bollinger_Band_120_2STD", 'p_s', 'p_e', 'p_b',
            'p_fcf', "momentum"
        ]
        # print(f"selected_features_list: {selected_features_list}")
        features_list = selected_features_list.copy()
        # Append "Date" to the selected_features_list in position 0
        features_list.insert(0, "date")
        features_list.insert(1, "open_Price")
        features_list.insert(2, "1D")
        stock_mod_df = stock_df.copy()
        for run in range(prediction_days):
            future_df = stock_mod_df.iloc[-1].copy().to_frame().transpose()
            for feature in range(len(short_term_dynamic_list)):
                if short_term_dynamic_list[feature] in features_list:
                    future_day = pd.to_datetime(stock_mod_df.iloc[-1]["date"]) + relativedelta(days=1)
                    if future_day.weekday() == 5:
                        future_day = future_day + datetime.timedelta(days=2)
                        future_day = future_day.strftime("%Y-%m-%d")
                        future_df["date"] = str(future_day)
                    elif future_day.weekday() == 6:
                        future_day = future_day + datetime.timedelta(days=1)
                        future_day = future_day.strftime("%Y-%m-%d")
                        future_df["date"] = str(future_day)
                    else:
                        future_day = future_day.strftime("%Y-%m-%d")
                        future_df["date"] = str(future_day)

                    # Calculate the return for the last 1 month
                    if short_term_dynamic_list[feature] == "1M":
                        if len(stock_mod_df) <= 21:
                            hist_df = pd.DataFrame(yf.download(ticker, period="6y"))["Open"].reset_index()
                            hist_df = hist_df.loc[hist_df["Date"] < stock_mod_df.iloc[0]["date"]]
                            hist_df = hist_df.rename(columns={ticker: "open_Price", "Date": "date"})
                            hist_df = pd.concat([hist_df, stock_mod_df[["date", "open_Price"]]], axis=0)
                            return_1M = hist_df.iloc[-22:]["Price"].pct_change(21).iloc[-1]
                            future_df["1M"] = return_1M
                        elif len(stock_mod_df) > 21:
                            return_1M = stock_mod_df.iloc[-22:]["open_Price"].pct_change(21).iloc[-1]
                            future_df["1M"] = return_1M

                    # Calculate the return for the last 3 months
                    if short_term_dynamic_list[feature] == "3M":
                        if len(stock_mod_df) <= 63:
                            hist_df = pd.DataFrame(yf.download(ticker, period="6y"))["Open"].reset_index()
                            hist_df = hist_df.loc[hist_df["Date"] < stock_mod_df.iloc[0]["date"]]
                            hist_df = hist_df.rename(columns={ticker: "open_Price", "Date": "date"})
                            hist_df = pd.concat([hist_df, stock_mod_df[["date", "open_Price"]]], axis=0)
                            return_3M = hist_df.iloc[-64:]["open_Price"].pct_change(63).iloc[-1]
                            future_df["3M"] = return_3M
                        elif len(stock_mod_df) > 63:
                            return_3M = stock_mod_df.iloc[-64:]["open_Price"].pct_change(63).iloc[-1]
                            future_df["3M"] = return_3M

                    # Calculate the return for the last 6 months
                    if short_term_dynamic_list[feature] == "6M":
                        if len(stock_mod_df) <= 126:
                            hist_df = pd.DataFrame(yf.download(ticker, period="6y"))["Open"].reset_index()
                            hist_df = hist_df.loc[hist_df["Date"] < stock_mod_df.iloc[0]["date"]]
                            hist_df = hist_df.rename(columns={ticker: "open_Price", "Date": "date"})
                            hist_df = pd.concat([hist_df, stock_mod_df[["date", "open_Price"]]], axis=0)
                            return_6M = hist_df.iloc[-127:]["open_Price"].pct_change(126).iloc[-1]
                            future_df["6M"] = return_6M
                        elif len(stock_mod_df) > 126:
                            return_6M = stock_mod_df.iloc[-127:]["open_Price"].pct_change(126).iloc[-1]
                            future_df["6M"] = return_6M

                    # Calculate the return for the last 9 months
                    if short_term_dynamic_list[feature] == "9M":
                        if len(stock_mod_df) <= 189:
                            hist_df = pd.DataFrame(yf.download(ticker, period="6y"))["Open"].reset_index()
                            hist_df = hist_df.loc[hist_df["Date"] < stock_mod_df.iloc[0]["date"]]
                            hist_df = hist_df.rename(columns={ticker: "open_Price", "Date": "date"})
                            hist_df = pd.concat([hist_df, stock_mod_df[["date", "open_Price"]]], axis=0)
                            return_9M = hist_df.iloc[-190:]["open_Price"].pct_change(189).iloc[-1]
                            future_df["9M"] = return_9M
                        elif len(stock_mod_df) > 189:
                            return_9M = stock_mod_df.iloc[-190:]["open_Price"].pct_change(189).iloc[-1]
                            future_df["9M"] = return_9M

                    # Calculate the return for the last 1 year
                    if short_term_dynamic_list[feature] == "1Y":
                        if len(stock_mod_df) <= 252:
                            hist_df = pd.DataFrame(yf.download(ticker, period="6y"))["Open"].reset_index()
                            hist_df = hist_df.loc[hist_df["Date"] < stock_mod_df.iloc[0]["date"]]
                            hist_df = hist_df.rename(columns={ticker: "open_Price", "Date": "date"})
                            hist_df = pd.concat([hist_df, stock_mod_df[["date", "open_Price"]]], axis=0)
                            return_1Y = hist_df.iloc[-253:]["open_Price"].pct_change(252).iloc[-1]
                            future_df["1Y"] = return_1Y
                        elif len(stock_mod_df) > 252:
                            return_1Y = stock_mod_df.iloc[-253:]["open_Price"].pct_change(252).iloc[-1]
                            future_df["1Y"] = return_1Y

                    # Calculate the return for the last 2 years
                    if short_term_dynamic_list[feature] == "2Y":
                        if len(stock_mod_df) <= 504:
                            hist_df = pd.DataFrame(yf.download(ticker, period="6y"))["Open"].reset_index()
                            hist_df = hist_df.loc[hist_df["Date"] < stock_mod_df.iloc[0]["date"]]
                            hist_df = hist_df.rename(columns={ticker: "open_Price", "Date": "date"})
                            hist_df = pd.concat([hist_df, stock_mod_df[["date", "open_Price"]]], axis=0)
                            return_2Y = hist_df.iloc[-505:]["open_Price"].pct_change(504).iloc[-1]
                            future_df["2Y"] = return_2Y
                        elif len(stock_mod_df) > 504:
                            return_2Y = stock_mod_df.iloc[-505:]["open_Price"].pct_change(504).iloc[-1]
                            future_df["2Y"] = return_2Y

                    # Calculate the return for the last 3 years
                    if short_term_dynamic_list[feature] == "3Y":
                        if len(stock_mod_df) <= 756:
                            hist_df = pd.DataFrame(yf.download(ticker, period="6y"))["Open"].reset_index()
                            # print(hist_df)
                            # print(hist_df.info())
                            # print(stock_mod_df)
                            # print(stock_mod_df.info())
                            hist_df = hist_df.loc[hist_df["Date"] < stock_mod_df.iloc[0]["date"]]
                            hist_df = hist_df.rename(columns={ticker: "open_Price", "Date": "date"})
                            hist_df = pd.concat([hist_df, stock_mod_df[["date", "open_Price"]]], axis=0)
                            return_3Y = hist_df.iloc[-757:]["open_Price"].pct_change(756).iloc[-1]
                            future_df["3Y"] = return_3Y
                        elif len(stock_mod_df) > 756:
                            return_3Y = stock_mod_df.iloc[-757:]["open_Price"].pct_change(756).iloc[-1]
                            future_df["3Y"] = return_3Y

                    # Calculate the return for the last 4 years
                    if short_term_dynamic_list[feature] == "4Y":
                        if len(stock_mod_df) <= 1008:
                            hist_df = pd.DataFrame(yf.download(ticker, period="6y"))["Open"].reset_index()
                            hist_df = hist_df.loc[hist_df["Date"] < stock_mod_df.iloc[0]["date"]]
                            hist_df = hist_df.rename(columns={ticker: "open_Price", "Date": "date"})
                            hist_df = pd.concat([hist_df, stock_mod_df[["date", "open_Price"]]], axis=0)
                            return_4Y = hist_df.iloc[-1009:]["open_Price"].pct_change(1008).iloc[-1]
                            future_df["4Y"] = return_4Y
                        elif len(stock_mod_df) > 1008:
                            return_4Y = stock_mod_df.iloc[-1009:]["open_Price"].pct_change(1008).iloc[-1]
                            future_df["4Y"] = return_4Y

                    # Calculate the return for the last 5 years
                    if short_term_dynamic_list[feature] == "5Y":
                        if len(stock_mod_df) <= 1260:
                            # print(stock_mod_df)
                            hist_df = pd.DataFrame(yf.download(ticker, period="6y"))["Open"].reset_index()
                            hist_df = hist_df.loc[hist_df["Date"] < stock_mod_df.iloc[0]["date"]]
                            hist_df = hist_df.rename(columns={ticker: "open_Price", "Date": "date"})
                            hist_df = pd.concat([hist_df, stock_mod_df[["date", "open_Price"]]], axis=0)
                            # print("hist_df")
                            # print(hist_df)
                            return_5Y = hist_df.iloc[-1261:]["open_Price"].pct_change(1260).iloc[-1]
                            # print(return_5Y)
                            future_df["5Y"] = return_5Y
                        elif len(stock_mod_df) > 1260:
                            # print(stock_mod_df)
                            return_5Y = stock_mod_df.iloc[-1261:]["open_Price"].pct_change(1260).iloc[-1]
                            # print(return_5Y)
                            future_df["5Y"] = return_5Y

                    # Calculate the 40 days simple moving average
                    if short_term_dynamic_list[feature] == "sma_40":
                        sma_40 = stock_mod_df.iloc[-40:]["open_Price"].mean()
                        future_df["sma_40"] = sma_40

                    # Calculate the 120 days simple moving average
                    if short_term_dynamic_list[feature] == "sma_120":
                        sma_120 = stock_mod_df.iloc[-120:]["open_Price"].mean()
                        future_df["sma_120"] = sma_120

                    # Calculate the 40 days exponential moving average
                    if short_term_dynamic_list[feature] == "ema_40":
                        ema_40 = stock_mod_df.iloc[-40:]["open_Price"].ewm(span=40).mean().iloc[-1]
                        future_df["ema_40"] = ema_40

                    # Calculate the 120 days exponential moving average
                    if short_term_dynamic_list[feature] == "ema_120":
                        ema_120 = stock_mod_df.iloc[-120:]["open_Price"].ewm(span=120).mean().iloc[-1]
                        future_df["ema_120"] = ema_120

                    # Calculate the 40 days standard deviation
                    if short_term_dynamic_list[feature] == "std_Div_40":
                        std_div_40 = stock_mod_df.iloc[-40:]["open_Price"].std()
                        future_df["std_Div_40"] = std_div_40

                    # Calculate the 120 days standard deviation
                    if short_term_dynamic_list[feature] == "std_Div_120":
                        std_div_120 = stock_mod_df.iloc[-120:]["open_Price"].std()
                        future_df["std_Div_120"] = std_div_120

                    # Calculate the Bollinger Band for the last 40 days
                    if short_term_dynamic_list[feature] == "bollinger_Band_40_2STD":
                        std_div_40 = stock_mod_df.iloc[-40:]["open_Price"].std()
                        sma_40 = stock_mod_df.iloc[-40:]["open_Price"].mean()
                        bollinger_Band_40_Upper = (sma_40 + (std_div_40 * 2))
                        bollinger_Band_40_Lower = (sma_40 - (std_div_40 * 2))
                        bollinger_band_40 = bollinger_Band_40_Upper - bollinger_Band_40_Lower
                        future_df["bollinger_Band_40_2STD"] = bollinger_band_40

                    # Calculate the Bollinger Band for the last 120 days
                    if short_term_dynamic_list[feature] == "bollinger_Band_120_2STD":
                        std_div_120 = stock_mod_df.iloc[-120:]["open_Price"].std()
                        sma_120 = stock_mod_df.iloc[-120:]["open_Price"].mean()
                        bollinger_Band_120_Upper = (sma_120 + (std_div_120 * 2))
                        bollinger_Band_120_Lower = (sma_120 - (std_div_120 * 2))
                        bollinger_band_120 = bollinger_Band_120_Upper - bollinger_Band_120_Lower
                        future_df["bollinger_Band_120_2STD"] = bollinger_band_120
                        
                    # Calculate the p_s ratio
                    if short_term_dynamic_list[feature] == "p_s":
                        p_s = stock_mod_df.iloc[-1]["open_Price"] / (stock_mod_df.iloc[-1]["revenue"] / stock_mod_df.iloc[-1]["average_shares"])
                        future_df["p_s"] = p_s

                    # Calculate the p_e ratio
                    if short_term_dynamic_list[feature] == "p_e":
                        p_e = stock_mod_df.iloc[-1]["open_Price"] / stock_mod_df.iloc[-1]["eps"]
                        future_df["p_e"] = p_e

                    # Calculate the p_b ratio
                    if short_term_dynamic_list[feature] == "p_b":
                        p_b = stock_mod_df.iloc[-1]["open_Price"] / stock_mod_df.iloc[-1]["book_Value_Per_Share"]
                        future_df["p_b"] = p_b

                    # Calculate the p_fcf ratio
                    if short_term_dynamic_list[feature] == "p_fcf":
                        p_fcf = stock_mod_df.iloc[-1]["open_Price"] / stock_mod_df.iloc[-1]["free_Cash_Flow_Per_Share_Growth"]
                        future_df["p_fcf"] = p_fcf

                    # Calculate the momentum
                    if short_term_dynamic_list[feature] == "momentum":
                        if stock_mod_df.iloc[-1]["open_Price"] >= stock_mod_df.iloc[-2]["open_Price"]:
                            if stock_mod_df.iloc[-1]["momentum"] <= 0:
                                momentum = 1
                                # Update the momentum column with the calculated value
                                future_df["momentum"] = momentum
                            elif stock_mod_df.iloc[-1]["momentum"] > 0:
                                momentum = stock_mod_df.iloc[-1]["momentum"] + 1
                                # Update the momentum column with the calculated value
                                future_df["momentum"] = momentum
                        elif stock_mod_df.iloc[-1]["open_Price"] < stock_mod_df.iloc[-2]["open_Price"]:
                            if stock_mod_df.iloc[-1]["momentum"] >= 0:
                                momentum = -1
                                # Update the momentum column with the calculated value
                                future_df["momentum"] = momentum
                            elif stock_mod_df.iloc[-1]["momentum"] < 0:
                                momentum = stock_mod_df.iloc[-1]["momentum"] - 1
                                # Update the momentum column with the calculated value
                                future_df["momentum"] = momentum

            # Change the data type of the date column to datetime
            future_df["date"] = pd.to_datetime(future_df["date"]) 
            # print("future_df")
            # print(future_df)
            # Concat the pandas series as new line to the stock_mod_df
            prediction_df = pd.concat([stock_mod_df.iloc[-1].to_frame().transpose(), future_df], axis=0).reset_index(drop=True)
            # print("prediction_df")
            # print(prediction_df)
            # print("prediction_df info")
            # print(prediction_df.info())
            # prediction_df = prediction_df.drop(["name", "open_Price", "trade_Volume"], axis=1)
            prediction_df = prediction_df.drop(["date", "ticker", "currency", "open_Price", "1D"], axis=1)
            # print("prediction_df")
            # print(prediction_df)
            # print("prediction_df info")
            # print(prediction_df.info())
            # Concat the future_df to the stock_mod_df
            stock_mod_df = pd.concat([stock_mod_df, future_df], axis=0).reset_index(drop=True)
            # print("stock_mod_df")
            # print(stock_mod_df)
            # print("stock_mod_df info")
            # print(stock_mod_df.info())
            # Scale the prediction_df
            scaled_prediction_df = scaler.transform(prediction_df)
            scaled_prediction_df = np.array(scaled_prediction_df[selected_features_list])
            # Predict the future stock price
            forecast = model.predict(scaled_prediction_df)
            forecast_price_change = forecast[1]
            # Update the 1D column with the calculated value
            stock_mod_df.loc[len(stock_mod_df)-1, "1D"] = forecast_price_change
            # Update the Price column with the calculated value
            stock_mod_df.loc[len(stock_mod_df)-1, "open_Price"] = stock_mod_df.loc[len(stock_mod_df)-2, "open_Price"] * (1 + forecast_price_change)
            # print("stock_mod_df")
            # print(stock_mod_df)
            # print("stock_mod_df info")
            # print(stock_mod_df.info())

        stock_mod_df = stock_mod_df[features_list]
        return stock_mod_df

    except ValueError as e:
        print("The prediction could not be completed. Please check the input data.")
        print("future_df")
        print(future_df)
        print("prediction_df")
        print(prediction_df)
        raise ValueError("The prediction could not be completed. Please check the input data.") from e

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
        # print("forecast_df columns")
        # print(forecast_df.columns)
        # print("prediction_days")
        # print(prediction_days)
        predicted_return_df = forecast_df.loc[len(forecast_df)-prediction_days:, "open_Price"]
        predicted_return = ((predicted_return_df.iloc[-1] / predicted_return_df.iloc[0]) - 1) * 100
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
    stock_data_df["date"] = stock_data_df["date"].astype('datetime64[ns]')
    forecast_data_df["date"] = forecast_data_df["date"].astype('datetime64[ns]')
    forecast_data_df = forecast_data_df.loc[forecast_data_df["date"] >= stock_data_df.iloc[-1]["date"]]
    stock_data_df = stock_data_df.set_index("date")
    forecast_data_df = forecast_data_df.set_index("date")
    stock_data_df["open_Price"].plot()
    forecast_data_df["open_Price"].plot()
    legend_list = ["Stock Price", "Predicted Stock Price"]
    plt.legend(legend_list,
        loc="best"
    )
    plt.xlabel("Date")
    plt.ylabel("Opening price")
    plt.title(f"Stock Price Prediction of {stock_data_df.iloc[0]["ticker"]}")
    # Change " " in stock_data_df.iloc[0]["Name"] to "_" to avoid error when saving the graph
    stock_data_df = stock_data_df.replace({"ticker": [" ", "/"]}, {"ticker": "_"}, regex=True)
    stock_name = stock_data_df.iloc[0]["ticker"]
    graph_name = str(f"stock_prediction_of_{stock_name}.png")
    my_path = os.path.abspath(__file__)
    path = os.path.dirname(my_path)
    # Save the graph
    try:
        plt.savefig(os.path.join(path, "generated_graphs", graph_name), bbox_inches="tight", pad_inches=0.5, transparent=False, format="png")
        plt.clf()
        plt.close("all")


    except FileNotFoundError as e:
        raise FileNotFoundError("The graph could not be saved. Please check the file name or path.") from e

    # Show the graph
    # plt.show()

# Run the main function
if __name__ == "__main__":
    import fetch_secrets
    import db_connectors
    import db_interactions
    start_time = time.time()
    # Import stock symbols from DB
    stock_symbols_list = db_interactions.import_ticker_list()
    print(stock_symbols_list)
    stock_symbol = stock_symbols_list[0]
    # stock_symbol = "ORSTED.CO"
    print(stock_symbol)
    stock_data_df = db_interactions.import_stock_dataset(stock_symbol)
    # Change the date column to datetime 64
    stock_data_df["date"] = pd.to_datetime(stock_data_df["date"])
    print("stock_data_df info")
    print(stock_data_df.info())
    print("stock_data_df")
    print(stock_data_df)
    # Drop the columns that are empty
    stock_data_df = stock_data_df.dropna(axis=0, how="any")
    stock_data_df = stock_data_df.dropna(axis=1, how="any")
    print("stock_data_df info")
    print(stock_data_df.info())
    print("stock_data_df")
    print(stock_data_df)
    # Split the dataset into traning, test data and prediction data
    test_size = 0.20
    scaler, x_training_data, x_test_data, y_training_data_df, y_test_data_df, prediction_data = split_dataset.dataset_train_test_split(stock_data_df, test_size, 1)
    # Feature selection
    feature_amount = 20
    x_training_dataset, x_test_dataset, x_prediction_dataset, selected_features_model, selected_features_list = dimension_reduction.feature_selection(feature_amount, x_training_data, x_test_data, y_training_data_df, y_test_data_df, prediction_data, stock_data_df)
    # print("selected_features_list")
    # print(selected_features_list)
    # Combine the reduced dataset with the stock price
    x_training_dataset_df = pd.DataFrame(x_training_dataset, columns=selected_features_list)
    # print("x_training_dataset_df")
    # print(x_training_dataset_df)
    y_training_data_df = y_training_data_df.reset_index(drop=True)
    # print("y_training_data_df")
    # print(y_training_data_df)
    traning_dataset_df = x_training_dataset_df.join(y_training_data_df)
    # print("traning_dataset_df")
    # print(traning_dataset_df)
    x_test_dataset_df = pd.DataFrame(x_test_dataset, columns=selected_features_list)
    # print("x_test_dataset_df")
    # print(x_test_dataset_df)
    y_test_data_df = y_test_data_df.reset_index(drop=True)
    # print("y_test_data_df")
    # print(y_test_data_df)
    test_dataset_df = x_test_dataset_df.join(y_test_data_df)
    # print("test_dataset_df")
    # print(test_dataset_df)
    x_prediction_dataset_df = pd.DataFrame(x_prediction_dataset, columns=selected_features_list)
    # Predict the stock price
    iterations = 7500
    nn_model = neural_network_model(traning_dataset_df, test_dataset_df, feature_amount*16, feature_amount*12, feature_amount*20, feature_amount*16, iterations, 1)
    amount_of_days = 10
    forecast_df = predict_future_price_changes(stock_symbol, scaler, nn_model, selected_features_list, stock_data_df, amount_of_days)
    # print("forecast_df")
    # print(forecast_df)
    # Calculate the predicted profit
    calculate_predicted_profit(forecast_df, amount_of_days)
    # Plot the graph
    plot_graph(stock_data_df, forecast_df)
    # Run a Monte Carlo simulation
    year_amount = 10
    sim_amount = 1000
    monte_carlo_day_df, monte_carlo_year_df = monte_carlo_sim.monte_carlo_analysis(0, stock_data_df, forecast_df, year_amount, sim_amount)
    forecast_df = forecast_df.rename(columns={"open_Price": stock_symbol + "_price"})
    print("forecast_df")
    print(forecast_df)
    # Calculate the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds to build dataset and ML models.")
