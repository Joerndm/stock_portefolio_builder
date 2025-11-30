import os, shutil, time, io, sys
import pandas as pd
import time
import datetime
import math
import tempfile

from dateutil.relativedelta import relativedelta
import yfinance as yf
import numpy as np
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np. int = int
import matplotlib
import matplotlib.pyplot as plt
import shutil
from sklearn import tree
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn. linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold
import tensorflow as tf
from tensorflow. keras. models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
import keras
import keras_tuner as kt
from keras_tuner. tuners import Sklearn

matplotlib.use('Agg')
pd.set_option('future.no_silent_downcasting', True)

# import stock_data_fetch
import split_dataset
import dimension_reduction
import monte_carlo_sim

# Force UTF-8 logging globally
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="ignore")

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

def build_random_forest_model(hp):
    """Builds a RandomForestRegressor with expanded tunable hyperparameters."""
    
    max_features_choice = hp.Choice('max_features', ['sqrt', 'log2', '0.3', '0.5', '0.8'])
    
    if max_features_choice in ['0.3', '0.5', '0.8']:
        max_features_value = float(max_features_choice)
    else:
        max_features_value = max_features_choice
    
    # Get bootstrap choice
    bootstrap = hp.Boolean('bootstrap')
    
    # max_samples can only be used with bootstrap=True
    if bootstrap:
        max_samples_value = hp.Float('max_samples', 0.5, 1.0, step=0.1)
    else:
        max_samples_value = None

    model = RandomForestRegressor(
        n_estimators=hp. Int('n_estimators', 100, 1500, step=100),
        max_depth=hp.Int('max_depth', 3, 50, step=2),
        min_samples_split=hp.Choice('min_samples_split', [2, 5, 10, 15]),
        min_samples_leaf=hp.Choice('min_samples_leaf', [1, 2, 4, 8, 16]),
        criterion=hp.Choice('criterion', ['squared_error', 'absolute_error', 'friedman_mse']),
        bootstrap=bootstrap,
        max_features=max_features_value,
        max_samples=max_samples_value,  # Now conditionally set
        random_state=42,
        n_jobs=-1
    )
    return model

def tune_random_forest_model(stock_symbol, x_training_dataset_df, y_training_dataset_df, max_trials=20):
    """
    Improved Random Forest tuning with better hyperparameters and optimization.  
    
    Parameters:
    - stock_symbol (str): The stock ticker symbol
    - x_training_dataset_df (pd.DataFrame): Training dataset (ALREADY SCALED)
    - y_training_dataset_df (pd.Series or np.ndarray): Training labels (UNSCALED for RF)
    - max_trials (int): Maximum number of tuning trials
    
    Returns:
    - best_rf_model: The tuned Random Forest model
    """
    
    # Convert to numpy arrays to avoid feature name warnings
    x_train = x_training_dataset_df.values
    y_train = y_training_dataset_df.values

    # Define the MSE scorer
    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

    # Setup directory structure
    overwrite_val = False
    directory_val = "tuning_dir"
    project_name_val = f"RF_tuning_{stock_symbol}"
    project_path = os.path.join(directory_val, project_name_val)

    # Cleanup old tuning directory if needed
    if os.path.exists(project_path):
        if overwrite_val:
            try:
                shutil.rmtree(project_path)
                print(f"🗑️  Deleted old tuning directory: {project_path}")
            except Exception as e:
                print(f"⚠️  Warning: Could not delete old tuning directory {project_path}. Error: {e}")

    # Create tuner with Hyperband (faster than Bayesian Optimization)
    tuner = Sklearn(
        oracle=kt.oracles. Hyperband(
            objective=kt. Objective('score', 'min'),
            max_epochs=max_trials,
            factor=3,
            seed=42
        ),
        hypermodel=build_random_forest_model,
        scoring=mse_scorer,
        cv=RepeatedKFold(n_splits=5, n_repeats=2, random_state=42),
        directory=directory_val,
        project_name=project_name_val,
        overwrite=overwrite_val
    )

    # Search for best hyperparameters
    print(f"🔍 Starting Random Forest hyperparameter tuning for {stock_symbol}...")
    tuner.search(x_train, y_train)

    # Get best hyperparameters
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Build and train best model
    best_rf_model = tuner.hypermodel. build(best_hp)
    best_rf_model.fit(x_train, y_train)

    # Print best hyperparameters
    print("\n🌳 Best Random Forest hyperparameters found:")
    for param, value in best_hp.values. items():
        print(f"  ✓ {param}: {value}")

    # Feature importance logging
    importances = best_rf_model. feature_importances_
    feature_names = x_training_dataset_df.columns

    print("\n📊 Top 10 Feature Importances:")
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    for idx, row in feature_importance_df.head(10).iterrows():
        print(f"  • {row['feature']}: {row['importance']:.4f}")

    return best_rf_model

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

def build_svm_model(hp):
    kernel_choice = hp.Choice('kernel', ['linear', 'rbf', 'poly'])
    model = SVR(
        kernel=kernel_choice,
        C=hp.Float('C', 1e-3, 1e3, sampling='log'),
        gamma=hp.Float('gamma', 1e-4, 1e1, sampling='log'),
        degree=hp.Int('degree', 2, 5) if kernel_choice == 'poly' else 3
    )
    return model

def tune_svm_model(stock_symbol, training_dataset_df, max_trials=20):
    x_train = training_dataset_df.drop(["prediction"], axis=1).values
    y_train = training_dataset_df["prediction"].values

    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

    tuner = Sklearn(
        oracle=kt.oracles.BayesianOptimization(
            objective=kt.Objective('score', 'min'),
            max_trials=max_trials,
            seed=42
        ),
        hypermodel=build_svm_model,
        scoring=mse_scorer,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        directory="tuning_dir",
        project_name=f"SVM_tuning_{stock_symbol}",
        overwrite=False
    )

    tuner.search(x_train, y_train)

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_svm_model = tuner.hypermodel.build(best_hp)
    best_svm_model.fit(x_train, y_train)

    print("⚖️ Best SVM hyperparameters found:")
    for param, value in best_hp.values.items():
        print(f"- {param}: {value}")

    return best_svm_model

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

# Fits an LSTM model to the traning dataset and predicts the stock price
def lstm_model(traning_dataset, test_dataset, epochs=500, batch_size=128):
    """
    Builds and trains an LSTM model.

    Parameters:
    - traning_dataset (pandas.DataFrame): A DataFrame containing the training dataset.
    - test_dataset (pandas.DataFrame): A DataFrame containing the test dataset.
    - epochs (int): The number of epochs to train the model.
    - batch_size (int): The batch size for training.

    Returns:
    tensorflow.keras.Model: The trained LSTM model.
    """
    # Split the training dataset into features and labels
    x_train = traning_dataset.drop(["prediction"], axis=1).values
    y_train = traning_dataset["prediction"].values

    # Split the test dataset into features and labels
    x_test = test_dataset.drop(["prediction"], axis=1).values
    y_test = test_dataset["prediction"].values

    # Reshape the data to fit the LSTM input requirements
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    model = Sequential()
    model.add(Input(shape=(x_train.shape[1], x_train.shape[2])))
    model.add(LSTM(units=250, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=250, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')

    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))

    trainScore = model.evaluate(x_train, y_train, verbose=0)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
    testScore = model.evaluate(x_test, y_test, verbose=0)
    print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))

    return model

    # Example usage:
    # Assuming traning_dataset and test_dataset are already defined and preprocessed
    # traning_dataset and test_dataset should be pandas DataFrames with a "Price" column for labels

    # traning_dataset, test_dataset = ...  # Your data here

    # lstm_model = build_lstm_model(traning_dataset, test_dataset, epochs=50, batch_size=32)

    # Predicts the future stock price

def create_sequences(data, time_steps):
    """Converts a 2D feature array into a 3D sequence array for LSTM."""
    Xs = []
    for i in range(len(data) - time_steps + 1):
        Xs.append(data[i:(i + time_steps)])
    return np.array(Xs)

def build_lstm_model(hp, input_shape):
    model = Sequential()
    # First LSTM layer
    model.add(
        Bidirectional(
            LSTM(
                units=hp.Int(
                    "input_units",
                    min_value=32,
                    max_value=512,
                    step=16
                ),
                return_sequences=True,
                kernel_regularizer=regularizers.l2(
                    hp.Float(
                        "l2_reg_input",
                        1e-6, 1e-3,
                        sampling="log"
                    )
                ),
                input_shape=input_shape
            )
        )
    )

    max_amount_layers = 20
    for i in range(1, hp.Int('n_layers', 1, max_amount_layers)):
        model.add(
            Bidirectional(
                LSTM(
                    units=hp. Int(
                        f"units_{i}",
                        min_value=32,
                        max_value=512,
                        step=16
                    ),
                    return_sequences=True,
                    kernel_regularizer=regularizers.l2(
                        hp.Float(
                            f"l2_reg_{i}",
                            1e-6, 1e-3,
                            sampling="log"
                        )
                    )
                )
            )
        )

        model.add(
            Dropout(
                hp.Float(
                    f"dropout_{i}",
                    min_value=0.1,
                    max_value=0.5,
                    step=0.1
                )
            )
        )

    # Final LSTM (collapse to 2D)
    model.add(
        Bidirectional(
            LSTM(
                units=hp.Int(
                    f"final_units",
                    min_value=32,
                    max_value=512,
                    step=16
                ),
                return_sequences=False,
                kernel_regularizer=regularizers. l2(
                    hp. Float(
                        f"l2_reg_final",
                        1e-6, 1e-3,
                        sampling="log"
                    )
               )
            )
        )
    )

    model.add(
        Dense(
            units=hp.Int(
                "dense_1",
                min_value=16,
                max_value=128,
                step=16
            ),
            activation=hp.Choice(
                "dense_1_activation",
                ["relu", "tanh"]
            )
        )
    )

    model.add(
        Dropout(
            hp.Float(
                "dropout__dense_1",
                min_value=0.1,
                max_value=0.8,
                step=0.05
            )
        )
    )

    model.add(
        Dense(
            units=hp.Int(
                "dense_2",
                min_value=4,
                max_value=96,
                step=4),
            activation=hp.Choice(
                "dense_2_activation",
                ["relu", "sigmoid"]
            )
        )
    )

    model.add(
        Dropout(
            hp.Float(
                "dropout_dense_2",
                min_value=0.1,
                max_value=0.8,
                step=0.05
            )
        )
    )

    # Output layers
    model.add(
        Dense(
            1,
            activation="linear"
        )
    )

    # Optimizer choice
    optimizer_choice = hp.Choice("optimizer", ["adam", "rmsprop", "nadam"])
    learning_rate = hp.Float("learning_rate", 1e-5, 1e-2, sampling="log")

    if optimizer_choice == "adam":
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_choice == "rmsprop":
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        optimizer = Nadam(learning_rate=learning_rate)

    # Loss choice
    loss_choice = hp.Choice(
        "loss",
        ["mean_squared_error", "huber", "mean_absolute_percentage_error"]
    )

    # Optimizer with tunable learning rate
    model.compile(
        optimizer=optimizer,
        loss=loss_choice,
        metrics=["mean_absolute_error", "mean_squared_error", "accuracy"]
    )
    return model

def tune_lstm_model(stock, x_training_dataset, y_training_dataset, x_test_dataset, y_test_dataset, max_trials=10, executions_per_trial=1, epochs=100, retries=3, delay=5, onedrive_base=r"C:\Users\joern\OneDrive\Dokumenter\Privat\Aktie_database\code\stock_portefolio_builder"):
    x_train = x_training_dataset.values
    y_train = y_training_dataset.values
    x_test = x_test_dataset.values
    y_test = y_test_dataset.values

    # Reshape for LSTM (unchanged)
    x_train_lstm = create_sequences(x_train, TIME_STEPS)
    y_train_lstm = y_train[TIME_STEPS-1:]
    x_test_lstm = create_sequences(x_test, TIME_STEPS)
    y_test_lstm = y_test[TIME_STEPS-1:]

    # Basic sanity checks (same approach as tune_lstm_model)
    if x_test_lstm.ndim == 1:
        num_features = x_train_lstm.shape[2]
        x_test_lstm = x_test_lstm.reshape(-1, TIME_STEPS, num_features)

    # If x_test_lstm.shape is (N, 1) or similar, the data is too short.
    if len(x_test_lstm.shape) < 3:
        raise ValueError(
            "Test data sequence creation failed. Check that the test dataset size is larger than TIME_STEPS."
        )

    # Reshape y data to be 2D arrays
    y_train_lstm = y_train_lstm.reshape(-1, 1)
    y_test_lstm = y_test_lstm.reshape(-1, 1)

    # --- Setup directories ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tuning_dir = os.path.join(script_dir, "tuning_dir")
    temp_dir = os.path.join(tempfile.gettempdir(), "temp_tuning_dir")
    project_name_val = f"LSTM_tuning_{stock}"
    finished_project_path = os.path.join(tuning_dir, project_name_val)
    temp_project_path = os.path.join(temp_dir, project_name_val)

    print(f"Script directory: {script_dir}")
    print(f"Finished tuning directory: {finished_project_path}")
    print(f"Temporary tuning directory: {temp_project_path}")

    # --- PRIORITY 1: Check for finished tuning ---
    best_model = load_best_model_from_finished_tuning(
        finished_project_path, 
        x_train_lstm. shape[1], 
        x_train_lstm.shape[2]
    )
    if best_model is not None:
        print(f"✅ Loaded best model from finished tuning: {finished_project_path}")
        return best_model

    # --- PRIORITY 2: Check for partial tuning and continue ---
    overwrite_val = False
    if os.path.exists(temp_project_path):
        print(f"⏸️  Found partial tuning at {temp_project_path}. Continuing tuning...")
        overwrite_val = False  # Continue from existing
    else:
        print(f"🆕 No partial tuning found.  Starting new tuning...")
        overwrite_val = False

    # Define tuner
    tuner = kt.RandomSearch(
        lambda hp: build_lstm_model(
            hp,
            input_shape=(
                x_train_lstm.shape[1],
                x_train_lstm.shape[2]
            )
        ),
        objective="val_loss",
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory=temp_dir,
        project_name=project_name_val,
        overwrite=overwrite_val
    )

    # Early stopping patience as a tunable hyperparameter
    def build_callbacks(hp):
        patience = hp.Int(
            "patience",
            min_value=5,
            max_value=30,
            step=5
        )

        # Learning rate schedule choice
        lr_schedule_choice = hp.Choice("lr_schedule", ["none", "reduce_on_plateau", "exp_decay"])

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose=1
            )
        ]

        if lr_schedule_choice == "reduce_on_plateau":
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                    verbose=1
                )
            )
        elif lr_schedule_choice == "exp_decay":
            def scheduler(epoch, lr):
                decay_rate = hp.Float("decay_rate", 0.8, 0.99, step=0.01)
                return lr * decay_rate
            callbacks.append(tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1))

        return callbacks

    # --- Retry loop for search ---
    for attempt in range(retries):
        try:
            # Run search with tunable batch size and patience
            tuner.search(
                x_train_lstm,
                y_train_lstm,
                epochs=epochs,
                batch_size=kt.HyperParameters(). Choice(
                    "batch_size",
                    values=[32, 64, 128, 256]
                ),
                validation_data=(
                    x_test_lstm,
                    y_test_lstm
                ),
                callbacks=build_callbacks(tuner.oracle.hyperparameters),
                verbose=0
            )
            break  # Exit loop if search is successful
        except (UnicodeDecodeError, tf.errors. FailedPreconditionError) as e:
            print(f"Attempt {attempt+1} failed: {e}")
            # Cleanup failed trial directory before retry
            if os.path.exists(temp_project_path):
                try:
                    shutil.rmtree(temp_project_path)
                    print("Cleaned up failed trial directory.")
                except Exception as ce:
                    print(f"Could not clean trial dir: {ce}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise

    # Get the best model and hyperparameters
    best_trials = tuner.oracle.get_best_trials(num_trials=1)
    if not best_trials:
        raise RuntimeError("Keras Tuner failed to find any successful trials.")
    else:
        print(best_trials)

    best_hp = best_trials[0].hyperparameters # Get the best model hyperparameters

    print("✅ Best hyperparameters found:")
    for param, value in best_hp.values.items():
        print(f"- {param}: {value}")

    best_model = build_lstm_model(
        best_hp,
        input_shape=(
            x_train_lstm.shape[1],
            x_train_lstm.shape[2]
        )
    )

    # Explicitly build with batch dimension
    best_model.build(
        input_shape=(
            None, x_train_lstm.shape[1],
            x_train_lstm. shape[2]
        )
    )
    print("Best model architecture:")
    print(best_model.summary())

    # --- Move tuning folder to script directory ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    final_dest = os.path.join(script_dir, f"tuning_dir\{project_name_val}")
    print(f"Moving tuning folder to: {final_dest}")
    if os.path.exists(final_dest):
        shutil.rmtree(final_dest)

    shutil.move(temp_project_path, final_dest)

    # --- Cleanup temp dir ---
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            print("🧹 Cleaned up local temp directory.")
        except Exception as e:
            print(f"Warning: Could not delete temp directory {temp_dir}. Error: {e}")

    return best_model

def load_best_model_from_finished_tuning(finished_project_path, seq_length, num_features):
    """
    Attempts to load the best model from a finished tuning directory.
    
    Returns:
        best_model: The loaded Keras model, or None if no finished tuning exists
    """
    if not os. path.exists(finished_project_path):
        print(f"No finished tuning found at {finished_project_path}")
        return None
    
    try:
        # Initialize a tuner to access the oracle (without running tuning)
        tuner = kt.RandomSearch(
            lambda hp: build_lstm_model(
                hp,
                input_shape=(seq_length, num_features)
            ),
            objective="val_loss",
            max_trials=1,
            directory=os.path.dirname(finished_project_path),
            project_name=os.path.basename(finished_project_path),
            overwrite=False
        )
        
        # Get the best trials
        best_trials = tuner.oracle.get_best_trials(num_trials=1)
        
        if not best_trials:
            print(f"No completed trials found in {finished_project_path}")
            return None
        
        best_hp = best_trials[0].hyperparameters
        
        print("✅ Best hyperparameters from finished tuning:")
        for param, value in best_hp.values.items():
            print(f"- {param}: {value}")
        
        # Rebuild the best model with the saved hyperparameters
        best_model = build_lstm_model(
            best_hp,
            input_shape=(seq_length, num_features)
        )
        
        # Explicitly build with batch dimension
        best_model.build(
            input_shape=(None, seq_length, num_features)
        )
        
        print("Best model architecture from finished tuning:")
        print(best_model.summary())
        
        return best_model
        
    except Exception as e:
        print(f"Error loading best model from finished tuning: {e}")
        return None

def predict_future_price_changes(ticker, scaler_x, scaler_y, model, selected_features_list, stock_df, prediction_days, historical_prediction_dataset_df=None):
    """
    Predicts the future stock price. 

    Parameters:
    - ticker (str): The stock ticker. 
    - scaler_x (MinMaxScaler): The scaler for x values.
    - scaler_y (MinMaxScaler): The scaler for y values (for inverse-transforming LSTM predictions).
    - model (dict): Dictionary containing 'lstm' and 'rf' models.
    - selected_features_list (list): The list of selected features.
    - stock_df (pandas.DataFrame): A DataFrame containing the stock data.
    - prediction_days (int): The number of days to predict.
    - historical_prediction_dataset_df (pd.DataFrame): Historical prediction data.

    Returns:
    pandas.DataFrame: A DataFrame containing the predicted stock prices. 

    Raises:
    - ValueError: If the prediction could not be completed.
    """
    
    try:
        # Extract individual models from the 'model' dictionary
        lstm_model = model['lstm']
        rf_model = model['rf']

        # Predict the future stock open_Price
        short_term_dynamic_list = [
            '1M', '3M', '6M', '9M', '1Y', '2Y', '3Y', '4Y',
            '5Y', 'sma_40', 'sma_120', 'ema_40', 'ema_120',
            "std_Div_40", "std_Div_120", "bollinger_Band_40_2STD",
            "bollinger_Band_120_2STD", 'p_s', 'p_e', 'p_b',
            'p_fcf', "momentum"
        ]

        features_list = selected_features_list. copy()

        # Append "Date" to the selected_features_list in position 0
        features_list.insert(0, "date")
        features_list.insert(1, "open_Price")
        features_list.insert(2, "1D")

        stock_mod_df = stock_df.copy()

        # --- 3. Prepare historical prediction dataset if provided ---
        if historical_prediction_dataset_df is not None and len(historical_prediction_dataset_df) > 0:
            pred_count = len(historical_prediction_dataset_df)
            pred_dates = stock_df["date"].iloc[-pred_count:].copy(). to_frame()
            
            for run in range(len(historical_prediction_dataset_df)):
                if pred_count > 1:
                    x_lstm_df = stock_mod_df.iloc[-(TIME_STEPS+pred_count-1):-(pred_count-1)][selected_features_list]
                    scaled_x_lstm_df = scaler_x.transform(x_lstm_df)
                    scaled_x_input_lstm = scaled_x_lstm_df.values.reshape(1, TIME_STEPS, scaled_x_lstm_df.shape[1])
                elif pred_count <= 1:
                    x_lstm_df = stock_mod_df.iloc[-(TIME_STEPS):][selected_features_list]
                    scaled_x_lstm_df = scaler_x.transform(x_lstm_df)
                    scaled_x_input_lstm = scaled_x_lstm_df.values.reshape(1, TIME_STEPS, scaled_x_lstm_df.shape[1])

                if pred_count > 1:
                    scaled_x_input_rf_df = historical_prediction_dataset_df.iloc[-pred_count:-(pred_count-1)][selected_features_list]
                elif pred_count <= 1:
                    scaled_x_input_rf_df = historical_prediction_dataset_df.iloc[-pred_count:][selected_features_list]

                # --- Predict with both models ---
                
                # LSTM prediction (scaled) -> inverse transform to original scale
                forecast_lstm_scaled = lstm_model.predict(scaled_x_input_lstm, verbose=0)[0][0]
                forecast_lstm = scaler_y.inverse_transform([[forecast_lstm_scaled]])[0][0]

                # Random Forest prediction (already unscaled - RF is scale-invariant)
                forecast_rf = rf_model.predict(scaled_x_input_rf_df)[0]

                # ENSEMBLE: Average the predictions
                forecast_price_change = (forecast_lstm + forecast_rf) / 2

                # Append a new row to stock_mod_df for this prediction
                stock_mod_df. loc[stock_mod_df["date"] == pred_dates["date"].iloc[run], "1D"] = forecast_price_change
                # Update the open_Price column with the calculated value
                stock_mod_df. loc[stock_mod_df["date"] == pred_dates["date"].iloc[run], "open_Price"] = stock_mod_df["open_Price"].iloc[-(pred_count+1)] * (1 + forecast_price_change)
                pred_count -= 1

        for run in range(prediction_days):
            future_df = stock_mod_df.iloc[-1].copy().to_frame().transpose()
            future_day = pd.to_datetime(stock_mod_df.iloc[-1]["date"]) + relativedelta(days=1)
            while future_day.weekday() >= 5: # 5=Sat, 6=Sun
                future_day += datetime.timedelta(days=1)

            future_df["date"] = future_day.strftime("%Y-%m-%d")

            for feature in short_term_dynamic_list:
                if feature in features_list:
                    # Calculate the return for the last 1 month
                    if feature == "1M":
                        if len(stock_mod_df) <= 21:
                            hist_df = pd.DataFrame(yf.download(ticker, period="1y"))["Open"].reset_index()
                            hist_df = hist_df.loc[hist_df["Date"] < stock_mod_df.iloc[0]["date"]]
                            hist_df = hist_df.rename(columns={ticker: "open_Price", "Date": "date"})
                            hist_df = pd.concat([hist_df, stock_mod_df[["date", "open_Price"]]], axis=0)
                            return_1M = hist_df.iloc[-22:]["Price"].pct_change(21).iloc[-1]
                            future_df["1M"] = return_1M
                        elif len(stock_mod_df) > 21:
                            return_1M = stock_mod_df.iloc[-22:]["open_Price"].pct_change(21).iloc[-1]
                            future_df["1M"] = return_1M

                    # Calculate the return for the last 3 months
                    if feature == "3M":
                        if len(stock_mod_df) <= 63:
                            hist_df = pd.DataFrame(yf.download(ticker, period="1y"))["Open"].reset_index()
                            hist_df = hist_df.loc[hist_df["Date"] < stock_mod_df.iloc[0]["date"]]
                            hist_df = hist_df.rename(columns={ticker: "open_Price", "Date": "date"})
                            hist_df = pd.concat([hist_df, stock_mod_df[["date", "open_Price"]]], axis=0)
                            return_3M = hist_df.iloc[-64:]["open_Price"].pct_change(63).iloc[-1]
                            future_df["3M"] = return_3M
                        elif len(stock_mod_df) > 63:
                            return_3M = stock_mod_df.iloc[-64:]["open_Price"].pct_change(63).iloc[-1]
                            future_df["3M"] = return_3M

                    # Calculate the return for the last 6 months
                    if feature == "6M":
                        if len(stock_mod_df) <= 126:
                            hist_df = pd.DataFrame(yf.download(ticker, period="1y"))["Open"].reset_index()
                            hist_df = hist_df.loc[hist_df["Date"] < stock_mod_df.iloc[0]["date"]]
                            hist_df = hist_df.rename(columns={ticker: "open_Price", "Date": "date"})
                            hist_df = pd.concat([hist_df, stock_mod_df[["date", "open_Price"]]], axis=0)
                            return_6M = hist_df.iloc[-127:]["open_Price"].pct_change(126).iloc[-1]
                            future_df["6M"] = return_6M
                        elif len(stock_mod_df) > 126:
                            return_6M = stock_mod_df.iloc[-127:]["open_Price"].pct_change(126).iloc[-1]
                            future_df["6M"] = return_6M

                    # Calculate the return for the last 9 months
                    if feature == "9M":
                        if len(stock_mod_df) <= 189:
                            hist_df = pd.DataFrame(yf.download(ticker, period="1y"))["Open"].reset_index()
                            hist_df = hist_df.loc[hist_df["Date"] < stock_mod_df.iloc[0]["date"]]
                            hist_df = hist_df.rename(columns={ticker: "open_Price", "Date": "date"})
                            hist_df = pd.concat([hist_df, stock_mod_df[["date", "open_Price"]]], axis=0)
                            return_9M = hist_df.iloc[-190:]["open_Price"].pct_change(189).iloc[-1]
                            future_df["9M"] = return_9M
                        elif len(stock_mod_df) > 189:
                            return_9M = stock_mod_df.iloc[-190:]["open_Price"].pct_change(189).iloc[-1]
                            future_df["9M"] = return_9M

                    # Calculate the return for the last 1 year
                    if feature == "1Y":
                        if len(stock_mod_df) <= 252:
                            hist_df = pd.DataFrame(yf.download(ticker, period="2y"))["Open"].reset_index()
                            hist_df = hist_df.loc[hist_df["Date"] < stock_mod_df.iloc[0]["date"]]
                            hist_df = hist_df.rename(columns={ticker: "open_Price", "Date": "date"})
                            hist_df = pd.concat([hist_df, stock_mod_df[["date", "open_Price"]]], axis=0)
                            return_1Y = hist_df.iloc[-253:]["open_Price"].pct_change(252).iloc[-1]
                            future_df["1Y"] = return_1Y
                        elif len(stock_mod_df) > 252:
                            return_1Y = stock_mod_df.iloc[-253:]["open_Price"].pct_change(252).iloc[-1]
                            future_df["1Y"] = return_1Y

                    # Calculate the return for the last 2 years
                    if feature == "2Y":
                        if len(stock_mod_df) <= 504:
                            hist_df = pd.DataFrame(yf.download(ticker, period="3y"))["Open"].reset_index()
                            hist_df = hist_df.loc[hist_df["Date"] < stock_mod_df.iloc[0]["date"]]
                            hist_df = hist_df.rename(columns={ticker: "open_Price", "Date": "date"})
                            hist_df = pd.concat([hist_df, stock_mod_df[["date", "open_Price"]]], axis=0)
                            return_2Y = hist_df.iloc[-505:]["open_Price"].pct_change(504).iloc[-1]
                            future_df["2Y"] = return_2Y
                        elif len(stock_mod_df) > 504:
                            return_2Y = stock_mod_df.iloc[-505:]["open_Price"].pct_change(504).iloc[-1]
                            future_df["2Y"] = return_2Y

                    # Calculate the return for the last 3 years
                    if feature == "3Y":
                        if len(stock_mod_df) <= 756:
                            hist_df = pd.DataFrame(yf.download(ticker, period="4y"))["Open"].reset_index()
                            # print(hist_df)
                            # print(hist_df.info())
                            hist_df = hist_df.loc[hist_df["Date"] < stock_mod_df.iloc[0]["date"]]
                            hist_df = hist_df.rename(columns={ticker: "open_Price", "Date": "date"})
                            hist_df = pd.concat([hist_df, stock_mod_df[["date", "open_Price"]]], axis=0)
                            return_3Y = hist_df.iloc[-757:]["open_Price"].pct_change(756).iloc[-1]
                            future_df["3Y"] = return_3Y
                        elif len(stock_mod_df) > 756:
                            return_3Y = stock_mod_df.iloc[-757:]["open_Price"].pct_change(756).iloc[-1]
                            future_df["3Y"] = return_3Y

                    # Calculate the return for the last 4 years
                    if feature == "4Y":
                        if len(stock_mod_df) <= 1008:
                            hist_df = pd.DataFrame(yf.download(ticker, period="5y"))["Open"].reset_index()
                            hist_df = hist_df.loc[hist_df["Date"] < stock_mod_df.iloc[0]["date"]]
                            hist_df = hist_df.rename(columns={ticker: "open_Price", "Date": "date"})
                            hist_df = pd.concat([hist_df, stock_mod_df[["date", "open_Price"]]], axis=0)
                            return_4Y = hist_df.iloc[-1009:]["open_Price"].pct_change(1008).iloc[-1]
                            future_df["4Y"] = return_4Y
                        elif len(stock_mod_df) > 1008:
                            return_4Y = stock_mod_df.iloc[-1009:]["open_Price"].pct_change(1008).iloc[-1]
                            future_df["4Y"] = return_4Y

                    # Calculate the return for the last 5 years
                    if feature == "5Y":
                        if len(stock_mod_df) <= 1260:
                            hist_df = pd.DataFrame(yf.download(ticker, period="6y"))["Open"].reset_index()
                            hist_df = hist_df.loc[hist_df["Date"] < stock_mod_df.iloc[0]["date"]]
                            hist_df = hist_df.rename(columns={ticker: "open_Price", "Date": "date"})
                            hist_df = pd.concat([hist_df, stock_mod_df[["date", "open_Price"]]], axis=0)
                            return_5Y = hist_df.iloc[-1261:]["open_Price"].pct_change(1260).iloc[-1]
                            # print(return_5Y)
                            future_df["5Y"] = return_5Y
                        elif len(stock_mod_df) > 1260:
                            # print(stock_mod_df)
                            return_5Y = stock_mod_df.iloc[-1261:]["open_Price"].pct_change(1260).iloc[-1]
                            # print(return_5Y)
                            future_df["5Y"] = return_5Y

                    # Calculate the 40 days simple moving average
                    if feature == "sma_40":
                        sma_40 = stock_mod_df.iloc[-40:]["open_Price"].mean()
                        future_df["sma_40"] = sma_40

                    # Calculate the 120 days simple moving average
                    if feature == "sma_120":
                        sma_120 = stock_mod_df.iloc[-120:]["open_Price"].mean()
                        future_df["sma_120"] = sma_120

                    # Calculate the 40 days exponential moving average
                    if feature == "ema_40":
                        ema_40 = stock_mod_df.iloc[-40:]["open_Price"].ewm(span=40).mean().iloc[-1]
                        future_df["ema_40"] = ema_40

                    # Calculate the 120 days exponential moving average
                    if feature == "ema_120":
                        ema_120 = stock_mod_df.iloc[-120:]["open_Price"].ewm(span=120).mean().iloc[-1]
                        future_df["ema_120"] = ema_120

                    # Calculate the 40 days standard deviation
                    if feature == "std_Div_40":
                        std_div_40 = stock_mod_df.iloc[-40:]["open_Price"].std()
                        future_df["std_Div_40"] = std_div_40

                    # Calculate the 120 days standard deviation
                    if feature == "std_Div_120":
                        std_div_120 = stock_mod_df.iloc[-120:]["open_Price"].std()
                        future_df["std_Div_120"] = std_div_120

                    # Calculate the Bollinger Band for the last 40 days
                    if feature == "bollinger_Band_40_2STD":
                        std_div_40 = stock_mod_df.iloc[-40:]["open_Price"].std()
                        sma_40 = stock_mod_df.iloc[-40:]["open_Price"].mean()
                        bollinger_Band_40_Upper = (sma_40 + (std_div_40 * 2))
                        bollinger_Band_40_Lower = (sma_40 - (std_div_40 * 2))
                        bollinger_band_40 = bollinger_Band_40_Upper - bollinger_Band_40_Lower
                        future_df["bollinger_Band_40_2STD"] = bollinger_band_40

                    # Calculate the Bollinger Band for the last 120 days
                    if feature == "bollinger_Band_120_2STD":
                        std_div_120 = stock_mod_df.iloc[-120:]["open_Price"].std()
                        sma_120 = stock_mod_df.iloc[-120:]["open_Price"].mean()
                        bollinger_Band_120_Upper = (sma_120 + (std_div_120 * 2))
                        bollinger_Band_120_Lower = (sma_120 - (std_div_120 * 2))
                        bollinger_band_120 = bollinger_Band_120_Upper - bollinger_Band_120_Lower
                        future_df["bollinger_Band_120_2STD"] = bollinger_band_120
                        
                    # Calculate the p_s ratio
                    if feature == "p_s":
                        p_s = stock_mod_df.iloc[-1]["open_Price"] / (stock_mod_df.iloc[-1]["revenue"] / stock_mod_df.iloc[-1]["average_shares"])
                        future_df["p_s"] = p_s

                    # Calculate the p_e ratio
                    if feature == "p_e":
                        p_e = stock_mod_df.iloc[-1]["open_Price"] / stock_mod_df.iloc[-1]["eps"]
                        future_df["p_e"] = p_e

                    # Calculate the p_b ratio
                    if feature == "p_b":
                        p_b = stock_mod_df.iloc[-1]["open_Price"] / stock_mod_df.iloc[-1]["book_Value_Per_Share"]
                        future_df["p_b"] = p_b

                    # Calculate the p_fcf ratio
                    if feature == "p_fcf":
                        p_fcf = stock_mod_df.iloc[-1]["open_Price"] / stock_mod_df.iloc[-1]["free_Cash_Flow_Per_Share"]
                        future_df["p_fcf"] = p_fcf

                    # Calculate the momentum
                    if feature == "momentum":
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
            future_df["date"] = pd. to_datetime(future_df["date"])
            
            # Concat the newly calculated day (future_df) to the historical/predicted data (stock_mod_df)
            stock_mod_df = pd. concat([stock_mod_df, future_df], axis=0). reset_index(drop=True)
            
            # --- Prepare Input Data for both models ---

            # A.  LSTM Input (Last TIME_STEPS sequence)
            x_lstm_df = stock_mod_df.iloc[-TIME_STEPS:][selected_features_list]
            scaled_x_lstm_df = scaler_x.transform(x_lstm_df)
            scaled_x_input_lstm = scaled_x_lstm_df.values. reshape(1, TIME_STEPS, scaled_x_lstm_df.shape[1])

            # B.  Random Forest Input (Only the current day's features)
            x_input_rf_df = stock_mod_df. iloc[-1:][selected_features_list]
            scaled_x_input_rf = scaler_x.transform(x_input_rf_df)

            # --- Predict and Ensemble ---

            # LSTM prediction (scaled) -> inverse transform to original scale
            forecast_lstm_scaled = lstm_model.predict(scaled_x_input_lstm, verbose=0)[0][0]
            forecast_lstm = scaler_y.inverse_transform([[forecast_lstm_scaled]])[0][0]

            # Random Forest prediction (already unscaled - RF is scale-invariant)
            forecast_rf = rf_model.predict(scaled_x_input_rf)[0]
            
            # ENSEMBLE: Average the predictions
            forecast_price_change = (forecast_lstm + forecast_rf) / 2
            
            # --- Update stock_mod_df with the Ensemble Forecast ---
            
            # Update the 1D column with the calculated value
            stock_mod_df. loc[len(stock_mod_df)-1, "1D"] = forecast_price_change
            # Update the open_Price column with the calculated value
            stock_mod_df.loc[len(stock_mod_df)-1, "open_Price"] = stock_mod_df.loc[len(stock_mod_df)-2, "open_Price"] * (1 + forecast_price_change)

        # --- Final cleanup ---
        columns_to_convert = stock_mod_df.columns.drop(["date", "ticker", "currency"]).to_list()
        for column in columns_to_convert:
            stock_mod_df[column] = stock_mod_df[column].astype(float)

        stock_mod_df = stock_mod_df[features_list]
        return stock_mod_df

    except Exception as e:
        print("The prediction could not be completed.  Please check the input data.")
        raise ValueError("The prediction could not be completed. Please check the input data. ") from e

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
    # forecast_data_df = forecast_data_df.loc[forecast_data_df["date"] >= stock_data_df.iloc[-1]["date"]]
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
    plt.title(f"""Stock Price Prediction of {stock_data_df.iloc[0]["ticker"]}""")
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

# Run the main function
if __name__ == "__main__":
    import db_interactions

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    start_time = time.time()

    # Import stock symbols from DB
    stock_symbols_list = db_interactions.import_ticker_list()
    print(stock_symbols_list)
    stock_symbol = stock_symbols_list[0]
    stock_symbol = "DEMANT.CO"
    print(stock_symbol)

    # Import stock data
    stock_data_df = db_interactions.import_stock_dataset(stock_symbol)
    # Change the date column to datetime 64
    stock_data_df["date"] = pd.to_datetime(stock_data_df["date"])
    # Drop the columns that are empty
    stock_data_df = stock_data_df.dropna(axis=0, how="any")
    stock_data_df = stock_data_df.dropna(axis=1, how="any")

    # Split the dataset into training, test data and prediction data
    test_size = 0.20
    scaler_x, scaler_y, x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled, x_Predictions = split_dataset.dataset_train_test_split(stock_data_df, test_size, 1)

    # Inverse-transform y values for Random Forest (RF is scale-invariant, needs unscaled y)
    y_train_unscaled = scaler_y. inverse_transform(y_train_scaled. reshape(-1, 1)).flatten()
    y_test_unscaled = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

    # Convert to DataFrames for feature selection
    x_training_data = pd.DataFrame(x_train_scaled)
    x_test_data = pd.DataFrame(x_test_scaled)
    y_training_data_df = pd.Series(y_train_unscaled)  # UNSCALED for Random Forest
    y_test_data_df = pd.Series(y_test_unscaled)      # UNSCALED for Random Forest
    prediction_data = x_Predictions

    # Feature selection
    max_features = len(x_training_data.columns)
    # print(f"Max features:\n{max_features}")
    feature_amount = max_features
    x_training_dataset, x_test_dataset, x_prediction_dataset, selected_features_model, selected_features_list = dimension_reduction.feature_selection(feature_amount, x_training_data, x_test_data, y_training_data_df, y_test_data_df, prediction_data, stock_data_df)

    # Combine the reduced dataset with the stock price
    x_training_dataset_df = pd.DataFrame(x_training_dataset, columns=selected_features_list)
    y_training_data_df = y_training_data_df.reset_index(drop=True)

    x_test_dataset_df = pd.DataFrame(x_test_dataset, columns=selected_features_list)
    y_test_data_df = y_test_data_df.reset_index(drop=True)

    x_prediction_dataset_df = pd.DataFrame(x_prediction_dataset, columns=selected_features_list)
    TIME_STEPS = len(x_prediction_dataset_df)

    # Create scaled y values for LSTM training
    y_train_scaled_for_lstm = pd.Series(y_train_scaled)
    y_test_scaled_for_lstm = pd.Series(y_test_scaled)

    # Predict the stock price using different models
    # 1.  Tune and get the best LSTM model (uses SCALED y)
    print("Starting LSTM Hyperparameter Tuning...")
    lstm = tune_lstm_model(stock_symbol, x_training_dataset_df, y_train_scaled_for_lstm, x_test_dataset_df, y_test_scaled_for_lstm, max_trials=10, executions_per_trial=2, epochs=500)

    # 2. Tune and get the best Random Forest model (uses UNSCALED y)
    print("Starting Random Forest Hyperparameter Tuning...")
    rf_model = tune_random_forest_model(stock_symbol, x_training_dataset_df, y_training_data_df, max_trials=20)

    # 3. Predict the future stock price changes
    amount_of_days = 90
    forecast_df = predict_future_price_changes(
        ticker=stock_symbol, 
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        model={'lstm': lstm, 'rf': rf_model},
        selected_features_list=selected_features_list, 
        stock_df=stock_data_df, 
        prediction_days=amount_of_days,
        historical_prediction_dataset_df=x_prediction_dataset_df
    )
    # print("Forecast DataFrame:")
    # print(forecast_df)
    # print(forecast_df.info())
    # plt.plot(forecast_df["open_Price"], color="green")
    # plt.xlabel("Date")
    # plt.ylabel("Opening price")
    # legend_list = ["Predicted Stock Price"]
    # plt.legend(legend_list,
    #     loc="best"
    # )
    # stock_name = stock_data_df.iloc[0]["ticker"]
    # graph_name = str(f"future_stock_prediction_of_{stock_name}.png")
    # my_path = os.path.abspath(__file__)
    # path = os.path.dirname(my_path)
    # # Save the graph
    # try:
    #     plt.savefig(os.path.join(path, "generated_graphs", graph_name), bbox_inches="tight", pad_inches=0.5, transparent=False, format="png")
    #     plt.clf()
    #     plt.close("all")

    # except FileNotFoundError as e:
    #     raise FileNotFoundError("The graph could not be saved. Please check the file name or path.") from e

    # Calculate the predicted profit
    calculate_predicted_profit(forecast_df, amount_of_days)

    # Plot the graph
    plot_graph(stock_data_df, forecast_df)

    # Run a Monte Carlo simulation
    year_amount = 10
    sim_amount = 1000
    monte_carlo_day_df, monte_carlo_year_df = monte_carlo_sim.monte_carlo_analysis(0, stock_data_df, forecast_df, year_amount, sim_amount)
    forecast_df = forecast_df.rename(columns={"open_Price": stock_symbol + "_price"})

    # Calculate the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds to build dataset and ML models.")
