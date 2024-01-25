from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import math

import data_scalers

def dataset_train_test_split(dataset_dataframe, ts, rs):
    drop_colum_list = ["Date", "Name", "Ticker", "Currency"]
    for column in drop_colum_list:
        if column in dataset_dataframe.columns:
            dataset_dataframe = dataset_dataframe.drop([column], axis=1)
    

    train_data_df = dataset_dataframe
    forecast_out = int(math.ceil(0.05 * len(train_data_df)))
    prediction_data_dict = {"Prediction": train_data_df["Price"].shift(-forecast_out).values}
    prediction_data_df = pd.DataFrame(prediction_data_dict)
    train_data_df = train_data_df.join(prediction_data_df)
    x = np.array(train_data_df.drop(["Price", "Prediction"], axis=1))
    scaled_x = data_scalers.data_preprocessing_std_scaler(x)
    # print(scaled_x)
    x_Predictions = scaled_x[-forecast_out:]
    # print(x_Predictions)
    print(len(x_Predictions))
    x = scaled_x[:-forecast_out]
    print(len(x))
    train_data_df = train_data_df.dropna(axis=0, how="any")
    y = np.array(train_data_df["Prediction"])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=ts, random_state=rs)
    return x_train, x_test, y_train, y_test, x_Predictions