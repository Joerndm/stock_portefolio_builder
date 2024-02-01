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
    print(train_data_df)
    forecast_out = int(math.ceil(0.05 * len(train_data_df)))
    print(forecast_out)
    train_data_df["Prediction"] = train_data_df.iloc[0:-forecast_out]["Price"]
    print(train_data_df)
    print(train_data_df.columns)
    x = np.array(train_data_df.drop(["Price", "Prediction"], axis=1))
    scaled_x = data_scalers.data_preprocessing_minmax_scaler(x)
    # print(scaled_x)
    x_Predictions = scaled_x[-forecast_out:]
    # print(x_Predictions)
    print(len(x_Predictions))
    x = scaled_x[:-forecast_out]
    # print(x)
    print(len(x))
    train_data_df = train_data_df.dropna(axis=0, how="any")
    y = np.array(train_data_df["Prediction"])
    # print(y)
    print(len(y))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=ts, random_state=rs)
    return x_train, x_test, y_train, y_test, x_Predictions


if __name__ == "__main__":
    import import_csv_file

    stock_data_df = import_csv_file.import_as_df('stock_data_single_v2.csv')
    x_training_data, x_test_data, y_training_data, y_test_data, prediction_data = dataset_train_test_split(stock_data_df, 0.20, 1)
