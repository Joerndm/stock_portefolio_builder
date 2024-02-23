from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import math

import import_stock_data
import data_scalers

# Split the dataset into training and test data
def dataset_train_test_split(dataset_dataframe, ts, rs):
    """
    Split the dataset into training and test data.

    Parameters:
    - dataset_dataframe (pandas.DataFrame): The dataset to split.
    - ts (float): The size of the test data.
    - rs (int): The random state for the split.

    Returns:
    - numpy.ndarray: The training data.
    - numpy.ndarray: The test data.
    - numpy.ndarray: The training labels.
    - numpy.ndarray: The test labels.
    - numpy.ndarray: The prediction data.

    Raises:
    - KeyError: If the dataset does not have the required columns.
    """
    try:    
        # Drop the columns that are not needed
        drop_colum_list = ["Date", "Name", "Ticker", "Currency"]
        for column in drop_colum_list:
            if column in dataset_dataframe.columns:
                dataset_dataframe = dataset_dataframe.drop([column], axis=1)
        

        train_data_df = dataset_dataframe
        forecast_out = int(math.ceil(0.05 * len(train_data_df)))
        train_data_df["Prediction"] = train_data_df.iloc[0:-forecast_out]["1D"]
        scaler = data_scalers.data_preprocessing_minmax_scaler_fit(train_data_df.drop(["Price", "1D", "Prediction"], axis=1))
        scaler.set_output(transform="pandas")
        scaled_x = data_scalers.data_preprocessing_minmax_scaler_transform(scaler, train_data_df.drop(["Price", "1D", "Prediction"], axis=1))
        x_Predictions = scaled_x.iloc[-forecast_out:].copy()
        x = scaled_x.iloc[:-forecast_out].copy()
        train_data_df = train_data_df.dropna(axis=0, how="any")
        y = train_data_df["Prediction"]
        x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size=ts, random_state=rs)
        return scaler, x_train, x_test, y_train, y_test, x_Predictions
    

    except KeyError:
        raise KeyError("Dataset does not have the required columns.")

# Run the main function
if __name__ == "__main__":
    stock_data_df = import_stock_data.import_as_df_from_csv('stock_data_single.csv')
    scaler, x_training_data, x_test_data, y_training_data, y_test_data, prediction_data = dataset_train_test_split(stock_data_df, 0.20, 1)
    print(x_training_data)
    print(y_training_data)
    print(x_test_data)
    print(y_test_data)
    print(prediction_data)