"""
Data Scalers Module

This module provides functions for scaling datasets using scikit-learn's preprocessing scalers.
It includes implementations for both MinMaxScaler and StandardScaler with separate fit and 
transform operations for MinMaxScaler to support training/testing data workflows.

Functions:
    data_preprocessing_minmax_scaler_fit: Fits a MinMaxScaler to the dataset.
    data_preprocessing_minmax_scaler_transform: Transforms data using a fitted MinMaxScaler.
    data_preprocessing_std_scaler: Fits and transforms data using StandardScaler.
"""
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def data_preprocessing_minmax_scaler_fit(data):
    """
    Fits the scaler for the dataset with MinMaxScaler.

    Parameters:
    - data (pandas.DataFrame): The dataset to scale.

    Returns:
    - MinMaxScaler: The fitted scaler.

    Raises:
    - ValueError: If the specified component amount is greater than the number of features in the dataset.
    """

    try:
        # Scale data before applying PCA
        # scaling = MinMaxScaler(feature_range=(-1, 1))
        scaling = MinMaxScaler()
        # Fit the scaler to the data and transform it
        scaler = scaling.fit(data)
        return scaler

    except ValueError as e:
        raise ValueError("The specified component amount is greater than the number of features in the dataset.") from e

def data_preprocessing_minmax_scaler_transform(scaler, data):
    """
    Transform the dataset with MinMaxScaler.

    Parameters:
    - scaler (MinMaxScaler): The fitted scaler.

    Returns:
    - numpy.ndarray: The scaled dataset.

    Raises:
    - ValueError: If the specified component amount is greater than the number of features in the dataset.
    """

    try:
        # Transform the data
        Scaled_data = scaler.transform(data)
        return Scaled_data

    except ValueError as e:
        raise ValueError("The specified component amount is greater than the number of features in the dataset.") from e

def data_preprocessing_std_scaler(data):
    """
    Fits the scaler for the dataset with StandardScaler.
    
    Parameters:
    - data (pandas.DataFrame): The dataset to scale.

    Returns:
    - numpy.ndarray: The scaled dataset.
    
    Raises:
    - ValueError: If the specified component amount is greater than the number of features in the dataset.
    """

    try:
        # Scale data before applying PCA
        scaling = StandardScaler()
        # Fit the scaler to the data and transform it
        Scaled_data = scaling.fit_transform(data)
        return Scaled_data

    except ValueError as e:
        raise ValueError("The specified component amount is greater than the number of features in the dataset.") from e
