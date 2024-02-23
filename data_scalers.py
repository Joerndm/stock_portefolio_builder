import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Create a function to fits the scaler for the dataset with MinMaxScaler
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
        scaling = MinMaxScaler()
        # Fit the scaler to the data and transform it
        scaler = scaling.fit(data)
        return scaler


    except ValueError:
        raise ValueError("The specified component amount is greater than the number of features in the dataset.")
    
# Create a function to transform the dataset with MinMaxScaler
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
    

    except ValueError:
        raise ValueError("The specified component amount is greater than the number of features in the dataset.")

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


    except ValueError:
        raise ValueError("The specified component amount is greater than the number of features in the dataset.")