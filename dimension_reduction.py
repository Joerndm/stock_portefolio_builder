import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import r_regression
from sklearn.decomposition import PCA

import import_csv_file
import split_dataset

# Create a function to reduce the dataset dimensions with SelectKBest
def feature_selection(dimensions, x_traning_data, x_test_data, y_traning_data, y_test_data, prediction_data, dataset_df):
    """
    Reduce the dataset dimensions with SelectKBest.

    Parameters:
    - dimensions (int): The number of features to select.
    - x_traning_data (numpy.ndarray): The training data.
    - x_test_data (numpy.ndarray): The test data.
    - y_traning_data (numpy.ndarray): The training labels.
    - y_test_data (numpy.ndarray): The test labels.
    - prediction_data (numpy.ndarray): The prediction data.
    - dataset_df (pandas.DataFrame): The dataset.

    Returns:
    - numpy.ndarray: The reduced training data.
    - numpy.ndarray: The reduced test data.
    - numpy.ndarray: The reduced prediction data.

    Raises:
    - ValueError: If the specified dimension amount is greater than the number of features in the dataset.
    """

    if dimensions > x_traning_data.shape[1]:
        raise ValueError("The specified dimension amount is greater than the number of features in the dataset.")

    # Create a SelectKBest object to select features with two best ANOVA F-Values
    selector = SelectKBest(r_regression, k=dimensions)
    # Combince numpy arrays x_traning_data and x_test_data
    x_fitting_data = np.concatenate((x_traning_data, x_test_data), axis=0)
    # print(x_fitting_data.shape)
    y_fitting_data = np.concatenate((y_traning_data, y_test_data), axis=0)
    # print(y_fitting_data.shape)
    # Apply the SelectKBest object to the features and target
    selected_features = selector.fit(x_fitting_data, y_fitting_data)
    reduced_traning_dataset = selected_features.transform(x_traning_data)
    reduced_test_dataset = selected_features.transform(x_test_data)
    reduced_prediction_dataset = selected_features.transform(prediction_data)
    # Check the shape of the selected data
    print(f"Shape of traning dataset before feature selection: {x_traning_data.shape}")
    print(f"Shape of test dataset before feature selection: {x_test_data.shape}")
    print(f"Shape of prediction dataset before feature selection: {prediction_data.shape}")
    # Check the shape of the selected data
    print(f"Shape of traning dataset after feature selection: {reduced_traning_dataset.shape}")
    print(f"Shape of test dataset after feature selection: {reduced_test_dataset.shape}")
    print(f"Shape of prediction dataset after feature selection: {reduced_prediction_dataset.shape}")
    # Check the selected features
    dataset_column_list = dataset_df.columns
    drop_colum_list = ["Date", "Name", "Ticker", "Currency"]
    for column in drop_colum_list:
        if column in dataset_column_list:
            dataset_column_list = dataset_column_list.drop([column])


    selected_features_list = []
    for i in range(len(selected_features.get_support())):
        if selected_features.get_support()[i]:
            selected_features_list.append(dataset_column_list[i])


    # print(selected_features_list)
    # # Check the ANOVA F-Values
    # print("ANOVA F-Values: ")
    # print(selected_features.scores_)
    # # Check the p-values
    # print("P-Values: ")
    # print(selected_features.pvalues_)
    return reduced_traning_dataset, reduced_test_dataset, reduced_prediction_dataset, selected_features_list

# Create a function to reduce the dataset dimensions with PCA
def  pca_dataset_transformation(x_traning_data, x_test_data, prediction_data, component_amount):
    """
    Reduce the dataset dimensions with PCA.
    
    Parameters:
    - x_traning_data (numpy.ndarray): The training data.
    - x_test_data (numpy.ndarray): The test data.
    - prediction_data (numpy.ndarray): The prediction data.
    - component_amount (int): The number of principal components to keep.
    
    Returns:
    - numpy.ndarray: The reduced training data.
    - numpy.ndarray: The reduced test data.
    - numpy.ndarray: The reduced prediction data.
    
    Raises:
    - ValueError: If the specified component amount is greater than the number of features in the dataset.
    """

    if component_amount > x_traning_data.shape[1]:
        raise ValueError("The specified component amount is greater than the number of features in the dataset.")
    
    # Create a PCA model
    pca_model = PCA(n_components=component_amount)
    x_fitting_data = np.concatenate((x_traning_data, x_test_data), axis=0)
    # Fit the PCA model to the scaled data and transform it
    pca_model.fit(x_fitting_data)
    reduced_traning_dataset = pca_model.transform(x_traning_data)
    reduced_test_dataset = pca_model.transform(x_test_data)
    reduced_prediction_dataset = pca_model.transform(prediction_data)
    # Check the shape of the scaled data
    print(f"Shape of traning dataset before PCA transformation: {x_traning_data.shape}")
    print(f"Shape of test dataset before PCA transformation: {x_test_data.shape}")
    print(f"Shape of prediction dataset before PCA transformation: {prediction_data.shape}")
    # Check the dimensions of data after PCA
    print(f"Shape of traning dataset after PCA transformation: {reduced_traning_dataset.shape}")
    print(f"Shape of test dataset after PCA transformation: {reduced_test_dataset.shape}")
    print(f"Shape of prediction dataset after PCA transformation: {reduced_prediction_dataset.shape}")
    # check how much variance is explained by each principal component
    print("Variance explained by each principal component: ")
    print(pca_model.explained_variance_ratio_)
    # # Access the principal components
    # principal_components = pca_model.components_
    # # Check the shape of the principal components
    # print(principal_components.shape)
    # x = 1
    # for component in principal_components:
    #     # Print the principal components
    #     print(f"Principal Component {x}")
    #     print(component)
    #     x += 1     


    return reduced_traning_dataset, reduced_test_dataset, reduced_prediction_dataset

# Run the main function
if __name__ == "__main__":
    stock_data_df = import_csv_file.import_as_df('stock_data_single.csv')
    x_training_data, x_test_data, y_training_data, y_test_data, prediction_data = split_dataset.dataset_train_test_split(stock_data_df, 0.20, 1)
    x_training_dataset, x_test_dataset, x_prediction_dataset, selected_features_list = feature_selection(15, x_training_data, x_test_data, y_training_data, y_test_data, prediction_data, stock_data_df)
    # x_training_dataset, x_test_dataset, x_prediction_dataset = pca_dataset_transformation(x_training_data, x_test_data, prediction_data, 10)
    x_training_dataset_df = pd.DataFrame(x_training_dataset)
    x_test_dataset_df = pd.DataFrame(x_test_dataset)
    x_prediction_dataset_df = pd.DataFrame(x_prediction_dataset)
    # print(x_training_dataset_df)
    # print(x_test_dataset_df)
    # print(x_prediction_dataset_df)
