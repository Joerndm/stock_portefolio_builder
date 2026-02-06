"""
Dimension Reduction Module for Stock Portfolio Builder.

This module provides functions for reducing the dimensionality of stock market datasets
using various feature selection and transformation techniques. It supports three main approaches:

1. SelectKBest: Statistical feature selection using r_regression scores
2. Random Forest: Feature selection based on tree-based feature importance
3. PCA: Principal Component Analysis for dimensionality reduction

The module ensures proper train/validation/test separation to prevent data leakage
during the feature selection process.

Functions:
    feature_selection: Reduce dimensions using SelectKBest with r_regression.
    feature_selection_rf: Reduce dimensions using Random Forest feature importance.
    pca_dataset_transformation: Reduce dimensions using Principal Component Analysis.

Example:
    >>> import dimension_reduction as dr
    >>> x_train, x_val, x_test, x_pred, selector, features = dr.feature_selection(
    ...     30, x_training_data, x_val_data, x_test_data,
    ...     y_training_data, y_val_data, y_test_data,
    ...     prediction_data, dataset_df
    ... )

Author: Stock Portfolio Builder Team
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import r_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

import db_interactions
import split_dataset

def feature_selection(dimensions, x_training_data, x_val_data, x_test_data, y_training_data, y_val_data, y_test_data, prediction_data, dataset_df):
    """
    Reduce the dataset dimensions with SelectKBest.

    Parameters:
    - dimensions (int): The number of features to select.
    - x_training_data (pandas.DataFrame): The training data.
    - x_val_data (pandas.DataFrame): The validation data.
    - x_test_data (pandas.DataFrame): The test data.
    - y_training_data (pandas.Series): The training labels.
    - y_val_data (pandas.Series): The validation labels.
    - y_test_data (pandas.Series): The test labels.
    - prediction_data (numpy.ndarray): The prediction data.
    - dataset_df (pandas.DataFrame): The dataset.

    Returns:
    - numpy.ndarray: The reduced training data.
    - numpy.ndarray: The reduced validation data.
    - numpy.ndarray: The reduced test data.
    - numpy.ndarray: The reduced prediction data.
    - SelectKBest: The fitted selector object.
    - list: The list of selected feature names.

    Raises:
    - ValueError: If the specified dimension amount is greater than the number of features in the dataset.
    """

    if dimensions > x_training_data.shape[1]:
        raise ValueError("The specified dimension amount is greater than the number of features in the dataset.")

    # Create a SelectKBest object to select features with best r_regression scores
    selector = SelectKBest(r_regression, k=dimensions)
    
    # Fit the selector on TRAINING DATA ONLY to prevent data leakage
    # The test set must remain completely independent
    selected_features = selector.fit(x_training_data, y_training_data)
    
    # Transform all three datasets
    reduced_training_dataset = selected_features.transform(x_training_data)
    reduced_val_dataset = selected_features.transform(x_val_data)
    reduced_test_dataset = selected_features.transform(x_test_data)
    reduced_prediction_dataset = selected_features.transform(prediction_data)
    
    # Check the shape of the selected data
    print(f"\n{'='*60}")
    print("[DATA] FEATURE SELECTION RESULTS")
    print(f"{'='*60}")
    print(f"Shape of training dataset before feature selection: {x_training_data.shape}")
    print(f"Shape of validation dataset before feature selection: {x_val_data.shape}")
    print(f"Shape of test dataset before feature selection: {x_test_data.shape}")
    print(f"Shape of prediction dataset before feature selection: {prediction_data.shape}")
    
    print(f"\nShape of training dataset after feature selection: {reduced_training_dataset.shape}")
    print(f"Shape of validation dataset after feature selection: {reduced_val_dataset.shape}")
    print(f"Shape of test dataset after feature selection: {reduced_test_dataset.shape}")
    print(f"Shape of prediction dataset after feature selection: {reduced_prediction_dataset.shape}")
    
    # Check the selected features
    dataset_column_list = dataset_df.columns
    drop_colum_list = ["date", "ticker", "currency", "open_Price", "high_Price", "low_Price", "close_Price", "trade_Volume", "1D"]
    for column in drop_colum_list:
        if column in dataset_column_list:
            dataset_column_list = dataset_column_list.drop([column])

    selected_features_list = []
    for i in range(len(selected_features.get_support())):
        if selected_features.get_support()[i]:
            selected_features_list.append(dataset_column_list[i])

    print(f"\n[OK] Selected {len(selected_features_list)} features:")
    for i, feature in enumerate(selected_features_list, 1):
        print(f"   {i}.{feature}")
    
    print(f"{'='*60}\n")

    return reduced_training_dataset, reduced_val_dataset, reduced_test_dataset, reduced_prediction_dataset, selected_features, selected_features_list

def feature_selection_rf(dimensions, x_training_data, x_val_data, x_test_data, y_training_data, y_val_data, y_test_data, prediction_data, dataset_df):
    """
    Reduce the dataset dimensions using RandomForest feature importance.
    Better suited for tree-based models (RF, XGB) than linear correlation.
    Captures non-linear relationships and feature interactions.

    Parameters:
    - dimensions (int): The number of features to select.
    - x_training_data (pandas.DataFrame): The training data.
    - x_val_data (pandas.DataFrame): The validation data.
    - x_test_data (pandas.DataFrame): The test data.
    - y_training_data (pandas.Series): The training labels.
    - y_val_data (pandas.Series): The validation labels.
    - y_test_data (pandas.Series): The test labels.
    - prediction_data (numpy.ndarray): The prediction data.
    - dataset_df (pandas.DataFrame): The dataset.

    Returns:
    - numpy.ndarray: The reduced training data.
    - numpy.ndarray: The reduced validation data.
    - numpy.ndarray: The reduced test data.
    - numpy.ndarray: The reduced prediction data.
    - RandomForestRegressor: The fitted selector object.
    - list: The list of selected feature names.

    Raises:
    - ValueError: If the specified dimension amount is greater than the number of features in the dataset.
    """
    
    if dimensions > x_training_data.shape[1]:
        raise ValueError("The specified dimension amount is greater than the number of features in the dataset.")

    print(f"\n{'='*60}")
    print("[RF] RANDOM FOREST FEATURE SELECTION")
    print(f"{'='*60}")
    print(f"Training RF to identify {dimensions} most important features...")
    
    # Train a Random Forest to get feature importances
    # Use conservative hyperparameters to avoid overfitting the feature selection
    rf_selector = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    
    rf_selector.fit(x_training_data, y_training_data)
    
    # Get feature importances
    importances = rf_selector.feature_importances_
    
    # Select top K features based on importance
    top_indices = np.argsort(importances)[-dimensions:][::-1]
    
    # Transform all datasets using selected features
    reduced_training_dataset = x_training_data.iloc[:, top_indices].values
    reduced_val_dataset = x_val_data.iloc[:, top_indices].values
    reduced_test_dataset = x_test_data.iloc[:, top_indices].values
    reduced_prediction_dataset = prediction_data.iloc[:, top_indices].values
    
    # Print selection results
    print(f"\nShape of training dataset before feature selection: {x_training_data.shape}")
    print(f"Shape of validation dataset before feature selection: {x_val_data.shape}")
    print(f"Shape of test dataset before feature selection: {x_test_data.shape}")
    print(f"Shape of prediction dataset before feature selection: {prediction_data.shape}")
    
    print(f"\nShape of training dataset after feature selection: {reduced_training_dataset.shape}")
    print(f"Shape of validation dataset after feature selection: {reduced_val_dataset.shape}")
    print(f"Shape of test dataset after feature selection: {reduced_test_dataset.shape}")
    print(f"Shape of prediction dataset after feature selection: {reduced_prediction_dataset.shape}")
    
    # Get feature names
    dataset_column_list = dataset_df.columns
    drop_colum_list = ["date", "ticker", "currency", "open_Price", "high_Price", "low_Price", "close_Price", "trade_Volume", "1D"]
    for column in drop_colum_list:
        if column in dataset_column_list:
            dataset_column_list = dataset_column_list.drop([column])

    selected_features_list = [dataset_column_list[i] for i in top_indices]
    
    # Print selected features with their importance scores
    print(f"\n[OK] Selected {len(selected_features_list)} features by importance:")
    for i, (feature, importance) in enumerate(zip(selected_features_list, importances[top_indices]), 1):
        print(f"   {i}. {feature}: {importance:.6f}")
    
    print("\n[INFO] Feature selection method: RandomForest importance (tree-based)")
    print("   Advantages: Captures non-linear relationships and interactions")
    print(f"{'='*60}\n")
    
    return reduced_training_dataset, reduced_val_dataset, reduced_test_dataset, reduced_prediction_dataset, rf_selector, selected_features_list

def pca_dataset_transformation(x_training_data, x_val_data, x_test_data, prediction_data, component_amount):
    """
    Reduce the dataset dimensions with PCA.
    
    Parameters:
    - x_training_data (numpy.ndarray): The training data.
    - x_val_data (numpy.ndarray): The validation data.
    - x_test_data (numpy.ndarray): The test data.
    - prediction_data (numpy.ndarray): The prediction data.
    - component_amount (int): The number of principal components to keep.
    
    Returns:
    - numpy.ndarray: The reduced training data.
    - numpy.ndarray: The reduced validation data.
    - numpy.ndarray: The reduced test data.
    - numpy.ndarray: The reduced prediction data.
    
    Raises:
    - ValueError: If the specified component amount is greater than the number of features in the dataset.
    """

    if component_amount > x_training_data.shape[1]:
        raise ValueError("The specified component amount is greater than the number of features in the dataset.")

    # Create a PCA model
    pca_model = PCA(n_components=component_amount)
    
    # Fit the PCA model on TRAINING DATA ONLY to prevent data leakage
    # The test set must remain completely independent
    pca_model.fit(x_training_data)
    reduced_training_dataset = pca_model.transform(x_training_data)
    reduced_val_dataset = pca_model.transform(x_val_data)
    reduced_test_dataset = pca_model.transform(x_test_data)
    reduced_prediction_dataset = pca_model.transform(prediction_data)
    
    # Check the shape of the data
    print(f"\n{'='*60}")
    print("[PCA] PCA TRANSFORMATION RESULTS")
    print(f"{'='*60}")
    print(f"Shape of training dataset before PCA transformation: {x_training_data.shape}")
    print(f"Shape of validation dataset before PCA transformation: {x_val_data.shape}")
    print(f"Shape of test dataset before PCA transformation: {x_test_data.shape}")
    print(f"Shape of prediction dataset before PCA transformation: {prediction_data.shape}")
    
    print(f"\nShape of training dataset after PCA transformation: {reduced_training_dataset.shape}")
    print(f"Shape of validation dataset after PCA transformation: {reduced_val_dataset.shape}")
    print(f"Shape of test dataset after PCA transformation: {reduced_test_dataset.shape}")
    print(f"Shape of prediction dataset after PCA transformation: {reduced_prediction_dataset.shape}")
    
    # Check how much variance is explained by each principal component
    print("\n[VAR] Variance explained by each principal component:")
    for i, var in enumerate(pca_model.explained_variance_ratio_, 1):
        print(f"   PC{i}: {var:.4f} ({var*100:.2f}%)")
    
    print(f"   Total variance explained: {sum(pca_model.explained_variance_ratio_):.4f} ({sum(pca_model.explained_variance_ratio_)*100:.2f}%)")
    print(f"{'='*60}\n")

    return reduced_training_dataset, reduced_val_dataset, reduced_test_dataset, reduced_prediction_dataset

# Run the main function
if __name__ == "__main__":
    stock_data_df = db_interactions.import_stock_dataset("BAVA.CO")
    print(stock_data_df.info())
    print("stock_data_df")
    
    # Updated call to dataset_train_test_split with three-way split
    scaler_x, scaler_y, x_training_data, x_val_data, x_test_data, y_training_data, y_val_data, y_test_data, prediction_data = split_dataset.dataset_train_test_split(
        stock_data_df, test_size=0.20, validation_size=0.15, rs=1
    )
    
    x_training_data = pd.DataFrame(x_training_data)
    x_val_data = pd.DataFrame(x_val_data)
    x_test_data = pd.DataFrame(x_test_data)
    
    y_training_data = pd.Series(y_training_data)
    y_val_data = pd.Series(y_val_data)
    y_test_data = pd.Series(y_test_data)
    
    print(x_training_data.info())
    
    # Call feature selection with all three datasets
    x_training_dataset, x_val_dataset, x_test_dataset, x_prediction_dataset, selected_features, selected_features_list = feature_selection(
        30, x_training_data, x_val_data, x_test_data, y_training_data, y_val_data, y_test_data, prediction_data, stock_data_df
    )
    
    x_training_dataset_df = pd.DataFrame(x_training_dataset, columns=selected_features_list)
    x_val_dataset_df = pd.DataFrame(x_val_dataset, columns=selected_features_list)
    x_test_dataset_df = pd.DataFrame(x_test_dataset, columns=selected_features_list)
    x_prediction_dataset_df = pd.DataFrame(x_prediction_dataset, columns=selected_features_list)
    
    print("selected_features")
    print(selected_features)
    print("selected_features_list")
    print(selected_features_list)