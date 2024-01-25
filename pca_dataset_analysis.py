import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import import_csv_file
import split_dataset

def  pca_dataset_transformation(x_traning_data, x_test_data, prediction_data, component_amount):
    # Create a PCA model
    pca_model = PCA(n_components=component_amount)
    # Fit the PCA model to the scaled data and transform it
    pca_model.fit(x_traning_data)
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

if __name__ == "__main__":
    stock_data_df = import_csv_file.import_as_df('stock_data_single_v2.csv')
    x_training_data, x_test_data, y_training_data, y_test_data, prediction_data = split_dataset.dataset_train_test_split(stock_data_df, 0.20, 1)
    x_training_dataset, x_test_dataset, x_prediction_dataset = pca_dataset_transformation(x_training_data, x_test_data, prediction_data, 4)
    x_training_dataset_df = pd.DataFrame(x_training_dataset)
    x_test_dataset_df = pd.DataFrame(x_test_dataset)
    x_prediction_dataset_df = pd.DataFrame(x_prediction_dataset)
    print(x_training_dataset_df)
    print(x_test_dataset_df)
    print(x_prediction_dataset_df)
