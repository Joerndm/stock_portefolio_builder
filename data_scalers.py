from sklearn.preprocessing import StandardScaler, MinMaxScaler

def data_preprocessing_minmax_scaler(data):
    # Scale data before applying PCA
    scaling = MinMaxScaler()
    # Fit the scaler to the data and transform it
    Scaled_data = scaling.fit_transform(data)
    # Scaled_data_df = pd.DataFrame(Scaled_data, columns=data.columns)
    # # Print summary statistics of the scaled dataframe
    # print(Scaled_data_df.describe().transpose())
    return Scaled_data
    
def data_preprocessing_std_scaler(data):
    # Scale data before applying PCA
    scaling = StandardScaler()
    # Fit the scaler to the data and transform it
    Scaled_data = scaling.fit_transform(data)
    # Scaled_data_df = pd.DataFrame(Scaled_data, columns=data.columns)
    # # Print summary statistics of the scaled dataframe
    # print(Scaled_data_df.describe().transpose())
    return Scaled_data