""""""
import os
import pandas as pd

# Create a function to import stock symbols from a CSV file
def import_as_df_from_csv(csv_file):
    """
    Imports stock symbols from a CSV file and returns a pandas DataFrame.

    The CSV file should have a column named 'Symbol' containing the stock symbols.

    Parameters:
    - csv_file (str): The path to the CSV file.

    Returns:
    pandas.DataFrame: A DataFrame containing the imported stock symbols.

    Raises:
    - FileNotFoundError: If the specified CSV file does not exist.
    - KeyError: If the CSV file does not have a column named 'Symbol'.
    """

    try:
        # Read the CSV file into a DataFrame
        my_path = os.path.abspath(__file__)
        path = os.path.dirname(my_path)
        import_location = os.path.join(path, csv_file)
        # print(import_location)
        df = pd.read_csv(import_location)

        # Check if the 'Symbol' column exists in the DataFrame
        if 'date' not in df.columns:
            raise KeyError("CSV file does not have a column named 'date'.")

        # Return the DataFrame with stock symbols
        return df

    except FileNotFoundError as e:
        raise FileNotFoundError(f"CSV file '{csv_file}' does not exist.") from e
