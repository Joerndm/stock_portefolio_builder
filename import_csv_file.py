import pandas as pd

def import_as_df(csv_file):
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
        df = pd.read_csv(csv_file)

        # Check if the 'Symbol' column exists in the DataFrame
        if 'Date' not in df.columns:
            raise KeyError("CSV file does not have a column named 'Date'.")

        # Return the DataFrame with stock symbols
        return df

    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file '{csv_file}' does not exist.")
    