""""""

import os
from dotenv import load_dotenv

def secret_import():
    """
    Imports the secret environment variables from the .env file and returns them.

    Returns:
    tuple: A tuple containing the secret environment variables.

    Raises:
    FileNotFoundError: If the .env file does not exist.
    KeyError: If one or more secret environment variables are not set in the .env file.

    """
    try:
        load_dotenv("dev.env")
        try:
            db_host = os.getenv("DB_HOST")
            db_user = os.getenv("DB_USER")
            db_pass = os.getenv("DB_PASSWORD")
            db_name = os.getenv("DB_NAME")
            return db_host, db_user, db_pass, db_name
        

        except KeyError as e:
            raise KeyError("One or more secret environment variables are not set in the '.env' file.") from e


    except FileNotFoundError as e:
        raise FileNotFoundError("Environment file '.env' does not exist.") from e