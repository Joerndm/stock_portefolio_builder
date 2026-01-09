"""
Module for fetching secret environment variables.

This module provides functionality to load and retrieve database configuration
secrets from a .env file. It uses the python-dotenv library to load environment
variables and returns them for use in database connections.

Functions:
    secret_import: Loads and returns database configuration from environment variables.
"""
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
