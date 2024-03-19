"""Estabilishes a connection to the database"""
from sqlalchemy import create_engine
import mysql.connector as mysql

import fetch_secrets

def mysql_connector(mysql_host, mysql_user, mysql_password, mysql_database_name):
    """
    Estabilishes a connection to the database.

    Args:
    host (str): The host of the database

    user (str): The user of the database

    password (str): The password of the database

    database (str): The name of the database

    Returns:
    mysql.connector.connection.MySQLConnection: The connection to the database
    """
    try:
        mysql_con = mysql.connect(
            host = mysql_host,
            user = mysql_user,
            password = mysql_password,
            database = mysql_database_name
        )
        return mysql_con

    except mysql.Error as e:
        raise mysql.Error("Could not connect to the database.") from e

def pandas_mysql_connector(mysql_host, mysql_user, mysql_password, mysql_database_name):
    """
    Estabilishes a connection to the database.

    Args:
    host (str): The host of the database

    user (str): The user of the database

    password (str): The password of the database

    database (str): The name of the database

    Returns:
    sqlalchemy.engine.base.Engine: The connection to the database
    """
    try:
        host = mysql_host
        user = mysql_user
        password = mysql_password
        database = mysql_database_name
        mysql_con = create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}/{database}")
        return mysql_con

    except mysql.Error as e:
        raise mysql.Error("Could not connect to the database.") from e

if __name__ == "__main__":
    db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
    print(type(db_host), type(db_user), type(db_pass), type(db_name))
    db_con = pandas_mysql_connector(db_host, db_user, db_pass, db_name)
    print(db_con)
