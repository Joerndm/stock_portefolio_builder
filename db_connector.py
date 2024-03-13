"""Estabilishes a connection to the database"""

import mysql.connector

import fetch_secrets

def db_connector(host, user, password, database):
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
        mydb = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
        )

        return mydb

    except mysql.connector.Error as e:
        raise mysql.connector.Error("Could not connect to the database.") from e
    

if __name__ == "__main__":
    db_host, db_user, db_pass, db_name = fetch_secrets.secret_import()
    print(db_host)
    print(db_user)
    print(db_pass)
    print(db_name)
    

    mydb = db_connector(db_host, db_user, db_pass, db_name)
    mycursor = mydb.cursor()

