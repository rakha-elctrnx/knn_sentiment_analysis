import mysql.connector
import pandas as pd
import logging
from config.settings import Settings

class DatabaseManager:
    def __init__(self):
        self.config = {
            'host': Settings.DB_HOST,
            'user': Settings.DB_USER,
            'password': Settings.DB_PASSWORD,
            'database': Settings.DB_NAME,
            'charset': 'utf8mb4'
        }
        self.logger = logging.getLogger(__name__)

    def get_connection(self):
        """Manually manage connection without using context manager"""
        try:
            connection = mysql.connector.connect(**self.config)
            return connection
        except mysql.connector.Error as err:
            self.logger.error(f"Database connection error: {err}")
            raise

    def execute_query(self, query: str, params: tuple = None):
        """Execute a SQL query with provided parameters."""
        connection = self.get_connection()
        try:
            cursor = connection.cursor()
            cursor.execute(query, params or ())
            connection.commit()
        except mysql.connector.Error as err:
            self.logger.error(f"Query execution error: {err}")
            connection.rollback()  # Rollback in case of error
        finally:
            cursor.close()  # Ensure the cursor is closed
            connection.close()  # Ensure the connection is closed

    def fetch_data(self, query: str, params: tuple = None) -> pd.DataFrame:
        """Fetch data from the database and return it as a DataFrame."""
        connection = self.get_connection()
        try:
            return pd.read_sql(query, connection, params=params)
        finally:
            connection.close()  # Ensure the connection is closed after fetching
