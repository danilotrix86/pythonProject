# Import required libraries
import pandas as pd
from sqlalchemy import create_engine


class DatabaseHandler:
    """
    A class for handling database operations.

    Attributes:
        db_url (str): The URL of the database.

    Methods:
        load_data(data, table_name): Load data into a table in the database.
        read_table(table_name): Read data from a table in the database.
    """

    def __init__(self, db_url):
        """
        Initialize the DatabaseHandler object.

        Args:
            db_url (str): The URL of the database.
        """
        self.engine = create_engine(db_url)
    
    def load_data(self, data, table_name):
        """
        Load data into a table in the database.

        Args:
            data (pandas.DataFrame): The data to be loaded.
            table_name (str): The name of the table.

        Raises:
            ValueError: If the data is not a DataFrame.
            Exception: If there is an error during the data loading process.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a DataFrame.")

        try:
            data.to_sql(table_name, self.engine, if_exists='replace', index=False)
        except Exception as e:
            raise Exception("Error occurred while loading data into the table.") from e
    
    def read_table(self, table_name):
        """
        Read data from a table in the database.

        Args:
            table_name (str): The name of the table.

        Returns:
            pandas.DataFrame: The data from the table.

        Raises:
            Exception: If there is an error during the data reading process.
        """
        try:
            return pd.read_sql_query(f'SELECT * FROM {table_name}', self.engine)
        except Exception as e:
            raise Exception("Error occurred while reading data from the table.") from e