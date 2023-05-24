# Import required libraries
import pandas as pd

class DataLoader:
    """
    This class is responsible for loading data from CSV files and 
    storing it into a database using a provided DBHandler instance.
    """

    def __init__(self, db_handler):
        """
        Initialize DataLoader with a DBHandler instance.
        
        Args:
            db_handler (DBHandler): An instance of the DBHandler class.
        """
        self.db_handler = db_handler

    def load_csv_to_dataframe(self, path, table_name=None):
        """
        Load a CSV file into a pandas DataFrame. Optionally save the DataFrame into the database.

        Args:
            path (str): The path to the CSV file.
            table_name (str, optional): The name of the table to save the data in. 
                If None, the data is not saved to the database. Defaults to None.

        Returns:
            pandas.DataFrame: The loaded dataset.

        Raises:
            Exception: If there is an error during the dataset loading process.
        """
        try:
            dataset = pd.read_csv(path)

            if table_name is not None:
                self.db_handler.load_data(dataset, table_name)

            return dataset
        except Exception as e:
            raise Exception(f"Error occurred while loading the dataset from {path}.") from e