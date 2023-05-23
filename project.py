# Import required libraries
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, ColorBar
from bokeh.transform import linear_cmap
from bokeh.layouts import column
from bokeh.palettes import Viridis256

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
    
    
class DataProcessor:
    """
    A class for selecting the best ideal functions and processing data.

    Attributes:
        db_handler (DatabaseHandler): An instance of the DatabaseHandler class.
        training_path (str): The path to the training dataset.
        ideal_functions_path (str): The path to the ideal functions dataset.
        test_path (str): The path to the test dataset.
        training_data (pandas.DataFrame): The training dataset.
        ideal_functions (pandas.DataFrame): The ideal functions dataset.
        test_data (pandas.DataFrame): The test dataset.
        best_functions (dict): The best functions selected for each training function.
        max_deviations (dict): The maximum deviations for each training function.
        results (pandas.DataFrame): The processed test data results.

    Methods:
        _load_dataset(path, table_name): Load a dataset from a file and save it to the database.
        load_data(): Load the training, ideal functions, and test datasets.
        select_best_functions(): Select the best ideal function for each training function.
        process_test_data(): Process the test data and save the results to the database.
    """

    def __init__(self, db_handler, training_path, ideal_functions_path, test_path):
        """
        Initialize the DataProcessor object.

        Args:
            db_handler (DatabaseHandler): An instance of the DatabaseHandler class.
            training_path (str): The path to the training dataset.
            ideal_functions_path (str): The path to the ideal functions dataset.
            test_path (str): The path to the test dataset.
        """
        self.db_handler = db_handler
        self.train_path = training_path
        self.ideal_path = ideal_functions_path
        self.test_path = test_path
        self.training_data = None
        self.ideal_functions = None
        self.test_data = None
        self.best_functions = None
        self.max_deviations = None
        self.results = None

    def _load_dataset(self, path, table_name):
        """
        Load a dataset from a file and save it to the database.

        Args:
            path (str): The path to the dataset file.
            table_name (str): The name of the table to save the data in.

        Returns:
            pandas.DataFrame: The loaded dataset.
        
        Raises:
            Exception: If there is an error during the dataset loading process.
        """
        try:
            dataset = pd.read_csv(path)
            self.db_handler.load_data(dataset, table_name)
            return dataset
        except Exception as e:
            raise Exception(f"Error occurred while loading the dataset from {path}.") from e


    def load_data(self):
        """
        Load the training, ideal functions, and test datasets.

        Raises:
            Exception: If there is an error during the dataset loading process.
        """
        try:
            # reading and loading the training dataset
            self.training_data = self._load_dataset(self.train_path, 'training_data')

            # reading and loading the ideal functions
            self.ideal_functions = self._load_dataset(self.ideal_path, 'ideal_functions')

            # reading the test data
            try:
                self.test_data = pd.read_csv(self.test_path)
            except Exception as e:
                raise Exception(f"Error occurred while reading the test data from {self.test_path}.") from e

        except Exception as e:
            raise Exception("Error occurred while loading the datasets.") from e


    def select_best_functions(self):
        """
        Selects the best ideal function that minimizes the sum of y-deviations squared (Least-Square) for each training function.

        Returns: A tuple of two dictionaries: 
                - first containing best function for each training function,
                - second containing maximum deviation for each training function
        """
        
        try:
            # Initialize a dictionary to store the best functions for each training function
            best_functions = {}
            # This will hold the maximum deviations for each training function
            max_deviations = {}  

            # For each training function
            for training_func in ['y1', 'y2', 'y3', 'y4']:
                # Initialize a dictionary to store the sum of squared differences for each ideal function
                SSDs = {}

                # For each ideal function
                for ideal_func in self.ideal_functions.columns[1:]:
                    # Calculate the sum of squared differences for the current function
                    SSDs[ideal_func] = ((self.ideal_functions[ideal_func] - self.training_data[training_func]) ** 2).sum()

                # Select the best function with the minimum sum of squared differences
                best_function = min(SSDs, key=SSDs.get)

                # Store the best function and the maximum deviation (sqrt of min_SSD) for the current training function
                best_functions[training_func] = best_function
                max_deviations[training_func] = np.sqrt(SSDs[best_function])

            # Set the best functions and the maximum deviations
            self.best_functions = best_functions
            self.max_deviations = max_deviations
        
        except Exception as e:
            raise BestFunctionsSelectionError(original_exception=e)

    def process_test_data(self):
        """
        Process the test data: for each x-y pair, assign it to one of the selected ideal functions if it meets the criterion.
        Save the results to a new table in the database.

        Raises:
            Exception: If there is an error during the data processing or saving process.
        """
        try:
            # Read the ideal functions from the db
            ideal_functions = self.db_handler.read_table('ideal_functions')

            # Prepare a DataFrame to store the results
            results = pd.DataFrame(columns=['X', 'Y', 'Delta_Y', 'Ideal Function'])

            # For each x-y pair in the test data
            for i, row in self.test_data.iterrows():
                x, y = row['x'], row['y']

                # Check against each of the selected ideal functions
                for training_func, ideal_func in self.best_functions.items():
                    # Calculate the deviation from the ideal function
                    deviation = abs(y - ideal_functions.loc[i, ideal_func])

                    # If the deviation does not exceed the maximum deviation for this function, assign it
                    if deviation <= self.max_deviations[training_func] * np.sqrt(2):
                        results.loc[i] = [x, y, deviation, ideal_func]

                        # Break the loop as we've found a match for this x-y pair
                        break

            # Save the results to a new table in the db
            self.db_handler.load_data(results, 'test_data_results')
            self.results = results
            
        except Exception as e:
            raise TestDataProcessingError(original_exception=e)
   
   
class DataVisualizer:
    """
    A class for visualizing data processed by a DataProcessor.

    Attributes:
        data_processor (DataProcessor): An instance of the DataProcessor class.

    Methods:
        visualize_data(): Visualize the processed data.
    """
    def __init__(self, data_processor):
        """
        Initialize the DataVisualizer object.

        Args:
            data_processor (DataProcessor): An instance of the DataProcessor class.
        """
        self.data_processor = data_processor
     
     
    def visualize_data(self):
        """
        Visualizes the processed data using Bokeh. Five plots are created: Training Data, Test Data, Ideal Functions,
        Best Ideal Functions, and Test Data with Deviations. 

        Raises:
            DataVisualizationError: An error occurred during visualization. The original exception is stored inside the DataVisualizationError.

        """
        try:  
            # Access the processed data
            training_data = self.data_processor.training_data
            test_data = self.data_processor.test_data
            ideal_functions = self.data_processor.ideal_functions
            best_functions = self.data_processor.best_functions
            mapping_deviation = self.data_processor.db_handler.read_table('test_data_results')

            # Plot training data
            p1 = figure(title="Training Data")
            for col in training_data.columns[1:]:
                p1.circle(x=training_data['x'], y=training_data[col], legend_label=f'Training Data {col}', size=5)

            # Plot test data
            p2 = figure(title="Test Data")
            p2.circle(x=test_data['x'], y=test_data['y'], color="red", legend_label='Test Data', size=5)

            # Plot ideal functions
            p3 = figure(title="Ideal Functions")
            for col in ideal_functions.columns[1:]:
                p3.line(x=ideal_functions['x'], y=ideal_functions[col], legend_label=f'Ideal Function {col}')

            # Plot the best ideal functions
            p4 = figure(title="Best Ideal Functions")
            best_four_functions = ideal_functions[['x'] + list(best_functions.values())]
            for func_name, func_label in best_functions.items():
                p4.line(x=best_four_functions['x'], y=best_four_functions[func_label], legend_label=func_label)

            # Plot the test data with deviations
            p5 = figure(title="Test Data with Deviation")
            source = ColumnDataSource(mapping_deviation)
            color_mapper = linear_cmap(field_name='Delta_Y', palette=Viridis256, low=min(mapping_deviation['Delta_Y']), high=max(mapping_deviation['Delta_Y']))
            color_bar = ColorBar(color_mapper=color_mapper['transform'], width=8,  location=(0,0))
            p5.circle(x='X', y='Y', color=color_mapper, source=source, size=5)
            p5.add_layout(color_bar, 'right')

            # Show plots
            show(column(p1, p2, p3, p4, p5))
        
        except Exception as e:
            raise DataVisualizationError(original_exception=e)    
            
    
class ProcessingError(Exception):
    """Base class for other exceptions"""
    def __init__(self, message, original_exception=None):
        self.message = message
        self.original_exception = original_exception
        super().__init__(self.message)


class DataVisualizationError(ProcessingError):
    """Exception raised for errors in the data visualization process."""
    def __init__(self, original_exception=None, 
                 message="Error occurred while visualizing the data."):
        if original_exception:
            message += f" Original error: {original_exception}"
        super().__init__(message, original_exception)


class TestDataProcessingError(ProcessingError):
    """Exception raised for errors in the test data processing process."""
    def __init__(self, original_exception=None, 
                 message="Error occurred while processing the test data."):
        if original_exception:
            message += f" Original error: {original_exception}"
        super().__init__(message, original_exception)


class BestFunctionsSelectionError(ProcessingError):
    """Exception raised for errors in the selection of best functions process."""
    def __init__(self, original_exception=None, 
                 message="Error occurred while selecting best functions."):
        if original_exception:
            message += f" Original error: {original_exception}"
        super().__init__(message, original_exception)
    
        
        
if __name__ == "__main__":   
         
    # Initialize a DatabaseHandler object with a SQLite database.
    db_handler = DatabaseHandler('sqlite:///engine.db')

    # Initialize a DataProcessor object, specifying the database handler and file paths for the training, ideal functions, and test datasets.
    processor = DataProcessor(db_handler, 'dataset/train.csv', 'dataset/ideal.csv', 'dataset/test.csv')

    # Initialize a DataVisualizer object, specifying the data processor.
    visualizer = DataVisualizer(processor)

    # Load data into the DataProcessor.
    processor.load_data()

    # Select the best functions for the data.
    processor.select_best_functions()

    # Print the best functions and maximum deviations.
    print("Best functions: ", processor.best_functions)
    print("Max deviations: ", processor.max_deviations)

    # Process the test data and update the database.
    processor.process_test_data()

    # Print the results of the test data processing.
    print (processor.db_handler.read_table("test_data_results"))

    # Call the method to visualize data.
    visualizer.visualize_data()
