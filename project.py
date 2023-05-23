# Import required libraries
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, ColorBar
from bokeh.transform import linear_cmap
from bokeh.layouts import column
from bokeh.palettes import Viridis256


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
    This class is responsible for processing data and selecting the best functions.
    """

    def __init__(self, db_handler, data_loader):
        """
        Initialize the DataProcessor with a DBHandler and DataLoader instances.

        Args:
            db_handler (DBHandler): An instance of the DBHandler class to interact with the database.
            data_loader (DataLoader): An instance of the DataLoader class to load data.
        """
        self.db_handler = db_handler
        self.data_loader = data_loader
        self.training_data = None
        self.ideal_functions = None
        self.test_data = None

    def load_data(self, train_path, ideal_path, test_path):
        """
        Load the training, ideal functions, and test datasets using the DataLoader instance.

        Args:
            train_path (str): The path to the training dataset file.
            ideal_path (str): The path to the ideal functions file.
            test_path (str): The path to the test dataset file.
        """
        self.training_data = self.data_loader.load_csv_to_dataframe(train_path, 'training_data')
        self.ideal_functions = self.data_loader.load_csv_to_dataframe(ideal_path, 'ideal_functions')
        self.test_data = self.data_loader.load_csv_to_dataframe(test_path)

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
    
        
        
        
# Initialize the database handler with the connection string
db_handler = DatabaseHandler('sqlite:///engine.db')

# Initialize the data loader with the database handler
data_loader = DataLoader(db_handler)

# Initialize the data processor with the database handler and data loader
processor = DataProcessor(db_handler, data_loader)

# Initialize the data visualizer with the database handler
visualizer = DataVisualizer(db_handler)

"""
The load_data method of the DataProcessor class is responsible for loading 
the training, ideal functions, and test datasets from given file paths.
"""
processor.load_data('dataset/train.csv', 'dataset/ideal.csv', 'dataset/test.csv')

"""
The select_best_functions method of the DataProcessor class is responsible for 
finding the best function (with least deviation from the training data) 
for each type of data in the training dataset.
"""
processor.select_best_functions()

# Uncomment the following lines to print the best functions and max deviations
# print("Best functions: ", processor.best_functions)
# print("Max deviations: ", processor.max_deviations)

"""
The process_test_data method of the DataProcessor class is responsible for 
processing the test data using the best functions selected previously.
"""
processor.process_test_data()

# Print the processed test data results from the database
print(processor.db_handler.read_table("test_data_results"))

"""
The visualize_data method of the DataVisualizer class is responsible for 
visualizing the processed data, best functions, and maximum deviations.
"""
# Call the method to visualize data
visualizer = DataVisualizer(processor)
visualizer.visualize_data()