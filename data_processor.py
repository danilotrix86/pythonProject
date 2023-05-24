# Import required libraries
import pandas as pd
import numpy as np
from exceptions import BestFunctionsSelectionError, TestDataProcessingError

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

            # Set the best functions and the maximum deviations
            self.best_functions = best_functions
           
        
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
            # Prepare a list to store the results
            results = []

            # For each x-y pair in the test data
            for i, row in self.test_data.iterrows():
                x, y = row['x'], row['y']

                # Identifies the row number within the df ideal_data for which x of ideal data equals the x value of the test structure.
                index = self.ideal_functions.index[(self.ideal_functions['x'] == x)].tolist()

                # For each of the selected ideal functions
                for training_func, ideal_func in self.best_functions.items():
                    # Initialize a list to store deviations between training and ideal data
                    devTrainIdeal = []

                    # Here it is determined the maximum deviation between train_data and associated ideal_data
                    for i in range(len(self.training_data[training_func].tolist())):
                        devTrainIdeal.append(
                            abs(self.training_data[training_func].tolist()[i] - self.ideal_functions[ideal_func].tolist()[i]))

                    maxDevTrainIdeal = max(devTrainIdeal)

                    # Calculate the deviation between the Y_test value and the Y_ideal value with the same x values as Y_test
                    maxDevTestIdeal = abs(y - self.ideal_functions.loc[index, ideal_func].values[0])

                    if ((maxDevTestIdeal - maxDevTrainIdeal) <= np.sqrt(2)):
                        results.append((x, y, maxDevTestIdeal, ideal_func))

            # Convert results list to DataFrame
            results_df = pd.DataFrame(results, columns=['X', 'Y', 'Delta_Y', 'Ideal Function'])

            # Save the results to a new table in the db
            self.db_handler.load_data(results_df, 'test_data_results')
            self.results = results_df
            
                
        except Exception as e:
            raise TestDataProcessingError(original_exception=e)