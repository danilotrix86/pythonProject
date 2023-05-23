# Import required libraries
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import matplotlib.pyplot as plt

class DatabaseHandler:
    def __init__(self, db_url):
        self.engine = create_engine(db_url)
    
    def load_data(self, data, table_name):
        data.to_sql(table_name, self.engine, if_exists='replace', index=False)
    
    def read_table(self, table_name):
        return pd.read_sql_query(f'SELECT * FROM {table_name}', self.engine)
    
    
class DataProcessor:
    def __init__(self, db_handler, training_path, ideal_functions_path, test_path):
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
        dataset = pd.read_csv(path)
        self.db_handler.load_data(dataset, table_name)
        return dataset

    def load_data(self):
        # reading and loading the training dataset
        self.training_data = self._load_dataset(self.train_path, 'training_data')

        # reading and loading the ideal functions
        self.ideal_functions = self._load_dataset(self.ideal_path, 'ideal_functions')

    def select_best_functions(self):
        """
        Selects the best ideal function that minimizes the sum of y-deviations squared (Least-Square) for each training function.
        """
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



    def process_test_data(self):
        """
        Process the test data: for each x-y pair, assign it to one of the selected ideal functions if it meets the criterion.
        Save the results to a new table in the database.
        """
        # Read the test data and the ideal functions from the db
        test_data = pd.read_csv(self.test_path)
        ideal_functions = self.db_handler.read_table('ideal_functions')

        # Prepare a DataFrame to store the results
        results = pd.DataFrame(columns=['X', 'Y', 'Delta_Y', 'Ideal Function'])

        # For each x-y pair in the test data
        for i, row in test_data.iterrows():
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

    def visualize_data(self):
        # The function remains largely unchanged
        # ...
        a=0
   
class DataVisualizer:
    def __init__(self, db_handler):
        self.db_handler = db_handler

    def visualize_data(self, training_data, test_data, ideal_functions, best_functions, mapping_deviation):
        # The existing function code goes here
        # ...
        a=0   
     
        
db_handler = DatabaseHandler('sqlite:///engine.db')
processor = DataProcessor(db_handler, 'dataset/train.csv', 'dataset/ideal.csv', 'dataset/test.csv')
visualizer = DataVisualizer(db_handler)

processor.load_data()

processor.select_best_functions()

print("Best functions: ", processor.best_functions)
print("Max deviations: ", processor.max_deviations)

processor.process_test_data()
print (processor.db_handler.read_table("test_data_results"))
