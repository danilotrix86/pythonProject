
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, ColorBar
from bokeh.transform import linear_cmap
from bokeh.layouts import column
from bokeh.palettes import RdBu
from exceptions import DataVisualizationError

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
            p1 = figure(title="Training Data" , y_range=(-50, 50))
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
            p4 = figure(title="Best Ideal Functions", y_range=(-100, 100))
            best_four_functions = ideal_functions[['x'] + list(best_functions.values())]
            for func_name, func_label in best_functions.items():
                p4.line(x=best_four_functions['x'], y=best_four_functions[func_label], legend_label=func_label)

            # Plot the test data with deviations
          
            
            p5 = figure(title="Test Data with Deviation")
            source = ColumnDataSource(mapping_deviation)
            color_mapper = linear_cmap(field_name='Delta_Y', palette=RdBu[9], low=min(mapping_deviation['Delta_Y']), high=max(mapping_deviation['Delta_Y']))
            color_bar = ColorBar(color_mapper=color_mapper['transform'], width=8, location=(0, 0))
            p5.circle(x='X', y='Y', color=color_mapper, source=source, size=5)
            p5.add_layout(color_bar, 'right')

            # Show plots
            show(column(p1, p2, p3, p4, p5))
        
        except Exception as e:
            raise DataVisualizationError(original_exception=e)   