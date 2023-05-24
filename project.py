
from data_handler import DatabaseHandler
from data_loader import DataLoader
from data_processor import DataProcessor
from data_visualizer import DataVisualizer

if __name__ == '__main__':        
    db_handler = DatabaseHandler('sqlite:///engine.db')
    data_loader = DataLoader(db_handler)
    processor = DataProcessor(db_handler, data_loader)
    visualizer = DataVisualizer(processor)

    processor.load_data('dataset/train.csv', 'dataset/ideal.csv', 'dataset/test.csv')
    processor.select_best_functions()
    processor.process_test_data()
    visualizer.visualize_data()
