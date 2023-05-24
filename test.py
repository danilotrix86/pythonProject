import unittest
import pandas as pd
from sqlalchemy import create_engine
from data_handler import DatabaseHandler
import pandas as pd

class TestDatabaseHandler(unittest.TestCase):
    def setUp(self):
        # Create a DatabaseHandler instance with a sqlite database for testing
        self.db_handler = DatabaseHandler('sqlite:///test_engine.db')
        # Create a sample DataFrame for testing
        self.data = pd.DataFrame({
            'column1': ['data1', 'data2', 'data3'],
            'column2': ['data4', 'data5', 'data6']
        })
        self.table_name = 'test_table'

    def test_load_data(self):
        # Test with correct input
        try:
            self.db_handler.load_data(self.data, self.table_name)
            loaded_data = self.db_handler.read_table(self.table_name)
            pd.testing.assert_frame_equal(loaded_data, self.data)
        except Exception as e:
            self.fail(f"test_load_data failed with error: {e}")

    def test_read_table(self):
        # Test with an existing table
        try:
            self.db_handler.load_data(self.data, self.table_name)
            loaded_data = self.db_handler.read_table(self.table_name)
            pd.testing.assert_frame_equal(loaded_data, self.data)
        except Exception as e:
            self.fail(f"test_read_table failed with error: {e}")

        # Test with a non-existing table
        non_existing_table = 'non_existing_table'
        with self.assertRaises(Exception):
            self.db_handler.read_table(non_existing_table)

    def tearDown(self):
        # Delete the test database after testing
        # In this case, we just delete the table as we're using sqlite in memory
        engine = create_engine('sqlite:///test_engine.db')
        connection = engine.raw_connection()
        cursor = connection.cursor()
        command = "DROP TABLE IF EXISTS {};".format(self.table_name)
        cursor.execute(command)
        connection.commit()
        cursor.close()


if __name__ == '__main__':
    unittest.main()
