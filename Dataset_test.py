"""
Unittest for Functions of Dataset.py
"""
# --- Import
import unittest
import pandas as pd
import sqlalchemy as db
import Dataset as ds
import Exceptions as exc


# --- Test Datasets for Unittests
test_training_dataset_dict = {
    'x': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'y1': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    'y2': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    'y3': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'y4': [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
}
test_ideal_dataset_dict = {
    'x': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'y1': [-1, 6, 9, 16, 19, 26, 25, 36, 39, 46, 49],
    'y2': [1, 2, 4, 6, 8, 10, 12, 85, 45, 18, 20],
    'y3': [2, 5, 2, 8, 4, 6, 6, 7, 8, 9, 10],
    'y4': [0, 3, 4, 9, 12, 1, 18, 58, 24, 27, 30]
}
test_ideal_dataset_error_dict = {
    'x': [-5, -4, -3, -2, -1, 0, 1, 2, 3],
    'y1': [-1, 6, 9, 16, 19, 26, 25, 36, 39],
    'y2': [1, 2, 4, 6, 8, 10, 12, 85, 45],
    'y3': [2, 5, 2, 8, 4, 6, 6, 7, 8],
    'y4': [0, 3, 4, 9, 12, 1, 18, 58, 24]
}
rmse_result_dict = {
    'Train y1': [1.783765, 21.264567, 23.382200, 14.675273],
    'Train y2': [17.388607, 23.126136, 5.900693, 14.535224],
    'Train y3': [23.282846, 26.502144, 2.044949, 18.705857],
    'Train y4': [11.516786, 20.911067, 11.638494, 11.943047]
}

test_training_dataset = pd.DataFrame(test_training_dataset_dict)
test_ideal_dataset = pd.DataFrame(test_ideal_dataset_dict)
test_ideal_dataset_error = pd.DataFrame(test_ideal_dataset_error_dict)
rmse_result = pd.DataFrame(rmse_result_dict, index=[1, 2, 3, 4])
rmse_result.index.name = "Ideal Function"

class UnitTestTraining(unittest.TestCase):
    def setUp(self):
        self.dataset = test_training_dataset
        self.n_rows = len(self.dataset.index)                   # Number of rows
        self.n_cols = len(self.dataset.columns)                 # Number of columns
        self.rmse = pd.DataFrame()
        test_ideal_dataset.n_rows = len(test_ideal_dataset.index)   # Number of rows
        test_ideal_dataset.n_cols = len(test_ideal_dataset.columns) # Number of columns
        test_ideal_dataset.dataset = test_ideal_dataset             # To provide required inputs
        test_ideal_dataset_error.n_rows = len(test_ideal_dataset_error.index)  # Number of rows

    def test_calculate_rmse(self):
        """
        Test RMSE calculation.
        """
        database_file = 'database_test.db'
        engine = db.create_engine(f'sqlite:///{database_file}')
        ds.Training.calculate_rmse(self, test_ideal_dataset, engine)
        pd.testing.assert_frame_equal(self.rmse, rmse_result)

    def test_exception_calculate_rmse(self):
        """
        Test RMSE calculation.
        """
        database_file = 'database_test.db'
        engine = db.create_engine(f'sqlite:///{database_file}')
        with self.assertRaises(exc.RowCountMismatchError):
            ds.Training.calculate_rmse(self, test_ideal_dataset_error, engine)

    # Hallo :)
    # Gerne füge ich weitere Unittests hinzu, allerdings ist dies sehr zeitaufwändig, was ich derzeit nicht ohne weiteres schaffe neben meinem Job.
    # Das Prinzip der Unittests habe ich verstanden und werde diese in Zukunft auch im Beruf anwenden. :)

if __name__ == '__main__':
    unittest.main()