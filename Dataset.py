# --- Import
import pandas as pd
from pathlib import Path


# --- DATASET BASE CLASS ---
class Dataset:
    """
    Class to load a dataset
    ----------
    :param file_path:
        The file path of the .csv file.
    :param table_name:
        The name of the sql table.
        This is also the name of the .csv file.
    :param engine:
        The SQL engine.
    :dataset:
        The loaded dataset.
    """
    def __init__(self, file_path, table_name, engine):
        self.file_path = file_path
        self.table_name = table_name
        self.dataset = self.get_dataframe(table_name, engine, file_path)

    def get_dataframe(self, table_name, engine, file_path):
        """
        Get a Pandas dataframe from the corresponding SQL table.
        If the SQL table does not exist, create one from the .csv file.
        :param table_name:
        The name of the sql table.
        This is also the name of the .csv file.
        :param engine:
        The SQL engine.
        :param file_path:
        The file path of the .csv file.
        :return:
        Returns a pandas dataframe of the dataset.
        """
        try:
            if not engine.has_table(table_name):                                # If table does not exist, create one
                print(f'Table {table_name} does not exists, creating one.')
                csv_dataset = self.load_csv(file_path)                          # Load dataset from csv file
                csv_dataset.to_sql(table_name, engine, index=False)

            dataframe = pd.read_sql_table(table_name, con=engine)

        except Exception as ex:
            print("The following error occurred while loading the dataframe: \n", ex)

        else:
            print(f'Imported data frame from SQL table {table_name}:')
            print(dataframe)
            return dataframe

    @staticmethod
    def load_csv(file_path):
        try:
            csv_dataset = pd.read_csv(file_path)

            for column in csv_dataset:
                if csv_dataset[column].dtype != float:
                    csv_dataset.drop([column], axis=1, inplace=True)        # Drop all columns that contain non-float values

            if csv_dataset.isnull().any().any():
                print("Data file contains empty cells")

            if "x" not in csv_dataset.columns:
                print("X variable not found in data file")

            if len(csv_dataset.columns) < 2:
                print("No Y values found in data file")

        except Exception as ex:
            print("The following error occurred while loading the csv file: \n", ex)

        else:
            print(f"Loaded the csv file successfully from {file_path}:")
            print(csv_dataset)
            return csv_dataset


# --- TRAINING DATASET CLASS ---
class Training(Dataset):
    """
    Child class of "Dataset" for the training dataset.
    """
    def __init__(self, file_path, engine):
        self.table_name = Path(file_path).stem.lower()                  # Get the file name from the file path (also name of table) in lower case
        Dataset.__init__(self, file_path, self.table_name, engine)      # Call init of base class
