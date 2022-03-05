# --- Import
import pathlib

import pandas as pd
import logging as log
from pathlib import Path
from bokeh.plotting import figure, output_file, show


# --- DATASET BASE CLASS ---
class Dataset:
    """
    Class to load a dataset
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
            Returns a Pandas dataframe of the dataset.
        """
        try:
            if not engine.has_table(table_name):                                # If table does not exist, create one
                log.info(f'Table {table_name} does not exists, creating one.')
                csv_dataset = self.load_csv(file_path)                          # Load dataset from csv file
                csv_dataset.to_sql(table_name, engine, index=False)

            dataframe = pd.read_sql_table(table_name, con=engine)

        except Exception as ex:
            log.error("The following error occurred while loading the dataframe: \n", ex)

        else:
            log.info(f'Imported data frame from SQL table {table_name}')
            print(dataframe)
            return dataframe

    @staticmethod
    def load_csv(file_path):
        """
        Loads the dataset from a .csv file.
        :param file_path:
            The file path to the .csv file.
        :return:
            Returns a Pandas dataframe of the .csv file dataset.
        """
        try:
            csv_dataset = pd.read_csv(file_path)

            for column in csv_dataset:
                if csv_dataset[column].dtype != float:
                    csv_dataset.drop([column], axis=1, inplace=True)        # Drop all columns that contain non-float values

            if csv_dataset.isnull().any().any():
                log.warning("Data file contains empty cells")

            if "x" not in csv_dataset.columns:
                log.error("X variable not found in data file")

            if len(csv_dataset.columns) < 2:
                log.error("No Y values found in data file")

        except Exception as ex:
            log.error("The following error occurred while loading the csv file: \n", ex)

        else:
            log.info(f"Loaded the csv file successfully from {file_path}")
            print(csv_dataset)
            return csv_dataset


# --- TRAINING DATASET CLASS ---
class Training(Dataset):
    """
    Child class of "Dataset" for the training dataset.
    """
    def __init__(self, file_path, engine, training_plot_file):
        self.table_name = Path(file_path).stem.lower()                  # Get the file name from the file path (also name of table) in lower case
        Dataset.__init__(self, file_path, self.table_name, engine)      # Call init of base class
        self.plot_dataset(self.dataset, training_plot_file)

    def plot_dataset(self, dataset, training_plot_file):
        """
        Plots the training dataset via Bokeh.
        :param dataset:
            The dataset to be plotted.
        :param training_plot_file:
            The path for the Bokeh output plot file.
        """
        folder = pathlib.PurePath(training_plot_file).parent.name           # Get folder from path
        if not Path(folder).is_dir():                                       # If folder does not exist
            log.info(f'Creating Folder {folder}')
            Path(folder).mkdir(parents=True)                                # Create folder

        output_file(training_plot_file)                                     # Define output file
        plot = figure(width=1000, height=800, title="Training Dataset",     # Create figures
                      x_axis_label="X Axis", y_axis_label="Y Axis")
        plot.circle(dataset['x'], dataset['y1'], size=7, color="firebrick",
                    alpha=0.5, legend_label="y2")
        plot.circle(dataset['x'], dataset['y2'], size=7, color="blue",
                    alpha=0.5, legend_label="y1")
        plot.circle(dataset['x'], dataset['y3'], size=7, color="green",
                    alpha=0.5, legend_label="y3")
        plot.circle(dataset['x'], dataset['y4'], size=7, color="cyan",
                    alpha=0.5, legend_label="y4")
        show(plot)
