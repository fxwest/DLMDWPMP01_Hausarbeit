# --- Import
import pathlib
import bokeh.palettes
import pandas as pd
import logging as log
import numpy as np
import itertools
from pathlib import Path
from bokeh.plotting import figure, output_file, show
from Exceptions import RowCountMismatchError


# --- DATASET BASE CLASS ---
class Dataset:
    """
    Class to load a dataset
    :param file_path:
        The file path of the .csv file.
    :param plot_file:
        The path for the Bokeh output plot file.
    :param engine:
        The SQL engine.
    :dataset:
        The loaded dataset.
    """
    def __init__(self, file_path, plot_file, engine):
        self.file_path = file_path
        self.plot_file = plot_file
        self.table_name = Path(file_path).stem.lower()                              # Get the file name from the file path (also name of table) in lower case
        self.dataset = self.get_dataframe(self.table_name, engine, file_path)       # Get the dataframe from the dataset via SQL
        self.n_rows = len(self.dataset.index)                                       # Number of rows
        self.n_cols = len(self.dataset.columns)                                     # Number of columns
        self.plot_dataset(self.dataset, self.dataset_name, plot_file, self.n_cols)  # Plot the loaded dataset

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

    def load_csv(self, file_path):
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

    def plot_dataset(self, dataset, dataset_name, plot_file, n_cols):
        """
        Plots the dataset via Bokeh.
        :param dataset:
            The dataset to be plotted.
        :param dataset_name:
            The name of the loaded dataset.
        :param plot_file:
            The path for the Bokeh output plot file.
        :param n_cols:
            Number of columns of the current dataset.
        """
        try:
            folder = pathlib.PurePath(plot_file).parent.name                    # Get folder from path
            if not Path(folder).is_dir():                                       # If folder does not exist
                log.info(f'Creating Folder {folder}')
                Path(folder).mkdir(parents=True)                                # Create folder

            output_file(plot_file)                                              # Define output file
            colors = itertools.cycle(bokeh.palettes.Category20_20)              # Get endless color iterator
            plot = figure(width=1000, height=800, title=dataset_name,           # Create figures
                          x_axis_label="X Axis", y_axis_label="Y Axis")
            for y_values in dataset.iloc[:, 1:n_cols]:
                plot.circle(dataset['x'], dataset[y_values], size=7, alpha=0.5, legend_label=y_values, color=next(colors))
            show(plot)

        except Exception as ex:
            log.error("The following error occurred while plotting the dataset: \n", ex)

        else:
            log.info(f'Created Bokeh plot for {dataset_name}')


# --- TRAINING DATASET CLASS ---
class Training(Dataset):
    """
    Child class of "Dataset" for the training dataset.
    """
    def __init__(self, training_file_path, training_plot_file,  engine):
        self.dataset_name = "Training Dataset"
        self.rmse = pd.DataFrame()
        self.r2 = pd.DataFrame()
        self.best_fit = pd.DataFrame()
        Dataset.__init__(self, training_file_path, training_plot_file, engine)  # Call init of base class

    def calculate_rmse(self, ideal_dataset, engine):
        """
        Calculate the Root Mean Squared Error between the Ideal and Training Dataset.
        Results are stored in a Pandas Dataframe called rmse.
        :param ideal_dataset:
            The ideal dataset class.
        :param engine:
            The SQL engine.
        """
        try:
            if self.n_rows != ideal_dataset.n_rows:
                raise RowCountMismatchError

            for col_train in self.dataset.iloc[:, 1:self.n_cols]:                           # Iterate over y-dimension (columns) of the train dataset
                y_train = self.dataset[col_train]                                           # Select current y value from train dataset
                rmse_list = []
                for col_ideal in ideal_dataset.dataset.iloc[:, 1:ideal_dataset.n_cols]:     # Iterate over y-dimension (columns) of the ideal dataset
                    y_ideal = ideal_dataset.dataset[col_ideal]                              # Select current y value from ideal dataset
                    rmse = 0
                    for i in range(self.n_rows):                                            # Iterate over x-dimension (rows)
                        rmse += (y_ideal[i] - y_train[i]) ** 2                              # Squared error between ideal and train y value
                    rmse = np.sqrt(rmse/self.n_rows)                                        # Root Mean Squares Error
                    rmse_list.append(rmse)                                                  # Add rmse value of current ideal set to rmse list
                self.rmse[f"Train {col_train}"] = rmse_list                                 # Add rmse list of current train set to rmse dataframe
            self.rmse.index.name = "Ideal Function"                                         # Change name of dataframe index
            self.rmse.index += 1                                                            # Index start from 1
            self.rmse.to_sql("rmse_train_ideal", engine, index=False, if_exists='replace')  # Save rmse as table in SQL-Database

        except RowCountMismatchError:                                                       # TODO Unit-Test
            log.error(RowCountMismatchError().error_msg)

        except Exception as ex:
            log.error("The following error occurred while calculating the RMSE: \n", ex)

        else:
            log.info("Calculated RMSE between Training and Ideal Dataset and saved to SQL-Database")
            print(self.rmse)

    def calculate_rsquare(self, ideal_dataset, engine):
        """
        Calculate the R-Squared Value between the Ideal and Training Dataset.
        Results are stored in a Pandas Dataframe called r2.
        This value shows how close the ideal data is to the training data.
        :param ideal_dataset:
            The ideal dataset class.
        :param engine:
            The SQL engine.
        """
        try:
            if self.n_rows != ideal_dataset.n_rows:
                raise RowCountMismatchError

            for col_train in self.dataset.iloc[:, 1:self.n_cols]:                           # Iterate over y-dimension (columns) of the train dataset
                y_train = self.dataset[col_train]                                           # Select current y value from train dataset
                r2_list = []
                for col_ideal in ideal_dataset.dataset.iloc[:, 1:ideal_dataset.n_cols]:     # Iterate over y-dimension (columns) of the ideal dataset
                    y_ideal = ideal_dataset.dataset[col_ideal]                              # Select current y value from ideal dataset
                    mean_y_ideal = np.mean(y_ideal)                                         # Y-Mean of the current ideal function
                    ss_tot = 0                                                              # Total sum of squares
                    ss_res = 0                                                              # Total sum of squares of residuals
                    for i in range(self.n_rows):                                            # Iterate over x-dimension (rows)
                        ss_tot += (y_ideal[i] - mean_y_ideal) ** 2
                        ss_res += (y_ideal[i] - y_train[i]) ** 2
                    r2 = 1 - (ss_res/ss_tot)                                                # R-squared value
                    r2_list.append(r2)                                                      # Add r2 value of current ideal set to r2 list
                self.r2[f"Train {col_train}"] = r2_list                                     # Add r2 list of current train set to r2 dataframe
            self.r2.index.name = "Ideal Function"                                           # Change name of dataframe index
            self.r2.index += 1                                                              # Index start from 1
            self.r2.to_sql("r2_train_ideal", engine, index=False, if_exists='replace')      # Save r2 as table in SQL-Database

        except RowCountMismatchError:                                                       # TODO Unit-Test
            log.error(RowCountMismatchError().error_msg)

        except Exception as ex:
            log.error("The following error occurred while calculating the R-Squared Value: \n", ex)

        else:
            log.info("Calculated R2 between Training and Ideal Dataset and saved to SQL-Database")
            print(self.r2)

    def select_best_fit(self, ideal_dataset, engine, plot_file):
        """
        Find the best fitting function from the Ideal Dataset for each Training Dataset.
        The best fitting function is the function with the lowest R-Squared Value.
        :param engine:
            The SQL engine.
        """
        try:
            best_fit_list = []
            for col_train in self.r2.iloc[:, 0:len(self.r2.columns)]:                                                                       # Iterate over yTrain-dimension (columns) of the r2 results
                array = self.r2[col_train].to_numpy()                                                                                       # Transform dataframe to numpy array
                min_r2_idx = np.argmin(np.abs(array))+1                                                                                     # Get idx of min value (idx starts from 1)
                best_fit_list.append([col_train, min_r2_idx, self.rmse.loc[min_r2_idx, col_train], self.r2.loc[min_r2_idx, col_train]])     # Save best fit to list
            self.best_fit = pd.DataFrame(best_fit_list, columns=["Train Dataset", "Idx Ideal Function", "RMSE", "R2"])                      # Save best fits to Pandas dataframe
            self.best_fit.to_sql("best_fit_train_ideal", engine, index=False, if_exists='replace')                                          # Save best fits as table in SQL-Database

            # Plot best fitting ideal functions
            output_file(plot_file)                                              # Define output file
            colors = itertools.cycle(bokeh.palettes.Category20_20)  # Get endless color iterator
            plot = figure(width=1000, height=800, title="Best fitting Ideal Functions (Train vs. Ideal)",
                          x_axis_label="X Axis", y_axis_label="Y Axis")
            for best_fit in best_fit_list:
                plot.circle(ideal_dataset.dataset['x'], ideal_dataset.dataset[f"y{best_fit[1]}"], size=7, alpha=0.5, legend_label=f"y{best_fit[1]}", color=next(colors))
            show(plot)

        except Exception as ex:
            log.error("The following error occurred while finding the best fitting functions: \n", ex)

        else:
            log.info("Selected and plotted the best fitting functions and saved to SQL-Database")
            print(self.best_fit)


# --- IDEAL DATASET CLASS ---
class Ideal(Dataset):
    """
    Child class of "Dataset" for the ideal dataset.
    """
    def __init__(self, ideal_file_path, ideal_plot_file,  engine):
        self.dataset_name = "Ideal Dataset"
        Dataset.__init__(self, ideal_file_path, ideal_plot_file, engine)        # Call init of base class


# --- TEST DATASET CLASS ---
class Test(Dataset):
    """
    Child class of "Dataset" for the test dataset.
    """
    def __init__(self, test_file_path, test_plot_file,  engine):
        self.dataset_name = "Test Dataset"
        Dataset.__init__(self, test_file_path, test_plot_file, engine)          # Call init of base class