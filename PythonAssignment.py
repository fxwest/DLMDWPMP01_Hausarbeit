"""
---------------- ASSIGNMENT -----------------
--- DLMDWPMP01 – Programmieren mit Python ---
------------ Felix A. Westphal --------------
Find the four best fits from the ideal dataset, containing 50 functions, using four training datasets.
Validate the fits with a test dataset.
Plot the datasets and the final result.
All relevant dataframes are saved in a SQL-Database.
"""
# --- Import
import logging

import sqlalchemy as db
import Dataset as ds
import logging as log

# --- SQL Parameter
database_name = 'dlmdwpmp01_hausarbeit'
database_file = 'database.db'

# --- File Parameter
training_file_path = r'Datasets\ExampleDatasets\trainExample.csv'
ideal_file_path = r'Datasets\ExampleDatasets\idealExample.csv'
test_file_path = r'Datasets\ExampleDatasets\testExample.csv'
training_plot_file = r'Figures\trainExamplePlot.html'
ideal_plot_file = r'Figures\idealExamplePlot.html'
test_plot_file = r'Figures\testExamplePlot.html'
best_fit_plot_file = r'Figures\bestFitExamplePlot.html'
matching_functions_plot_file = r'Figures\matchExamplePlot.html'
result_plot_file = r'Figures\resultExamplePlot.html'


# --- SQL GET ENGINE ---
def get_connection():
    """
    Connect to the local MySQL Database.
    :return: engine object
    """
    con_str = f'sqlite:///{database_file}'
    try:
        engine = db.create_engine(url=con_str)

    except Exception as ex:
        log.error("SQL connection could not be established due to the following error: \n", ex)

    else:
        log.info(f"Connection to the {database_name} SQL database created successfully.")
        return engine


# --- MAIN FUNCTION ---
def main():
    """
    Main function to control the whole program.
    """
    try:
        engine = get_connection()  # Get the engine for the database
        training_dataset = ds.Training(training_file_path, training_plot_file, engine)
        ideal_dataset = ds.Ideal(ideal_file_path, ideal_plot_file, engine)
        test_dataset = ds.Test(test_file_path, test_plot_file, engine)
        training_dataset.calculate_rmse(ideal_dataset, engine)
        training_dataset.calculate_rsquare(ideal_dataset, engine)
        training_dataset.calculate_least_square(ideal_dataset, engine)
        training_dataset.select_best_fit(ideal_dataset, engine, best_fit_plot_file)
        test_dataset.matching_functions(training_dataset.best_fit, ideal_dataset, engine)
        result = test_dataset.get_result(training_dataset.best_fit, engine)
        test_dataset.plot_matching_functions(matching_functions_plot_file, training_dataset.best_fit, ideal_dataset)
        test_dataset.plot_result(result_plot_file, training_dataset.best_fit, ideal_dataset)

    except Exception as ex:
        log.error("The following error occurred during program execution: \n", ex)

    else:
        log.info("The Python Assignment program finished without errors!")


# --- Call main function
if __name__ == '__main__':
    # Save logs into file and output via console
    log.basicConfig(level=log.NOTSET, filename="logfile.txt", format="%(asctime)s - %(message)s", filemode="w")
    stderrLogger = logging.StreamHandler()
    stderrLogger.setFormatter(log.Formatter(log.BASIC_FORMAT))
    log.getLogger().addHandler(stderrLogger)
    log.info("Started Python Assignment...")
    # Call main function
    main()
