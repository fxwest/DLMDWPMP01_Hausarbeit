# --- Import
import logging

import sqlalchemy as db
import Dataset as ds
import logging as log

# --- SQL Parameter
database_name = 'dlmdwpmp01_hausarbeit'
database_file = 'database.db'

# --- File Parameter
training_file_path = r'C:\Users\felix\OneDrive\Studium\3_Master\1_IU\2_Module\0_Python\3_Hausarbeit\DLMDWPMP01_Hausarbeit\Datasets\ExampleDatasets\trainExample.csv'
training_plot_file = r'Figures\trainExamplePlot.html'
ideal_file_path = r'C:\Users\felix\OneDrive\Studium\3_Master\1_IU\2_Module\0_Python\3_Hausarbeit\DLMDWPMP01_Hausarbeit\Datasets\ExampleDatasets\idealExample.csv'
ideal_plot_file = r'Figures\idealExamplePlot.html'
test_file_path = r'C:\Users\felix\OneDrive\Studium\3_Master\1_IU\2_Module\0_Python\3_Hausarbeit\DLMDWPMP01_Hausarbeit\Datasets\ExampleDatasets\testExample.csv'
test_plot_file = r'Figures\testExamplePlot.html'


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
    # try:
    engine = get_connection()  # Get the engine for the database
    training_dataset = ds.Training(training_file_path, training_plot_file, engine)
    ideal_dataset = ds.Ideal(ideal_file_path, ideal_plot_file, engine)
    test_dataset = ds.Test(test_file_path, test_plot_file, engine)
    training_dataset.calculate_rmse(ideal_dataset)
    training_dataset.calculate_rsquare(ideal_dataset)

# except Exception as ex:
#   print("The following error occurred during program execution: \n", ex)


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
