# --- Import
import sqlalchemy as db
import Dataset as ds

# --- SQL Parameter
user = 'root'
password = 'Python12345'
host = '127.0.0.1'
port = 3306
database = 'dlmdwpmp01_hausarbeit'


# --- SQL GET ENGINE ---
def get_connection():
    """
    Connect to the local MySQL Database.
    :return: engine object
    """
    con_str = f'mysql+pymysql://{user}:{password}@{host}:{port}/{database}'
    try:
        engine = db.create_engine(url=con_str)

    except Exception as ex:
        print("Connection could not be made due to the following error: \n", ex)

    else:
        print(f"Connection to the {host} for user {user} created successfully.")
        return engine


# --- MAIN FUNCTION ---
def main():
    """
    Main function to control the whole program.
    """
    # try:
    engine = get_connection()  # Get the engine for the database
    file_path = r'C:\Users\felix\OneDrive\Studium\3_Master\1_IU\2_Module\0_Python\3_Hausarbeit\DLMDWPMP01_Hausarbeit\Datasets\ExampleDatasets\trainExample.csv'
    training = ds.Training(file_path, engine)


# except Exception as ex:
#   print("The following error occurred during program execution: \n", ex)


# --- Call main function
if __name__ == '__main__':
    main()
