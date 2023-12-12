# main_script.py
from db_utils import RDSDatabaseConnector, LocalDataLoader
import pandas as pd

class DataTransform:
    
    @staticmethod
    def convert_to_numeric(df, numeric_columns):
        try:
            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
            return df
        except Exception as e:
            print(f"Error converting columns to numeric: {e}")
            return None

    @staticmethod
    def convert_to_datetime(df, date_columns, date_format='%b-%Y'):
        try:
            df[date_columns] = df[date_columns].apply(pd.to_datetime, format=date_format, errors='coerce')
            return df
        except Exception as e:
            print(f"Error converting columns to datetime: {e}")
            return None


def main():
    # Step 3: Create RDSDatabaseConnector instance
    rds_connector = RDSDatabaseConnector()

    # Step 4: Load credentials
    credentials = rds_connector.load_database_credentials()

    if credentials:
        # Step 6: Extract data from the RDS database as a DataFrame
        query = "SELECT * FROM loan_payments"
        data_df = rds_connector.extract_data_as_dataframe(query)

        # Step 7: Save data to a local CSV file
        rds_connector.save_data_to_csv(data_df, "loan_payments_data.csv")

        # Step 10: Load data from local CSV file into Pandas DataFrame
        file_path = "loan_payments_data.csv"
        loaded_data = LocalDataLoader.load_data_from_csv(file_path)

        if loaded_data is not None:
            # Step 11: Print the shape of the data
            print("Data Shape:", loaded_data.shape)

            # Step 12: Display a sample of the data
            print("Sample of the Data:")
            print(loaded_data.head())  # Display the first few rows as a sample

            # Step 13: Print information about each column
            print("\nColumn Information:")
            print(loaded_data.info())

            # Step 14: Display the first 5 values of each column
            print("\nFirst 5 Values of Each Column:")
            for column in loaded_data.columns:
                print(f"{column}: {loaded_data[column].head().tolist()}")

            # Step 15: Apply transformations using DataTransform
            transformer = DataTransform()
            numeric_columns = ['loan_amount', 'funded_amount', 'annual_inc', 'dti']
            loaded_data = transformer.convert_to_numeric(loaded_data, numeric_columns)

if __name__ == "__main__":
    main()

