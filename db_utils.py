# database_connector.py
import yaml
from sqlalchemy import create_engine
import pandas as pd
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

"""All the modules that will be used"""


class RDSDatabaseConnector:
    ''' Used to extract data from the cloud '''

    def __init__(self, credentials_file='credentials.yaml'):
        self.credentials_file = credentials_file
        self.credentials = self.load_database_credentials()
        self.engine = self.create_engine()

    def load_database_credentials(self):
        with open(self.credentials_file, 'r') as file:
            credentials = yaml.safe_load(file)
        return credentials

    def create_engine(self):
        engine = create_engine(
            f"postgresql+psycopg2://{self.credentials['RDS_USER']}:{self.credentials['RDS_PASSWORD']}@{self.credentials['RDS_HOST']}:{self.credentials['RDS_PORT']}/{self.credentials['RDS_DATABASE']}"
        )
        return engine

    def extract_data_as_dataframe(self, query):
        try:
            data_df = pd.read_sql_query(query, self.engine)
            return data_df
        except Exception as e:
            print(f"Error extracting data from the database: {e}")
            return None

    def save_data_to_csv(self, data_df, file_name):
        try:
            data_df.to_csv(file_name, index=False)
            print(f"Data saved to {file_name} successfully.")
        except Exception as e:
            print(f"Error saving data to CSV: {e}")



class LocalDataLoader:
    @staticmethod
    def load_data_from_csv(file_path):
        try:
            data_df = pd.read_csv(file_path)
            return data_df
        except Exception as e:
            print(f"Error loading data from CSV: {e}")
            return None



class DataFrameInfo:
    @staticmethod
    def describe_columns(df):
        return df.dtypes

    @staticmethod
    def extract_statistics(df):
        return df.describe()

    @staticmethod
    def count_distinct_values(df):
        return df.select_dtypes(include=['object']).nunique()

    @staticmethod
    def print_shape(df):
        print(f"DataFrame Shape: {df.shape}")

    @staticmethod
    def count_null_values(df):
        null_count = df.isnull().sum()
        null_percentage = (df.isnull().sum() / len(df)) * 100
        null_info = pd.DataFrame({
            'Null Count': null_count,
            'Null Percentage': null_percentage
        })
        return null_info


class Plotter:
    @staticmethod
    def plot_null_distribution(df):
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.isnull(), cmap='viridis', cbar=False, yticklabels=False)
        plt.title("Null Values Distribution")
        plt.show()


class DataFrameTransform:
    
    @staticmethod
    def drop_columns_with_nulls(df, threshold=0.5):
        null_percentage = df.isnull().mean()
        columns_to_drop = null_percentage[null_percentage > threshold].index
        df = df.drop(columns=columns_to_drop)
        return df

    @staticmethod
    def impute_missing_values(df, strategy='median'):
        numeric_columns = df.select_dtypes(include='number').columns
        imputer = SimpleImputer(strategy=strategy)
        df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
        return df
    
    @staticmethod
    def identify_skewed_columns(df, skewness_threshold=0.5):
        skewed_columns = df.apply(lambda x: x.skew()).abs() > skewness_threshold
        return df.columns[skewed_columns].tolist()


def identify_skewed_columns(df, skew_threshold=0.5):
    """
    Identify skewed columns in the DataFrame.

    Parameters:
    - df: DataFrame
    - skew_threshold: float, default=1.0
      The threshold for considering a column as skewed.

    Returns:
    - List of column names identified as skewed.
    """
    skewed_columns = df.apply(lambda x: x.skew()).sort_values(ascending=False)
    return skewed_columns[abs(skewed_columns) > skew_threshold].index.tolist()

