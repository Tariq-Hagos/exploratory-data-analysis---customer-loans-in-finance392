#db_utils.py
# seperate all the functions
import yaml
from sqlalchemy import create_engine
import pandas as pd
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
    
    @staticmethod
    def identify_and_visualize_skewed_columns(df, skewness_threshold=0.5):
        numeric_columns = df.select_dtypes(include='number').columns
        numeric_df = df[numeric_columns]
        
        if not numeric_df.empty:
            skewed_columns = numeric_df.apply(lambda x: x.skew()).abs() > skewness_threshold
            
            # Use the original DataFrame columns to avoid indexing issues
            selected_columns = [col for col in df.columns if col in numeric_columns and col in skewed_columns.index and skewed_columns[col]]
            return selected_columns
        else:
            return []

    @staticmethod
    def identify_skewed_columns(df, skewness_threshold=0.5):
        numeric_columns = df.select_dtypes(include='number').columns
        numeric_df = df[numeric_columns]

        if not numeric_df.empty:
            skewed_columns = numeric_df.apply(lambda x: x.skew()).abs() > skewness_threshold
            return numeric_df.columns[skewed_columns].tolist()
        else:
            return []


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
    def transform_skewed_columns(df, skewed_columns):
        for column in skewed_columns:
            # Check if the column exists in the DataFrame
            if column in df.columns:
                df[column] = np.log1p(df[column])
        return df

class Plotter:

    def plot_null_distribution(df):
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.isnull(), cmap='viridis', cbar=False, yticklabels=False)
        plt.title("Null Values Distribution")
        plt.show()


    def visualize_skewness(df, skewed_columns):
        plt.figure(figsize=(12, 8))
        for column in skewed_columns:
            sns.histplot(df[column], kde=True, label=column)
        plt.title("Skewed Columns Distribution")
        plt.legend()
        plt.show()

    def multi_hist_plot(df, num_cols):
        sns.set(font_scale=0.7)
        f = pd.melt(df, value_vars=num_cols)
        g = sns.FacetGrid(f, col="variable", col_wrap=4,
                          sharex=False, sharey=False)
        g = g.map(sns.histplot, "value", kde=True)
        plt.show()

    def qq_plot(cols):
        remainder = 1 if len(cols) % 4 != 0 else 0
        rows = int(len(cols) / 4 + remainder)

        fig, axes = plt.subplots(
            ncols=4, nrows=rows, sharex=False, figsize=(12, 6))
        for col, ax in zip(cols, np.ravel(axes)):
            sm.qqplot(df[col], line='s', ax=ax, fit=True)
            ax.set_title(f'{col} QQ Plot')
        pyplot.tight_layout()

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
   
    def transform_skewed_columns(df, skewed_columns):
        for column in skewed_columns:
            df[column] = np.log1p(df[column])  
        return df


def identify_skewed_columns(df, skewness_threshold=0.5):
    numeric_columns = df.select_dtypes(include='number').columns
    numeric_df = df[numeric_columns]

    skewed_columns = numeric_df.apply(lambda x: x.skew()).abs() > skewness_threshold
    return numeric_df.columns[skewed_columns].tolist()



