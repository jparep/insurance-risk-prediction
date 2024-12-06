# Import libraries
import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import logging

# Logging Configuration
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    handlers=[logging.StreamHandler(), logging.FileHandler('app.log')])

# Global Variables
# Construct a relative path to the data directory
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')

# Get file path
FILE_PATH = os.path.join(DIR, 'insurance.csv')

def load_data(file_path):
    """Load data into DataFrame"""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data successfully loaded from file: {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from file {file_path}: {str(e)}")
        raise

def preprocess_data(df):
    """Preprocess data"""
    try:
        # Handle negative or invalid 'age' values
        if 'age' in df.columns:
            df['age'] = df['age'].abs()
        
        # Convert 'children' to int and handle negatives
        if 'children' in df.columns:
            df['children'] = df['children'].astype('Int64').abs()
        
        # Map 'sex' to standardized values
        if 'sex' in df.columns:
            sex_mapping = {
                'female': 'Female', 'woman': 'Female', 'Woman': 'Female',
                'F': 'Female', 'f': 'Female', 'male': 'Male',
                'man': 'Male', 'Man': 'Male', 'M': 'Male', 'm': 'Male',
            }
            df['sex'] = df['sex'].map(sex_mapping)
        
        # Map 'region' to standardized values
        if 'region' in df.columns:
            region_map = {
                'southwest': 'Southwest', 'southeast': 'Southeast',
                'northwest': 'Northwest', 'northeast': 'Northeast'
            }
            df['region'] = df['region'].map(region_map)
        
        # Handle 'charges' column: Remove '$' sign, convert to float, and drop rows with missing target
        if 'charges' in df.columns:
            df['charges'] = df['charges'].replace({'\$': ''}, regex=True).astype(float)
            df = df.dropna(subset=['charges']).reset_index(drop=True)
        
        # Remove rows with more than 5 missing values
        df = df[df.isnull().sum(axis=1) <= 5].reset_index(drop=True)

        logging.info("Data preprocessing completed successfully.")
        return df
    except Exception as e:
        logging.error(f"Error during preprocessing: {str(e)}")
        raise

def main():
    try:
        # Load data
        df = load_data(FILE_PATH)

        # Preprocess data
        df_cleaned = preprocess_data(df)

        # Output the cleaned data
        print(df_cleaned.head())
    except Exception as e:
        logging.critical(f"Error in main: {str(e)}")
        raise

if __name__ == '__main__':
    main()
