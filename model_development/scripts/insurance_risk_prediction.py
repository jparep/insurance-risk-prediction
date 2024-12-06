# Import libraries
import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import logging

# Logging Cnfiguration
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    handlers=[logging.StreamHandler(), logging.FileHandler('app.log')])

# Gloabal Variables
# Construct a relative path to the data dir
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')

# Get File path
FILE_PATH = os.path.join(DIR, 'insurance.csv')

def load_data(file_path):
    """Load data into DataFrame"""
    try:
        df = pd.read_csv(file_path)
        logging.info(f'Data Successfully loadded from file {file_path}')
        return df
    except Exception as e:
        logging.error(f"Issue loadding data from file {file_path}: {str(e)}")
        raise

def preprocess_data(df):
    """Preprocess data"""
    try:
        # Covert negative age values to positives
        if 'age' in df.columns:
            df['age'] = df['age'].abs()
    
        # Convert children to Int and handle negatives
        if 'children' in df.columns:
            df['children'] = df['children'].astype('int64').abs()
        
        # Map sex to male and female
        if 'sex' in df.columns:
            sex_mapping = {
                'female': 'Female',
                'woman': 'Female',
                'Woman': 'Female',
                'F': 'Female',
                'f': 'Female',
                'male': 'Male',
                'man': 'Male',
                'Man': 'Male',
                'M': 'Male',
                'm':'Male',
                }
            df['sex'] = df['sex'].map(sex_mapping)

           # Map Region values
        if 'region' in df.columns:
            region_map = {
                'southwest': 'Southwest',
                'southeast': 'Southeast',
                'northwest': 'Northwest',
                'northeast': 'Northeast'
            }
            df['region'] = df['region'].map(region_map)
        
        # Remove $, convert to Int andd drop Rows with missing target column values
        if 'chages' in df.columns:
            df['charges'] = df['charges'].replace({'\$': ''}, regex=True).astype(float)
            df = df.dropna(subset=['charges']).reset_index(drop=True)
    
        # Remove reows with more than 5 missing values from 7 columns
        df = df[df.isnull().sum(axis=1) < 5].reset_index(drop=True)
    
        return df
    except Exception as e:
        logging.error(f"Error during preprocessing {str(e)}")
        raise

def main():
    # Load data
    try:
        df = load_data(FILE_PATH)
        df_cleaned = preprocess_data(df)
        print(df_cleaned.head())
    except Exception as e:
        logging.critical(f"Error in main: {str(e)}")
        raise

if __name__=='__main__':
    main()


