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

def preprocess_data(df):
    """Preprocess data"""
    # Remove $ and convert to Int from object
    df['charges'] = df['charges'].replace({'\$': ''}, regex=True).astype(float)
    # Covert negative age values to positives
    df['age'] = abs(df['age'])
    
    # Convert children to Int from object type
    df['children'] = df['children'].astype('int64')
    # Convert neagtive number of children to positive
    df['children'] = abs(df['children'])
    
    # Map sex to male and female
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

    # Appply mapping on sex
    df['sex'] = df['sex'].map(sex_mapping)
    
    # Map Region
    region_map = {
        'southwest': 'Southwest',
        'southeast': 'Southeast',
        'northwest': 'Northwest',
        'northeast': 'Northeast'
    }
    # Apply mapping on region
    df['region'] = df['region'].map(region_map)
    
    # Remove rows that are entirely empty accross the columns
    df = df.dropna(how='all').reset_index(drop=True)
    
    # If 5 out of 7 coulmns as empty values in the same row index, drop them
    df = df[df.isnull().sum() > 5].reset_index(drop=True)
    
    # Remove NAs from target: It will also remove values from features in the same row index
    df = df.dropna(subset=['target']).set_index(drop=True)
    return df

def main():
    # Load data
    try:
        df = load_data(FILE_PATH)
        df_cleaned = preprocess_data(df)
        print(df_cleaned.head())
    except Exception as e:
        logging.critical(f"Error in main: {str(e)}")

if __name__=='__main__':
    main()


