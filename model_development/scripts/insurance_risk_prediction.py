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
        logging.error(f"Issue loadding data from file {file_path}")

def main():
    # Load data
    df = load_data(FILE_PATH)
    print(df.head())

if __name__=='__main__':
    main()


