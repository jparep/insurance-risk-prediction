# Import libraries
import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Gloabal Variables
# Construct a relative path to the data dir
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')

# Get File path
FILE_PATH = os.path.join(DIR, 'insurance.csv')

# Load Data
df = pd.read_csv(FILE_PATH)

# Read First 5 rows
print(df.head())
