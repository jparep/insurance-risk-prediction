import os

# Base Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data and model paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LODS_DIR = os.path.join(BASE_DIR, 'logs')

# File paths
FILE_PATH = os.path.join(DATA_DIR, 'insurance.csv')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.joblib')
LOG_FILE = os.path.join(LODS_DIR, 'app.log')