# Import libraries
import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

# Logging Configuration
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    handlers=[logging.StreamHandler(), logging.FileHandler('app.log')])

# Global Variables
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')
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

def clean_data(df):
    """Preprocess data"""
    try:
        # Handle negative or invalid 'age' values
        df['age'] = df['age'].abs()

        # Convert 'children' to integers and handle negatives
        df['children'] = df['children'].astype('Int64').abs()

        # Map 'sex' to standardized values
        sex_mapping = {'female': 'Female', 'male': 'Male', 'man': 'Male', 'woman': 'Female'}
        df['sex'] = df['sex'].map(sex_mapping)

        # Map 'region' to standardized values
        region_map = {'southwest': 'Southwest', 'southeast': 'Southeast',
                      'northwest': 'Northwest', 'northeast': 'Northeast'}
        df['region'] = df['region'].map(region_map)

        # Handle 'charges' column: Remove '$' sign, convert to float
        df['charges'] = df['charges'].replace({'\$': ''}, regex=True).astype(float)

        # Drop rows with missing target values
        df = df.dropna(subset=['charges']).reset_index(drop=True)

        logging.info("Data preprocessing completed successfully.")
        return df
    except Exception as e:
        logging.error(f"Error during preprocessing: {str(e)}")
        raise

def preprocess_and_transform_data(X):
    """Preprocess and transform data into pipeline"""
    # Get numerical and categorical columns
    num_columns = X.select_dtypes(include=['int64', 'Int64', 'float64']).columns
    cat_columns = X.select_dtypes(include=['object']).columns

    # Create preprocessing pipeline for numerical data
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', RobustScaler())
    ])

    # Create preprocessing pipeline for categorical data
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing pipelines
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, num_columns),
        ('cat', cat_pipeline, cat_columns)
    ])

    return preprocessor

def build_model(preprocessor):
    """Build Model Pipeline"""
    return Pipeline([
        ('preprocess', preprocessor),
        ('model', Ridge(alpha=1.0))  # Default Ridge regularization
    ])

def model_evaluation(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    eval_mx = {
        'Mean Absolute Error (MAE)': mean_absolute_error(y_test, y_pred),
        'Mean Squared Error (MSE)': mean_squared_error(y_test, y_pred),
        'R-squared (R2)': r2_score(y_test, y_pred) * 100
    }

    print("\nModel Evaluation Metrics:")
    for metric, value in eval_mx.items():
        print(f"{metric}: {value:.2f}")

def main():
    try:
        # Load data
        df = load_data(FILE_PATH)

        # Clean data
        df_cleaned = clean_data(df)

        # Separate features and target
        X = df_cleaned.drop(columns=['charges'], axis=1)
        y = df_cleaned['charges']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

        # Preprocess and transform data
        preprocessor = preprocess_and_transform_data(X)

        # Build Model
        model_pipeline = build_model(preprocessor)

        # Train the model
        model_pipeline.fit(X_train, y_train)

        # Evaluate the model
        model_evaluation(model_pipeline, X_test, y_test)
    except Exception as e:
        logging.critical(f"Error in main: {str(e)}")
        raise

if __name__ == '__main__':
    main()
