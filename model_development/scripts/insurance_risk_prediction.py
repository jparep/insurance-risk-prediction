# Import libraries
import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
import joblib

# Logging Configuration
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    handlers=[logging.StreamHandler(), logging.FileHandler('app.log')])

# File directory
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

# Data file path
FILE_PATH = os.path.join(DIR, 'data', 'insurance.csv')
MODEL_PATH = os.path.join(DIR, 'models', 'model.joblib')

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
    """
    Cleans the input DataFrame by validating required columns, handling missing values,
    and transforming specific columns as necessary.
    """
    try:
        # Required columns for validation
        required_columns = ['age', 'children', 'sex', 'region', 'charges']
        missing_cols = set(required_columns) - set(df.columns)
        
        # Check for missing columns
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Handle negative or invalid 'age' values
        if 'age' in df.columns:
            df['age'] = df['age'].abs()  # Convert negative ages to positive
        
        # Convert 'children' to integer and handle negatives
        if 'children' in df.columns:
            df['children'] = df['children'].astype('Int64').abs()  # Convert and ensure positives
        
        # Map 'sex' to standardized values
        if 'sex' in df.columns:
            sex_mapping = {
                'female': 'Female', 'woman': 'Female', 'Woman': 'Female',
                'F': 'Female', 'f': 'Female', 'male': 'Male',
                'man': 'Male', 'Man': 'Male', 'M': 'Male', 'm': 'Male',
            }
            df['sex'] = df['sex'].map(sex_mapping)  # Standardize gender values

        # Map 'region' to standardized values
        if 'region' in df.columns:
            region_map = {
                'southwest': 'Southwest', 'southeast': 'Southeast',
                'northwest': 'Northwest', 'northeast': 'Northeast'
            }
            df['region'] = df['region'].map(region_map)  # Standardize region values
        
        # Handle 'charges' column: Remove '$' sign and convert to float
        if 'charges' in df.columns:
            df['charges'] = df['charges'].replace({'\$': ''}, regex=True).astype(float)
            df = df.dropna(subset=['charges']).reset_index(drop=True)  # Drop rows with missing 'charges'
        
        # Remove rows with excessive missing values (e.g., more than 5 missing)
        df = df[df.isnull().sum(axis=1) <= 5].reset_index(drop=True)

        logging.info("Data preprocessing completed successfully.")
        return df

    except Exception as e:
        logging.error(f"Error during data cleaning: {str(e)}")
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
        ('model', Ridge())  # Using Ridge regression for regularization
    ])

def hyperparameter_tuning(model_pipeline, X_train, y_train):
    """Tune hyperparameters"""
    cv = KFold(n_splits=5, shuffle=True, random_state=12)

    param_grid = {
        'model__alpha': np.logspace(-3, 3, 12,)  # Ridge regularization parameter
    }

    grid_search = GridSearchCV(estimator=model_pipeline,
                               param_grid=param_grid,
                               scoring='r2',
                               cv=cv,
                               n_jobs=-1)
    grid_search.fit(X_train, y_train)
    logging.info(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def model_evaluation(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)

    eval_mx = {
        'Mean Absolute Error (MAE)': mean_absolute_error(y_test, y_pred),
        'Mean Squared Error (MSE)': mean_squared_error(y_test, y_pred),
        'R-squared (R2)': r2_score(y_test, y_pred),
        'R-squared (R2) Percentage': r2_score(y_test, y_pred) * 100
    }

    # Print the evaluation metrics
    print("\nModel Evaluation Metrics:")
    for metric, value in eval_mx.items():
        print(f"{metric}: {value:.2f}")

def save_model(model, model_path):
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        logging.info(f"Model saved successfully to {model_path}")
    except Exception as e:
        logging.error(f'Error while saving model file: {str(e)}')
        raise

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

        # Train and tune model
        best_model = hyperparameter_tuning(model_pipeline, X_train, y_train)

        # Evaluate the model
        model_evaluation(best_model, X_test, y_test)
        
        # Save Model
        save_model(best_model, MODEL_PATH)
    except Exception as e:
        logging.critical(f"Error in main: {str(e)}")
        raise

if __name__ == '__main__':
    main()
