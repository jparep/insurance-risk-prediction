import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
import joblib

# Logging Configuration
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO,
                    handlers=[logging.StreamHandler(), logging.FileHandler('app.log')])

# File paths
DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(DIR, '../data/insurance.csv')
VALIDATION_PATH = os.path.join(DIR, '../data/validation_dataset.csv')
MODEL_PATH = os.path.join(DIR, '../models/model_pipeline.joblib')


def load_data(file_path):
    """
    Loads data from a CSV file into a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data successfully loaded from file: {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from file {file_path}: {str(e)}")
        raise


def clean_data(df):
    """
    Cleans the input DataFrame by validating required columns and transforming specific columns.

    Args:
        df (pandas.DataFrame): Raw dataset.

    Returns:
        pandas.DataFrame: Cleaned dataset.

    Raises:
        ValueError: If required columns are missing.
    """
    try:
        required_columns = ['age', 'bmi', 'children', 'sex', 'region', 'charges']
        missing_cols = set(required_columns) - set(df.columns)

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Absolute values for age and children
        df['age'] = df['age'].abs()
        df['children'] = df['children'].astype('Int64').abs()

        # Map sex and region
        sex_mapping = {
            'female': 'Female', 'woman': 'Female', 'F': 'Female', 'f': 'Female',
            'male': 'Male', 'man': 'Male', 'M': 'Male', 'm': 'Male'
        }
        region_mapping = {
            'southwest': 'Southwest', 'southeast': 'Southeast',
            'northwest': 'Northwest', 'northeast': 'Northeast'
        }
        df['sex'] = df['sex'].map(sex_mapping)
        df['region'] = df['region'].map(region_mapping)

        # Clean the charges column: Remove '$' and convert to float
        if 'charges' in df.columns:
            df['charges'] = df['charges'].replace({'\$': ''}, regex=True).astype(float)

        # Remove rows with missing or invalid values
        df = df.dropna().reset_index(drop=True)

        logging.info("Data cleaning completed successfully.")
        return df
    except Exception as e:
        logging.error(f"Error during data cleaning: {str(e)}")
        raise


def preprocess_and_transform_data(X):
    """
    Preprocesses numerical and categorical features using a pipeline.

    Args:
        X (pandas.DataFrame): Features dataset.

    Returns:
        sklearn.compose.ColumnTransformer: Preprocessing pipeline.
    """
    num_columns = X.select_dtypes(include=['int64', 'Int64', 'float64']).columns
    cat_columns = X.select_dtypes(include=['object']).columns

    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', RobustScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, num_columns),
        ('cat', cat_pipeline, cat_columns)
    ])

    return preprocessor


def build_model(preprocessor):
    """
    Builds a machine learning model pipeline.

    Args:
        preprocessor (ColumnTransformer): Preprocessing pipeline.

    Returns:
        sklearn.pipeline.Pipeline: Complete model pipeline.
    """
    return Pipeline([
        ('preprocess', preprocessor),
        ('model', RandomForestRegressor(random_state=42))
    ])


def hyperparameter_tuning(model_pipeline, X_train, y_train):
    """
    Performs hyperparameter tuning using grid search.

    Args:
        model_pipeline (Pipeline): Model pipeline to tune.
        X_train (pandas.DataFrame): Training features.
        y_train (pandas.Series): Training target variable.

    Returns:
        sklearn.pipeline.Pipeline: Best model pipeline after tuning.
    """
    cv = KFold(n_splits=5, shuffle=True, random_state=12)

    param_grid = {
        'model__n_estimators': [100, 200, 300],  # Number of trees in the forest
        'model__max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
        'model__min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
        'model__min_samples_leaf': [1, 2, 4]  # Minimum samples required to be at a leaf node
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
    """
    Evaluates the performance of the model.

    Args:
        model (Pipeline): Trained model pipeline.
        X_test (pandas.DataFrame): Test features.
        y_test (pandas.Series): Test target variable.

    Returns:
        None
    """
    y_pred = model.predict(X_test)
    eval_metrics = {
        'Mean Absolute Error (MAE)': mean_absolute_error(y_test, y_pred),
        'Mean Squared Error (MSE)': mean_squared_error(y_test, y_pred),
        'R-squared (R2)': r2_score(y_test, y_pred)
    }

    logging.info("Model Evaluation Metrics:")
    for metric, value in eval_metrics.items():
        logging.info(f"{metric}: {value:.2f}")


def validate_on_new_data(model_pipeline, validation_file):
    """
    Validates the trained model on a new validation dataset without a target variable.

    Args:
        model_pipeline (Pipeline): Trained model pipeline.
        validation_file (str): Path to the validation dataset.

    Returns:
        pandas.DataFrame: Predictions for the validation dataset.
    """
    try:
        df_val = pd.read_csv(validation_file)
        logging.info(f"Validation dataset successfully loaded from {validation_file}")

        # Preprocess validation data
        predictions = model_pipeline.predict(df_val)
        df_val['Predicted Charges'] = predictions

        logging.info("Validation predictions generated successfully.")
        return df_val
    except Exception as e:
        logging.error(f"Error during validation: {str(e)}")
        raise


def save_model_pipeline(model_pipeline, model_path):
    """
    Saves the complete model pipeline (preprocessing + model) to a file.
    """
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model_pipeline, model_path)
        logging.info(f"Model pipeline saved successfully to {model_path}")
    except Exception as e:
        logging.error(f"Error while saving pipeline: {str(e)}")
        raise


def main():
    """
    Main function to execute the entire pipeline:
    - Load data
    - Clean data
    - Preprocess data
    - Train and tune model
    - Evaluate model
    - Save pipeline (preprocessing + tuned model)
    - Validate on new dataset
    """
    try:
        # Load and clean training data
        df = load_data(FILE_PATH)
        df_cleaned = clean_data(df)

        # Split into features and target
        X = df_cleaned.drop(columns=['charges'], axis=1)
        y = df_cleaned['charges']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

        # Build pipeline and tune hyperparameters
        preprocessor = preprocess_and_transform_data(X)
        model_pipeline = build_model(preprocessor)
        best_model_pipeline = hyperparameter_tuning(model_pipeline, X_train, y_train)

        # Evaluate the model
        model_evaluation(best_model_pipeline, X_test, y_test)

        # Save the complete pipeline
        save_model_pipeline(best_model_pipeline, MODEL_PATH)

        # Validate on new dataset
        validation_results = validate_on_new_data(best_model_pipeline, VALIDATION_PATH)
        print("\nValidation Predictions:\n", validation_results.head())

    except Exception as e:
        logging.critical(f"Error in main: {str(e)}")
        raise


if __name__ == '__main__':
    main()
