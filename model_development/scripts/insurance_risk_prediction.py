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

# File paths
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
FILE_PATH = os.path.join(DIR, 'data', 'insurance.csv')
MODEL_PATH = os.path.join(DIR, 'models', 'model_pipeline.joblib')


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

        # Clean charges and drop NA rows
        df['charges'] = df['charges'].replace({'\$': ''}, regex=True).astype(float)
        df = df.dropna().reset_index(drop=True)

        logging.info("Data cleaning completed successfully.")
        return df
    except Exception as e:
        logging.error(f"Error during data cleaning: {str(e)}")
        raise


def preprocess_and_transform_data(X):
    """
    Preprocesses numerical and categorical features using a pipeline.
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
    """
    return Pipeline([
        ('preprocess', preprocessor),
        ('model', Ridge())
    ])


def hyperparameter_tuning(model_pipeline, X_train, y_train):
    """
    Performs hyperparameter tuning using grid search and reintegrates the tuned model
    into the pipeline.
    """
    # Ensure the pipeline is fitted before hyperparameter tuning
    model_pipeline.fit(X_train, y_train)

    # Define cross-validation and hyperparameter grid
    cv = KFold(n_splits=5, shuffle=True, random_state=12)
    param_grid = {
        'model__alpha': np.logspace(-3, 3, 12)
    }

    # Perform grid search
    grid_search = GridSearchCV(estimator=model_pipeline,
                               param_grid=param_grid,
                               scoring='r2',
                               cv=cv,
                               n_jobs=-1)
    grid_search.fit(X_train, y_train)

    logging.info(f"Best parameters: {grid_search.best_params_}")

    # Extract the best model and reintegrate it into the pipeline
    best_model = grid_search.best_estimator_.named_steps['model']
    final_pipeline = Pipeline([
        ('preprocess', model_pipeline.named_steps['preprocess']),
        ('model', best_model)
    ])

    return final_pipeline


def model_evaluation(model_pipeline, X_test, y_test):
    """
    Evaluates the performance of the model pipeline.
    """
    y_pred = model_pipeline.predict(X_test)
    eval_metrics = {
        'Mean Absolute Error (MAE)': mean_absolute_error(y_test, y_pred),
        'Mean Squared Error (MSE)': mean_squared_error(y_test, y_pred),
        'R-squared (R2)': r2_score(y_test, y_pred)
    }

    logging.info("Model Evaluation Metrics:")
    for metric, value in eval_metrics.items():
        logging.info(f"{metric}: {value:.2f}")


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
    """
    try:
        # Load and clean data
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
        final_pipeline = hyperparameter_tuning(model_pipeline, X_train, y_train)

        # Save the complete pipeline
        save_model_pipeline(final_pipeline, MODEL_PATH)

        # Evaluate the model
        model_evaluation(final_pipeline, X_test, y_test)
    except Exception as e:
        logging.critical(f"Error in main: {str(e)}")
        raise


if __name__ == '__main__':
    main()
