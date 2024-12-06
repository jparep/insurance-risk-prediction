import os

def validate_file_exists(file_path):
    """
    Validates that the given file path exists.

    Args:
        file_path (str): Path to the file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

def validate_required_columns(df, required_columns):
    """
    Validates that the DataFrame contains all required columns.

    Args:
        df (pandas.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.

    Raises:
        ValueError: If any required columns are missing.
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
