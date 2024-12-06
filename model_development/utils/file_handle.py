import os
import logging

def ensure_directory_exists(path):
    """
    Ensure that the directory for the given path exists.
    Create the directiry if it does not exists.
    Args:
        path (str): File path or directory path.
    """
    
    try:
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Directory ensured: {path}")
    except Exception as e:
        logging.error(f"Error ensuring directory {str(e)}")
        raise