import logging
import os
from config import LOG_FILE

def setup_logger():
    """
    Configures loggin for the project.
    """
    
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[logging.StreamHandler, logging.FileHandler(LOG_FILE)]
    )
    
    logging.info('Logger initalized.')