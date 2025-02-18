import logging
import os
from datetime import datetime

def setup_logger(log_dir="logs", logger_name="RL_Project_Logger"):
    """
    Sets up and returns a logger that:
    1. Creates a log directory if it doesn't exist.
    2. Creates a log file for the current day if it doesn't exist.
    3. Initializes a logger.
    4. Adds a file handler (logs everything) and a console handler (logs INFO and above).
    5. Formats log messages with timestamps.
    6. Returns the configured logger.
    """

    # Ensure the logs directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Generate log filename based on today's date
    log_filename = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")

    # Create a logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # Capture all log levels

    # Check if logger already has handlers to prevent duplicate logs
    if not logger.hasHandlers():
        # Create a file handler
        file_handler = logging.FileHandler(log_filename, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Log everything to file

        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # Log INFO and above to console

        # Define log format
        formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger