import logging
import os
import sys
from datetime import datetime

def setup_logger(name="cvrp_benchmark", log_file=None, level=logging.INFO):
    """
    Sets up a logger with a standard format.
    If log_file is provided, logs will be written to that file as well as stdout.
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if setup is called multiple times
    if logger.handlers:
        return logger
        
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    
    # File handler (if requested)
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

def get_task_logger(task_id):
    """
    Creates a specific logger for a background task.
    """
    log_dir = "server_output/logs"
    log_file = os.path.join(log_dir, f"task_{task_id}.log")
    return setup_logger(f"task_{task_id}", log_file=log_file)
