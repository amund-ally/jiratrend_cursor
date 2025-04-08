"""Configuration for application logging."""
import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path

def setup_logging(log_level: str = "INFO") -> None:
    """Configure application-wide logging.
    
    Args:
        log_level: Minimum log level to display (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    log_file = log_dir / f"jiratrend_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s | %(name)s | %(message)s'
    )
    
    # Get root logger and remove any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()  # Clear existing handlers
    
    # Set base logging level
    root_logger.setLevel(logging.DEBUG)

    # Create and configure file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)  # Log everything to file
    
    # Create and configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    #root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)