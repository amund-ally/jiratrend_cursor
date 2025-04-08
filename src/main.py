from src.config.logging_config import setup_logging
from src.ui.app import create_ui

if __name__ == "__main__":
    # Setup logging with desired level
    setup_logging(log_level="INFO")  # Can be DEBUG, INFO, WARNING, ERROR
    create_ui()