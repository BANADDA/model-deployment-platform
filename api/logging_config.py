import logging
from logging.handlers import RotatingFileHandler
import json

class JsonFormatter(logging.Formatter):
    """Custom formatter to output logs in JSON format."""
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "logger": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineNo": record.lineno
        }
        return json.dumps(log_record)

def setup_logging():
    """Set up logging for the application."""
    # Create a custom logger
    logger = logging.getLogger("model_deployment_platform")
    logger.setLevel(logging.DEBUG)  # Capture all levels of logs

    # Create handlers
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Console shows INFO and above

    file_handler = RotatingFileHandler("app.log", maxBytes=5*1024*1024, backupCount=2)
    file_handler.setLevel(logging.DEBUG)  # File captures DEBUG and above

    # Create formatter
    # For JSON formatted logs, use JsonFormatter
    # For plain text, use the default formatter
    use_json = False  # Set to True if JSON logs are desired

    if use_json:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add formatter to handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
