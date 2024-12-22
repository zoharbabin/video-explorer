import logging
import os
import sys
from datetime import datetime
from typing import Optional

# ANSI color codes for different log levels
COLORS = {
    'DEBUG': '\033[36m',    # Cyan
    'INFO': '\033[32m',     # Green
    'WARNING': '\033[33m',  # Yellow
    'ERROR': '\033[31m',    # Red
    'CRITICAL': '\033[35m', # Magenta
    'RESET': '\033[0m'      # Reset
}

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors, timestamps, and line numbers"""
    
    def __init__(self, fmt: Optional[str] = None):
        super().__init__(fmt)
        
    def format(self, record: logging.LogRecord) -> str:
        # Add color based on log level
        levelname = record.levelname
        if levelname in COLORS:
            colored_levelname = f"{COLORS[levelname]}{levelname}{COLORS['RESET']}"
            record.levelname = colored_levelname
        
        # Add timestamp
        record.timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # Format the message
        result = super().format(record)
        
        return result

def setup_logging(level: str = "INFO") -> None:
    """
    Set up logging configuration with colors and formatting
    
    Args:
        level: The logging level to use (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Get log level from environment variable or use default
    log_level = os.getenv('LOG_LEVEL', level).upper()
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Create formatter
    formatter = ColoredFormatter(
        fmt='%(timestamp)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to root logger
    root_logger.addHandler(console_handler)
    
    # Configure third-party library logging
    logging.getLogger('boto3').setLevel(os.getenv('AWS_LOG_LEVEL', 'WARNING'))
    logging.getLogger('botocore').setLevel(os.getenv('AWS_LOG_LEVEL', 'WARNING'))
    logging.getLogger('urllib3').setLevel(os.getenv('HTTP_LOG_LEVEL', 'WARNING'))
    logging.getLogger('KalturaClient').setLevel(os.getenv('KALTURA_LOG_LEVEL', 'WARNING'))
    logging.getLogger('litellm').setLevel(os.getenv('LITELLM_LOG_LEVEL', 'WARNING'))

# Create a logger for this module
logger = logging.getLogger(__name__)