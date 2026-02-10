"""Colored logger for better console output visibility.

Provides colored logging with different colors for different log levels.
Works in PyCharm and other terminals.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',  # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'

    def format(self, record):
        """Format log record with colors."""
        # Get color for log level
        color = self.COLORS.get(record.levelname, '')

        # Format message
        log_message = super().format(record)

        # Add color to message
        if color:
            return f"{color}{log_message}{self.RESET}"
        return log_message


def get_logger(name: str = None, level: int = logging.INFO) -> logging.Logger:
    """Get a logger instance. Propagation is enabled to allow file logging via root."""
    logger = logging.getLogger(name or __name__)
    logger.setLevel(level)
    return logger


def setup_logging(level: int = logging.INFO):
    """
    Setup root logger to handle both Console and File output.
    Call this once at the start of your program (e.g., in train_vae.py).
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clean up existing handlers to avoid duplicates if re-initialized
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    log_file = create_log_file()

    # ---- console handler (with colors) ----
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_formatter = ColoredFormatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)

    # ---- file handler (plain text for the dashboard/logs) ----
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    print(f"Logging to: {log_file}")


def create_log_file():
    """Creates the logs directory and a timestamped log file."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = logs_dir / f"run_{timestamp}.log"
    return str(log_path)
