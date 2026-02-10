"""Colored logger for better console output visibility.

Provides colored logging with different colors for different log levels.
Works in PyCharm and other terminals.
"""

import logging
import sys


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
        """Format log record with colors.

        Args:
            record: LogRecord to format.

        Returns:
            Formatted string with ANSI color codes.
        """
        # Get color for log level
        color = self.COLORS.get(record.levelname, '')

        # Format message
        log_message = super().format(record)

        # Add color to level name
        if color:
            # Color the entire message
            colored_message = f"{color}{log_message}{self.RESET}"
            return colored_message

        return log_message


def get_logger(name: str = None, level: int = logging.INFO) -> logging.Logger:
    """Get or create a colored logger.

    Args:
        name: Logger name (usually __name__ of the module).
        level: Logging level (default: INFO).

    Returns:
        Configured logger with colored output.
    """
    logger = logging.getLogger(name or __name__)

    # Only configure if no handlers exist (avoid duplicate handlers)
    if not logger.handlers:
        logger.setLevel(level)

        # Create console handler - use stderr (same as tqdm) to avoid overlapping
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)

        # Create colored formatter
        formatter = ColoredFormatter(
            fmt='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(console_handler)

        # Prevent propagation to root logger
        logger.propagate = False

    return logger


def setup_logging(level: int = logging.INFO):
    """Setup colored logging for entire application.

    Args:
        level: Logging level to use (default: INFO).
    """
    # Configure root logger with colored formatter
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler with colored formatter - use stderr (same as tqdm)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)

    formatter = ColoredFormatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)


if __name__ == "__main__":
    # Test the colored logger
    setup_logging(logging.DEBUG)

    logger = get_logger(__name__)

    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    logger.critical("This is a CRITICAL message")
