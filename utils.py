import logging
import sys

def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Set up centralized logging configuration.

    Args:
        log_level (int): The minimum logging level to output.
        log_file (str, optional): Path to a log file. If None, logs only to console.
                                  If provided, logs to both console and file.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Common formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(processName)s - %(message)s')

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File Handler (optional)
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, mode='a') # Append mode
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            logging.info(f"Logging to file: {log_file}")
        except Exception as e:
            logging.error(f"Failed to set up file logger at {log_file}: {e}", exc_info=True)

    logging.info("Logging setup complete.")

if __name__ == '__main__':
    # Example usage of the logger setup
    setup_logging(log_level=logging.DEBUG, log_file="app.log")
    logging.debug("This is a debug message.")
    logging.info("This is an info message.")
    logging.warning("This is a warning message.")
    logging.error("This is an error message.")
    logging.critical("This is a critical message.")

    try:
        1 / 0
    except ZeroDivisionError:
        logging.error("ZeroDivisionError occurred", exc_info=True)
