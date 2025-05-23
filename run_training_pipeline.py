import subprocess
import logging
import json
import csv
import os
import datetime
from utils import setup_logging

TRAINING_METRICS_FILE = 'training_metrics.json'
TRAINING_LOG_CSV = 'training_log.csv'

def run_script(script_name):
    """Runs a Python script using subprocess and checks its return code."""
    logging.info(f"Starting execution of '{script_name}'...")
    try:
        process = subprocess.run(['python', script_name], check=True, capture_output=True, text=True)
        logging.info(f"'{script_name}' completed successfully.")
        logging.debug(f"Output from '{script_name}':\n{process.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during execution of '{script_name}':")
        logging.error(f"Return code: {e.returncode}")
        logging.error(f"Stdout:\n{e.stdout}")
        logging.error(f"Stderr:\n{e.stderr}")
        return False
    except FileNotFoundError:
        logging.error(f"Error: Script '{script_name}' not found.")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred while running '{script_name}': {e}", exc_info=True)
        return False

def read_training_metrics():
    """Reads metrics from the JSON file."""
    logging.info(f"Attempting to read metrics from '{TRAINING_METRICS_FILE}'...")
    if not os.path.exists(TRAINING_METRICS_FILE):
        logging.error(f"Metrics file '{TRAINING_METRICS_FILE}' not found.")
        return None
    try:
        with open(TRAINING_METRICS_FILE, 'r') as f:
            metrics = json.load(f)
        logging.info(f"Successfully read metrics: {metrics}")
        return metrics
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from '{TRAINING_METRICS_FILE}': {e}", exc_info=True)
        return None
    except Exception as e:
        logging.error(f"Error reading metrics file '{TRAINING_METRICS_FILE}': {e}", exc_info=True)
        return None

def log_metrics_to_csv(timestamp, accuracy, loss):
    """Appends training metrics to a CSV file."""
    logging.info(f"Logging metrics to '{TRAINING_LOG_CSV}'...")
    file_exists = os.path.isfile(TRAINING_LOG_CSV)
    try:
        with open(TRAINING_LOG_CSV, 'a', newline='') as csvfile:
            fieldnames = ['timestamp', 'accuracy', 'loss']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()
                logging.info(f"Created CSV header in '{TRAINING_LOG_CSV}'.")
            
            writer.writerow({'timestamp': timestamp, 'accuracy': accuracy, 'loss': loss})
        logging.info("Metrics successfully logged to CSV.")
    except IOError as e:
        logging.error(f"IOError writing to CSV file '{TRAINING_LOG_CSV}': {e}", exc_info=True)
    except Exception as e:
        logging.error(f"Unexpected error writing to CSV file '{TRAINING_LOG_CSV}': {e}", exc_info=True)

def cleanup_metrics_file():
    """Deletes the temporary metrics JSON file."""
    if os.path.exists(TRAINING_METRICS_FILE):
        try:
            os.remove(TRAINING_METRICS_FILE)
            logging.info(f"Successfully deleted temporary metrics file '{TRAINING_METRICS_FILE}'.")
        except OSError as e:
            logging.warning(f"Could not delete temporary metrics file '{TRAINING_METRICS_FILE}': {e}", exc_info=True)

def main_pipeline():
    """Main function to run the training pipeline."""
    logging.info("Starting training pipeline...")

    # Step 1: Run create_dataset.py
    if not run_script('create_dataset.py'):
        logging.critical("Dataset creation failed. Exiting pipeline.")
        return

    # Step 2: Run data_classify.py
    if not run_script('data_classify.py'):
        logging.critical("Model training failed. Exiting pipeline.")
        return

    # Step 3: Read metrics and log to CSV
    metrics = read_training_metrics()
    if metrics:
        current_timestamp = datetime.datetime.now().isoformat()
        accuracy = metrics.get('accuracy')
        loss = metrics.get('loss')

        if accuracy is not None and loss is not None:
            log_metrics_to_csv(current_timestamp, accuracy, loss)
        else:
            logging.error("Metrics file does not contain 'accuracy' or 'loss' keys.")
    else:
        logging.error("Failed to read training metrics. Cannot log to CSV.")

    # Step 4: Cleanup
    cleanup_metrics_file()

    logging.info("Training pipeline finished.")

if __name__ == "__main__":
    setup_logging(log_level=logging.INFO, log_file="training_pipeline.log")
    main_pipeline()
