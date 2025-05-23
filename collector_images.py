"""
ASL Alphabet Image Collector

This script collects images from a webcam for each letter of the American Sign Language (A-Z).
It creates a dataset for training machine learning models for ASL recognition.

Usage:
    - Run the script.
    - Follow the prompts to start collecting images for each letter.
    - Press 'Q' to begin capturing images for the current letter.
    - The script saves the specified number of images per letter in the 'data' directory.

Dependencies:
    - OpenCV (cv2)
    - Python 3.x

Author: @ronal-lc
"""

import os
import sys
import cv2
import threading
import queue
import time
import logging
from utils import setup_logging # Import the setup_logging function

DATA_DIR = './data'
NUMBER_OF_CLASSES = 26  # 26 letters (A-Z)
DATASET_SIZE = 100
CAMERA_INDICES = [0, 1, 2] # Users can add more camera indices if needed
WINDOW_NAME = 'ASL Alphabet Image Collector'
# Optional: Users can experiment with different frame sizes for performance.
# Smaller frames (e.g., 640x480) can lead to faster processing but lower image quality.
# Larger frames (e.g., 1280x720) provide higher quality but may slow down capture.
# To set a specific frame size, uncomment and adjust the following lines
# in the select_camera function, after cap.isOpened():
# FRAME_WIDTH = 640
# FRAME_HEIGHT = 480
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)


def ensure_dir(path):
    """Create directory if it does not exist."""
    try:
        os.makedirs(path, exist_ok=True)
        logging.info(f"Directory '{path}' ensured (created if not existed).")
    except Exception as e:
        logging.error(f"Failed to create directory {path}: {e}", exc_info=True)
        # Depending on the script's needs, this might be a critical error.
        # For now, we log and let the script decide if it can proceed.


def select_camera(indices):
    """Try to open a camera from the provided indices."""
    logging.info(f"Attempting to select camera from indices: {indices}")
    for idx in indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            logging.info(f"Camera opened successfully with index {idx}.")
            # Optional: Set frame width and height if desired
            # frame_width_to_set = 640 # Example
            # frame_height_to_set = 480 # Example
            # cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width_to_set)
            # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height_to_set)
            # actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            # actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            # logging.info(f"Attempted to set frame size to {frame_width_to_set}x{frame_height_to_set}. Actual: {actual_width}x{actual_height}")
            return cap
        cap.release()
        logging.warning(f"Could not open camera with index {idx}.")
    logging.critical("No camera could be opened. Please check your camera connections and permissions.")
    sys.exit("Critical: No camera found.")


def setup_window(window_name):
    """Set up the OpenCV window."""
    try:
        cv2.namedWindow(window_name)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        logging.info(f"Window '{window_name}' set up successfully.")
    except Exception as e:
        logging.error(f"Failed to set up OpenCV window '{window_name}': {e}", exc_info=True)
        # This might be critical depending on whether the GUI is essential.
        # For this script, it is, so we might consider exiting or raising.
        raise # Re-raise the exception as window setup is critical


def capture_initial_ready(cap, window_name):
    """
    Display a ready screen before starting image capture.
    Returns True if user presses 'Q' to start, False if window is closed.
    """
    logging.info("Displaying 'Ready' screen. Waiting for user to press 'Q'.")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            logging.warning("Error capturing frame for 'Ready' screen. Retrying...")
            time.sleep(0.1) # Avoid busy-waiting
            continue

        cv2.putText(frame, 'Ready? Press "Q" to start', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(25) & 0xFF

        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                logging.info("Window closed by user during 'Ready' screen.")
                return False # Window closed by user
        except cv2.error: # Handle cases where window might be destroyed abruptly
            logging.warning("OpenCV window check failed, assuming window was closed.")
            return False


        if key == ord('q'):
            logging.info("User pressed 'Q'. Starting image capture for the class.")
            break
    return True


def image_saver_worker(q, stop_event):
    """Worker thread function to save images from a queue."""
    logging.info("Image saver worker thread started.")
    images_saved_count = 0
    while not stop_event.is_set() or not q.empty():
        try:
            item = q.get(timeout=0.1) # Wait for an item with timeout
            if item is None: # Sentinel to indicate no more items
                logging.debug("Sentinel received by image saver worker.")
                q.task_done()
                break # Exit loop
            
            img_path, frame_to_save = item
            try:
                cv2.imwrite(img_path, frame_to_save)
                images_saved_count += 1
                logging.debug(f"Successfully saved image: {img_path}")
            except Exception as e_imwrite:
                logging.error(f"Failed to write image {img_path}: {e_imwrite}", exc_info=True)
            q.task_done() # Signal that the item from queue is processed
        except queue.Empty:
            # This is expected when the queue is empty and timeout occurs.
            # Allows checking stop_event periodically.
            continue
        except Exception as e_queue:
            logging.error(f"Exception in image saver worker: {e_queue}", exc_info=True)
            # If item was retrieved, mark as done to avoid deadlocks
            if 'item' in locals() and item is not None:
                 q.task_done()
    logging.info(f"Image saver worker thread finished. Total images saved by this worker: {images_saved_count}.")


def capture_images(cap, class_dir, dataset_size, window_name):
    """
    Capture images from the camera and put them in a queue for saving.
    Returns True if completed, False if window is closed or interrupted.
    """
    images_queued_count = 0
    save_queue = queue.Queue(maxsize=dataset_size * 2) # Maxsize to prevent runaway queue growth
    stop_event = threading.Event()

    saver_thread = threading.Thread(target=image_saver_worker, args=(save_queue, stop_event), daemon=True)
    saver_thread.start()
    logging.info(f"Image capture process started for class directory: {class_dir}. Target: {dataset_size} images.")

    try:
        while images_queued_count < dataset_size:
            ret, frame = cap.read()
            if not ret or frame is None:
                logging.warning(f"Error capturing frame {images_queued_count + 1}. Retrying...")
                time.sleep(0.1) # Avoid busy-waiting
                continue

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF 

            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    logging.info("Window closed by user during image capture.")
                    return False # Indicate interruption
            except cv2.error:
                logging.warning("OpenCV window check failed during capture, assuming window was closed.")
                return False


            # Add frame to queue for saving
            frame_copy = frame.copy() # Essential to avoid saving a later frame due to buffer reuse
            img_path = os.path.join(class_dir, f'{images_queued_count}.jpg')
            
            try:
                save_queue.put((img_path, frame_copy), timeout=1) # Timeout to prevent blocking if queue is full
                images_queued_count += 1
                # Use logging for progress, print can be too verbose or get lost
                if images_queued_count % 10 == 0 or images_queued_count == dataset_size: # Log every 10 images
                    logging.info(f"Capture progress: {images_queued_count}/{dataset_size}")
                # Simple print for console progress is also fine for interactive scripts
                print(f"\rProgress: {images_queued_count}/{dataset_size}", end="", flush=True)
            except queue.Full:
                logging.warning("Save queue is full. Image saving might be lagging. Pausing capture slightly.")
                time.sleep(0.5) # Give saver thread time to catch up

        print() # Newline after progress bar
        logging.info(f"Successfully queued {images_queued_count} images for saving.")
        return True # Indicate successful completion of queuing
    except KeyboardInterrupt:
        logging.info("\nImage capture interrupted by user (Ctrl+C).")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during image capture: {e}", exc_info=True)
        return False
    finally:
        logging.info("Signaling image saver worker to stop and process remaining queue...")
        stop_event.set() # Signal the worker to stop after processing queue
        save_queue.put(None) # Sentinel to ensure worker wakes up if waiting on empty queue
        
        # Wait for the saver thread to finish processing items in the queue
        # This ensures all captured images are attempted to be saved before moving to next class or exiting
        if saver_thread.is_alive():
            logging.info("Waiting for image saver thread to complete...")
            saver_thread.join(timeout=15) # Increased timeout for potentially large queues
            if saver_thread.is_alive():
                logging.warning("Image saver thread did not terminate cleanly after timeout. Some images might not be saved.")
            else:
                logging.info("Image saver thread completed.")
        logging.info(f"Image capture process for class directory '{class_dir}' finished.")


def run_image_collection():
    """Main function to run the image collection process."""
    logging.info("Starting ASL Alphabet Image Collector script.")
    ensure_dir(DATA_DIR)
    
    cap = select_camera(CAMERA_INDICES) # Exits if no camera found
    
    try:
        setup_window(WINDOW_NAME) # Raises exception if fails
    except Exception as e: # Catching exception from setup_window
        logging.critical(f"Failed to setup OpenCV window due to: {e}. Cannot continue.", exc_info=True)
        if cap: cap.release()
        return # Exit script if window setup fails.

    start_class_input = input(f"Start collecting from class 0 (A)? (y/n, default y): ").strip().lower()
    start_class_num = 0
    if start_class_input == 'n':
        while True:
            try:
                raw_input_val = input(f"Enter the class number to start from (0-{NUMBER_OF_CLASSES - 1}): ")
                start_class_num = int(raw_input_val)
                if 0 <= start_class_num < NUMBER_OF_CLASSES:
                    logging.info(f"User selected to start from class {start_class_num}.")
                    break
                else:
                    logging.warning(f"Invalid class number input. Please enter a number between 0 and {NUMBER_OF_CLASSES - 1}.")
            except ValueError:
                logging.warning("Invalid input. Please enter a valid number for the class.")
    elif start_class_input == 'y' or not start_class_input:
        start_class_num = 0
        logging.info("Starting collection from class 0 (A).")
    else:
        logging.warning("Invalid input for starting class. Defaulting to class 0 (A).")
        start_class_num = 0

    collection_fully_completed = True
    try:
        for cls_idx in range(start_class_num, NUMBER_OF_CLASSES):
            class_name = chr(65 + cls_idx)
            class_dir = os.path.join(DATA_DIR, class_name)
            ensure_dir(class_dir) # Ensures directory exists
            logging.info(f"Starting data collection for class '{class_name}' (Class Index {cls_idx}).")

            if not capture_initial_ready(cap, WINDOW_NAME):
                logging.warning("Data collection interrupted by user (closed window or other issue before 'Q' pressed).")
                collection_fully_completed = False
                break # Stop collecting further classes

            if not capture_images(cap, class_dir, DATASET_SIZE, WINDOW_NAME):
                logging.warning(f"Data collection for class '{class_name}' was interrupted or failed.")
                collection_fully_completed = False
                # Decide whether to break or continue with next class. For now, let's break.
                break 
            
            logging.info(f"Successfully completed data collection for class '{class_name}'.")

        if collection_fully_completed and start_class_num == 0 and cls_idx == NUMBER_OF_CLASSES -1 : # check if all classes were attempted and completed
            logging.info("All classes collected successfully. Data collection finished.")
        elif not collection_fully_completed:
            logging.warning("Data collection was interrupted. Not all classes may be complete or fully collected.")
        else: # All attempted classes completed, but might not have started from 0 or finished all
            logging.info("Finished collecting for the specified range of classes.")


    except Exception as e_main:
        logging.critical(f"An unexpected error occurred in the main collection loop: {e_main}", exc_info=True)
    finally:
        logging.info("Releasing camera and destroying OpenCV windows...")
        if cap: cap.release()
        cv2.destroyAllWindows()
        logging.info("Cleanup complete. Script finished.")


if __name__ == "__main__":
    setup_logging(log_level=logging.INFO, log_file="collector_images.log")
    run_image_collection()
