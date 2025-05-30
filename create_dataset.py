"""
ASL Dataset Creator

This script processes images of American Sign Language (A-Z) hand signs from a dataset directory,
extracts hand landmarks using MediaPipe, augments the data by flipping images horizontally,
and saves the resulting dataset as a pickle file for machine learning training.

Usage:
    - Place your images in subfolders under the DATA_DIR, one folder per letter (A-Z).
    - Run this script to generate 'data_signs.pickle' containing landmarks and labels.

Dependencies:
    - OpenCV (cv2)
    - MediaPipe
    - NumPy
    - tqdm

Author: @ronal-lc
"""

import os
import pickle
import mediapipe as mp
import cv2
import numpy as np
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
# Import logging
# Import the setup_logging function

# Initialize MediaPipe Hands solution globally for potential re-use if safe,
# otherwise, it might need to be initialized per process if it's not picklable
# or has issues with multiprocessing. For this refactoring, we assume it can be
# initialized globally or passed/re-initialized in the worker function.
# For ProcessPoolExecutor, it's generally safer to initialize such objects
# within the worker function if they maintain internal state or are not picklable.
# mp_hands_solution = mp.solutions.hands

DATA_DIR = './data'
EXPECTED_LANDMARK_LENGTH = 42  # 21 hand landmarks, each with (x, y)
CHECKPOINT_FILE = 'dataset_checkpoint.pkl'
CHECKPOINT_INTERVAL = 100 # Save checkpoint after every N original images processed

# Global hands object for the main process (e.g. for single-threaded fallback or utilities)
# This might not be used by worker processes if they initialize their own.
# hands_main_process = mp_hands_solution.Hands(static_image_mode=True, min_detection_confidence=0.5)


def initialize_worker():
    """Initializer for each worker process. Sets up MediaPipe Hands."""
    global hands_worker
    try:
        hands_worker = mp.solutions.hands.Hands(static_image_mode=True, min_detection_confidence=0.5)
        print(f"DEBUG: Worker process {os.getpid()} initialized MediaPipe Hands successfully.")
    except Exception as e:
        print(f"ERROR: Worker process {os.getpid()} failed to initialize MediaPipe Hands: {e}")
        hands_worker = None # Ensure it's None if initialization fails

def process_single_image_entry(image_path_tuple):
    """
    Processes a single image file (original and its flip for augmentation).
    This function is designed to be run in a worker process.
    Initializes MediaPipe Hands within the worker if not using an initializer.
    """
    # If not using ProcessPoolExecutor's initializer, initialize MediaPipe Hands here:
    # hands_local = mp.solutions.hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

    global hands_worker # Uses the hands object initialized by initialize_worker

    if hands_worker is None: # Check if worker initialization failed
        print(f"ERROR: Skipping processing for {image_path_tuple[0]} in worker {os.getpid()} due to MediaPipe initialization failure.")
        return image_path_tuple[0], []

    image_path, label_name = image_path_tuple
    landmarks_results = [] 
    print(f"DEBUG: Processing image: {image_path} for label: {label_name} in worker {os.getpid()}")

    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"WARNING: Failed to load image: {image_path}. Skipping.")
            return image_path, [] 

        # Process original image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results_original = hands_worker.process(img_rgb) # MediaPipe process
        if results_original.multi_hand_landmarks:
            landmarks_list = []
            # The rest of the landmark extraction logic seems fine.
            # Adding detailed logging for landmark extraction can be very verbose,
            # so we'll keep it high-level unless specific errors occur.
            for hand_landmarks in results_original.multi_hand_landmarks:
                coords = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark])
                x_min, y_min = coords.min(axis=0)
                coords[:, 0] -= x_min
                coords[:, 1] -= y_min
                landmarks_list.extend(coords.flatten())
            if len(landmarks_list) == EXPECTED_LANDMARK_LENGTH:
                landmarks_results.append({'landmarks': landmarks_list, 'label': label_name, 'source': 'original'})
                print(f"DEBUG: Extracted original landmarks for {image_path}")
            elif results_original.multi_hand_landmarks: # Log if landmarks were found but length was wrong
                print(f"WARNING: Found original hand landmarks for {image_path}, but length {len(landmarks_list)} != {EXPECTED_LANDMARK_LENGTH}.")


        # Process horizontally flipped image (augmentation)
        img_flipped = cv2.flip(img, 1)
        img_flipped_rgb = cv2.cvtColor(img_flipped, cv2.COLOR_BGR2RGB)
        results_flipped = hands_worker.process(img_flipped_rgb) # MediaPipe process
        if results_flipped.multi_hand_landmarks:
            landmarks_list_flipped = []
            for hand_landmarks in results_flipped.multi_hand_landmarks:
                coords = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark])
                x_min, y_min = coords.min(axis=0)
                coords[:, 0] -= x_min
                coords[:, 1] -= y_min
                landmarks_list_flipped.extend(coords.flatten())
            if len(landmarks_list_flipped) == EXPECTED_LANDMARK_LENGTH:
                 landmarks_results.append({'landmarks': landmarks_list_flipped, 'label': label_name, 'source': 'flipped'})
                 print(f"DEBUG: Extracted flipped landmarks for {image_path}")
            elif results_flipped.multi_hand_landmarks: # Log if landmarks were found but length was wrong
                 print(f"WARNING: Found flipped hand landmarks for {image_path}, but length {len(landmarks_list_flipped)} != {EXPECTED_LANDMARK_LENGTH}.")
        
        return image_path, landmarks_results
    except cv2.error as e_cv2: # Specific OpenCV errors
        print(f"ERROR: OpenCV error processing {image_path}: {e_cv2}")
        return image_path, [] 
    except Exception as e_generic: # Other errors (e.g., MediaPipe, NumPy)
        print(f"ERROR: Generic error processing {image_path} in worker {os.getpid()}: {e_generic}")
        return image_path, [] 

def load_checkpoint():
    """Loads data from the checkpoint file if it exists."""
    if os.path.exists(CHECKPOINT_FILE):
        print(f"INFO: Checkpoint file '{CHECKPOINT_FILE}' found. Attempting to load.")
        try:
            with open(CHECKPOINT_FILE, 'rb') as f:
                checkpoint_data = pickle.load(f)
            print("INFO: Checkpoint loaded successfully.")
            return checkpoint_data.get('data', []), checkpoint_data.get('labels', []), checkpoint_data.get('processed_files', set())
        except Exception as e:
            print(f"ERROR: Error loading checkpoint file '{CHECKPOINT_FILE}': {e}. Starting from scratch.")
            return [], [], set()
    else:
        print(f"INFO: No checkpoint file '{CHECKPOINT_FILE}' found. Starting from scratch.")
        return [], [], set()

def save_checkpoint(data, labels, processed_files):
    """Saves the current state to the checkpoint file."""
    print(f"INFO: Saving checkpoint to '{CHECKPOINT_FILE}' with {len(processed_files)} processed files.")
    checkpoint_data = {
        'data': data, # This can be large, consider if full data save is needed every time
        'labels': labels, # Same as above
        'processed_files': processed_files # This is the most critical part for resuming
    }
    try:
        with open(CHECKPOINT_FILE, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        print(f"INFO: Checkpoint saved successfully to '{CHECKPOINT_FILE}'.")
    except Exception as e:
        print(f"ERROR: Failed to save checkpoint to '{CHECKPOINT_FILE}': {e}")


def create_dataset_main():
    """Main function to create the dataset using parallel processing and checkpointing."""
    print("INFO: Starting dataset creation process.")
    data, labels, processed_files = load_checkpoint()
    
    initial_processed_count = len(processed_files)
    print(f"INFO: Resuming from checkpoint. Initially {initial_processed_count} files were processed.")
    
    start_time = time.time()

    # Discover all image files
    all_image_paths_with_labels = []
    if not os.path.exists(DATA_DIR):
        print(f"CRITICAL: Data directory '{DATA_DIR}' does not exist. Cannot proceed.")
        return
        
    for label_name in sorted(os.listdir(DATA_DIR)): 
        label_path = os.path.join(DATA_DIR, label_name)
        if os.path.isdir(label_path):
            image_files = glob.glob(os.path.join(label_path, '*.jpg')) + \
                          glob.glob(os.path.join(label_path, '*.jpeg')) + \
                          glob.glob(os.path.join(label_path, '*.png'))
            for img_path in sorted(image_files): 
                 all_image_paths_with_labels.append((img_path, label_name))
    print(f"INFO: Discovered a total of {len(all_image_paths_with_labels)} images across all class directories.")

    # Filter out already processed files
    image_tasks_to_process = [
        item for item in all_image_paths_with_labels if item[0] not in processed_files
    ]
    
    if not image_tasks_to_process:
        print("INFO: No new images to process. Dataset is up-to-date based on checkpoint.")
    else:
        print(f"INFO: Found {len(image_tasks_to_process)} new images to process out of {len(all_image_paths_with_labels)} total discovered images.")

    num_workers = os.cpu_count() or 1 
    print(f"INFO: Using {num_workers} worker processes for image processing.")

    processed_count_since_last_checkpoint = 0
    newly_processed_files_count = 0

    # Using try-except-finally to ensure resources are cleaned up if possible
    # Note: ProcessPoolExecutor itself handles worker shutdown on exit when used as a context manager.
    try:
        with ProcessPoolExecutor(max_workers=num_workers, initializer=initialize_worker) as executor:
            futures = {executor.submit(process_single_image_entry, img_task): img_task for img_task in image_tasks_to_process}

            with tqdm(total=len(image_tasks_to_process), desc="Processing images", dynamic_ncols=True) as pbar:
                for future in as_completed(futures):
                    original_image_path, landmarks_results_list = future.result()
                    
                    if landmarks_results_list: # Only add if results were actually produced
                        for result_entry in landmarks_results_list:
                            data.append(result_entry['landmarks'])
                            labels.append(result_entry['label'])
                        print(f"DEBUG: Successfully processed and added landmarks for {original_image_path}.")
                    else:
                        print(f"WARNING: No landmarks extracted or error processing {original_image_path}. It will be marked as processed.")
                    
                    processed_files.add(original_image_path) 
                    newly_processed_files_count += 1
                    pbar.update(1)
                    processed_count_since_last_checkpoint +=1

                    if processed_count_since_last_checkpoint >= CHECKPOINT_INTERVAL:
                        save_checkpoint(data, labels, processed_files)
                        processed_count_since_last_checkpoint = 0
        
        # Final save of checkpoint after loop finishes
        if processed_count_since_last_checkpoint > 0 or newly_processed_files_count > 0 : # Save if any new work was done
            save_checkpoint(data, labels, processed_files)
        print(f"INFO: Completed processing {newly_processed_files_count} new images.")

    except Exception as e_executor:
        print(f"CRITICAL: An error occurred during parallel processing: {e_executor}")
        # Data up to the last successful checkpoint is preserved.
        # Consider if a final checkpoint save attempt is useful here, though data might be inconsistent.

    # Save the final complete dataset
    if data and labels:
        print(f"INFO: Saving final dataset to 'data_signs.pickle'. Total landmark entries: {len(data)}.")
        try:
            with open('data_signs.pickle', 'wb') as f:
                pickle.dump({'data': np.array(data), 'labels': np.array(labels)}, f)
            print(f"INFO: Final dataset 'data_signs.pickle' saved successfully.")
            
            # Optional: remove checkpoint file after successful completion
            if os.path.exists(CHECKPOINT_FILE) and newly_processed_files_count == len(image_tasks_to_process) and len(image_tasks_to_process) > 0:
                 # Only remove if all tasks were processed successfully in this run
                 # os.remove(CHECKPOINT_FILE) 
                 # print(f"INFO: Checkpoint file '{CHECKPOINT_FILE}' removed after successful completion.")
                 print(f"INFO: Checkpoint file '{CHECKPOINT_FILE}' retained. Remove manually if not needed for future resumption.")
            elif os.path.exists(CHECKPOINT_FILE):
                 print(f"INFO: Checkpoint file '{CHECKPOINT_FILE}' retained as not all tasks may have been processed in this run or it was already up-to-date.")

        except Exception as e_pickle_save:
            print(f"CRITICAL: Failed to save the final dataset to 'data_signs.pickle': {e_pickle_save}")
    else:
        print("WARNING: No data was processed or loaded. Final dataset 'data_signs.pickle' was not created.")

    end_time = time.time()
    print(f"INFO: Total processing time for this run: {end_time - start_time:.2f} seconds.")
    print("INFO: Dataset creation process finished.")


if __name__ == "__main__":
    # setup_logging(log_level=logging.INFO, log_file="create_dataset.log") # Removed
    
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print(f"ERROR: Data directory '{DATA_DIR}' is missing or empty.")
        print("ERROR: Please populate it with image subfolders (A, B, C, etc.) before running.")
    else:
        create_dataset_main()
