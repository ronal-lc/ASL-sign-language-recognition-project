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

DATA_DIR = './data'
NUMBER_OF_CLASSES = 26  # 26 letters (A-Z)
DATASET_SIZE = 100
CAMERA_INDICES = [0, 1, 2]
WINDOW_NAME = 'ASL Alphabet Image Collector'


def ensure_dir(path):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def select_camera(indices):
    """Try to open a camera from the provided indices."""
    for idx in indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"Camera opened successfully with index {idx}")
            return cap
        cap.release()
    print("Could not open any camera. Please check your connection.")
    sys.exit(1)


def setup_window(window_name):
    """Set up the OpenCV window."""
    cv2.namedWindow(window_name)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)


def capture_initial_ready(cap, window_name):
    """
    Display a ready screen before starting image capture.
    Returns True if user presses 'Q' to start, False if window is closed.
    """
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error capturing frame. Retrying...")
            continue

        cv2.putText(frame, 'Ready? Press "Q" to start', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(25) & 0xFF

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            return False

        if key == ord('q'):
            break
    return True


def capture_images(cap, class_dir, dataset_size, window_name):
    """
    Capture and save images for the current class.
    Returns True if completed, False if window is closed.
    """
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"Error capturing frame {counter}. Retrying...")
            continue

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            return False

        img_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(img_path, frame)
        counter += 1
        print(f"\rProgress: {counter}/{dataset_size}", end="", flush=True)
    print()
    return True


def run_image_collection():
    """Main function to run the image collection process."""
    ensure_dir(DATA_DIR)
    cap = select_camera(CAMERA_INDICES)
    setup_window(WINDOW_NAME)

    start_class = input("Start collecting from class 0? (y/n): ").strip().lower()

    if start_class == 'n':
        while True:
            try:
                start_class = int(input(f"Enter the class number to start from (0-{NUMBER_OF_CLASSES - 1}): "))
                if 0 <= start_class < NUMBER_OF_CLASSES:
                    break
                else:
                    print(f"Please enter a number between 0 and {NUMBER_OF_CLASSES - 1}.")
            except ValueError:
                print("Please enter a valid number.")
    else:
        start_class = 0

    try:
        for cls in range(start_class, NUMBER_OF_CLASSES):
            class_name = chr(65 + cls)  # Convert index to letter (A-Z)
            class_dir = os.path.join(DATA_DIR, class_name)
            ensure_dir(class_dir)
            print(f'\nCollecting data for class {class_name}')

            if not capture_initial_ready(cap, WINDOW_NAME):
                print("\nData collection interrupted.")
                break

            if not capture_images(cap, class_dir, DATASET_SIZE, WINDOW_NAME):
                print("\nData collection interrupted.")
                break

            print(f"Class {class_name} completed.")
        else:
            print("\nData collection finished.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_image_collection()
