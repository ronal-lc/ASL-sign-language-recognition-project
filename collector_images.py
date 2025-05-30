"""
Image Collector for ASL Alphabet Training Data

This script facilitates the collection of images for training an American Sign Language (ASL)
alphabet recognizer. It captures images from a webcam for each letter of the ASL alphabet (A-Z).

Main Functionality:
- Iterates through each class (letter A-Z).
- Prompts the user to prepare for capturing images for the current class.
- Captures a predefined number of images (`DATASET_SIZE`) for each class.
- Saves images in a structured directory format: `DATA_DIR/CLASS_NAME/IMAGE_NUMBER.jpg`.

Important Constants:
- DATA_DIR: The root directory where collected images will be stored.
- NUMBER_OF_CLASSES: The total number of classes to collect (26 for A-Z).
- DATASET_SIZE: The number of images to capture for each class.
- CAMERA_INDICES: A list of camera indices to try if the default camera is not available.
- WINDOW_NAME: The name of the OpenCV window used for displaying the camera feed.
"""
import os
import sys
import cv2

DATA_DIR = './data'
NUMBER_OF_CLASSES = 26  # 26 letters (A-Z)
DATASET_SIZE = 200
CAMERA_INDICES = [0, 1, 2]  # List of camera indices to try
WINDOW_NAME = 'Image Collector'


def ensure_dir(path):
    """
    Creates a directory if it does not already exist.

    Args:
        path (str): The path of the directory to create.
    """
    os.makedirs(path, exist_ok=True)


def select_camera(indices):
    """
    Tries to open a camera from a list of specified indices.

    Args:
        indices (list of int): A list of camera indices to attempt to open.

    Returns:
        cv2.VideoCapture: The opened camera object.

    Exits:
        If no camera can be opened from the provided indices.
    """
    for idx in indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"Successfully opened camera with index {idx}")
            return cap
        cap.release()
    print("Could not open any camera. Please check your connection.")
    sys.exit(1)


def setup_window(window_name):
    """
    Creates and configures the OpenCV window for displaying the camera feed.
    Sets the window to be always on top.

    Args:
        window_name (str): The name for the OpenCV window.
    """
    cv2.namedWindow(window_name)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)


def capture_initial_ready(cap, window_name):
    """
    Waits for the user to signal readiness before starting image capture for a class.
    Displays "Ready? Press 'Q' to start" on the camera feed. The user presses 'Q' to proceed.

    Args:
        cap (cv2.VideoCapture): The camera object.
        window_name (str): The name of the OpenCV window.

    Returns:
        bool: True if the user signals readiness, False if the window is closed or an error occurs.
    """
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error capturing frame. Retrying...")
            continue

        cv2.putText(frame, "Ready? Press 'Q' to start", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(25) & 0xFF

        # Check if the window was closed by the user
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            return False

        if key == ord('q'):  # Start capture if 'q' is pressed
            break
    return True


def capture_images(cap, class_dir, dataset_size, window_name):
    """
    Captures a specified number of images for a given class.
    Images are saved into the specified class directory. Displays progress on the console.

    Args:
        cap (cv2.VideoCapture): The camera object.
        class_dir (str): The directory where images for the current class will be saved.
        dataset_size (int): The total number of images to capture for this class.
        window_name (str): The name of the OpenCV window.

    Returns:
        bool: True if all images are captured successfully, False if the window is closed
              or an error occurs during capture.
    """
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"Error capturing frame {counter}. Retrying...")
            continue

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF # Use waitKey(1) for smoother video display

        # Check if the window was closed by the user
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            return False

        # Save the captured frame
        img_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(img_path, frame)
        counter += 1
        print(f"\rProgress: {counter}/{dataset_size}", end="", flush=True)
    print()  # Newline after progress is complete
    return True


def run_image_collection():
    """
    Main function to orchestrate the image collection process.
    It handles:
    - Ensuring the main data directory exists.
    - Selecting and setting up the camera and display window.
    - Prompting the user for a starting class if they don't want to start from 'A'.
    - Iterating through each class (A-Z).
    - Calling helper functions to manage user readiness and image capture for each class.
    - Handling cleanup of camera resources.
    """
    ensure_dir(DATA_DIR)
    cap = select_camera(CAMERA_INDICES)
    setup_window(WINDOW_NAME)

    # Ask the user if they want to start from class 0 (A) or a different class
    start_from_specific_class = input("Do you want to start collection from class 0 (A)? (y/n): ").strip().lower()

    start_class_index = 0
    if start_from_specific_class == 'n':
        while True:
            try:
                # Prompt for the class number to start from
                start_class_index = int(input(f"Enter the class number to start from (0-{NUMBER_OF_CLASSES - 1}): "))
                if 0 <= start_class_index < NUMBER_OF_CLASSES:
                    break
                else:
                    print(f"Please enter a number between 0 and {NUMBER_OF_CLASSES - 1}.")
            except ValueError:
                print("Please enter a valid number.")
    
    try:
        # Loop through each class, starting from the specified index
        for cls_index in range(start_class_index, NUMBER_OF_CLASSES):
            class_name = chr(65 + cls_index)  # Convert index to letter (0-25 -> A-Z)
            class_dir_path = os.path.join(DATA_DIR, class_name)
            ensure_dir(class_dir_path)
            print(f'\nCollecting data for class: {class_name}')

            # Wait for user to be ready
            if not capture_initial_ready(cap, WINDOW_NAME):
                print("\nData collection interrupted by user (window closed).")
                break  # Exit the loop if user closes window

            # Capture images for the current class
            if not capture_images(cap, class_dir_path, DATASET_SIZE, WINDOW_NAME):
                print("\nData collection interrupted by user (window closed).")
                break  # Exit the loop if user closes window

            print(f"Class {class_name} completed.")
        else:
            # This 'else' block executes if the loop completes without a 'break'
            print("\nData collection finished for all specified classes.")
    finally:
        # Release camera and destroy OpenCV windows
        print("Releasing camera and closing windows.")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    run_image_collection()
