
"""Dataset Creation for ASL Alphabet Recognition

This script processes collected images of ASL hand signs to create a dataset
suitable for training a machine learning model. It uses the MediaPipe library
to extract hand landmarks from each image.

Key Operations:
1.  Loads images from the `DATA_DIR`.
2.  For each image, detects hand landmarks using MediaPipe Hands.
3.  Normalizes these landmarks relative to the hand's bounding box to ensure
    consistency across different hand sizes and positions.
4.  Flattens the (x, y) coordinates of the 21 hand landmarks into a single array.
5.  Performs data augmentation by horizontally flipping each image and processing
    the flipped version as well, effectively doubling the dataset size.
6.  Saves the processed data (landmark arrays) and corresponding labels (sign names)
    into a pickle file named 'data_signs.pickle'.

Dependencies:
- OpenCV (cv2): For image loading and manipulation.
- MediaPipe: For hand landmark detection.
- NumPy: For numerical operations, especially array manipulation.
- tqdm: For displaying a progress bar during processing.

Constants:
- DATA_DIR (str): Path to the directory containing the collected sign language images,
                  organized into subdirectories named after each sign (e.g., 'A', 'B').
- EXPECTED_LANDMARK_LENGTH (int): The expected number of flattened coordinates from
                                 MediaPipe (21 landmarks * 2 coordinates = 42).
"""
import os
import pickle
import mediapipe as mp
import cv2
import numpy as np
import time
from tqdm import tqdm

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
# Configure Hands:
# - static_image_mode=True: Optimized for processing individual images (not a video stream).
# - min_detection_confidence=0.5: Minimum confidence value for hand detection to be considered successful.
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

DATA_DIR = './data'  # Directory where collected images are stored
data = []  # List to store processed landmark data
labels = []  # List to store corresponding labels (sign names)

# Each hand has 21 landmarks, each with (x, y) coordinates.
# So, the expected flattened length is 21 * 2 = 42.
EXPECTED_LANDMARK_LENGTH = 42

def process_image(img):
    """
    Processes a single image to extract, normalize, and flatten hand landmarks.

    Args:
        img (numpy.ndarray): The input image in BGR format (as loaded by OpenCV).

    Returns:
        list or None: A flattened list of normalized hand landmark coordinates (x, y)
                      if a hand is detected and landmarks meet the expected criteria.
                      Returns None if no hand is detected or if the landmark data
                      does not conform to the EXPECTED_LANDMARK_LENGTH.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB for MediaPipe
    results = hands.process(img_rgb)  # Process the image to detect hand landmarks

    if results.multi_hand_landmarks:
        # Assuming only one hand per image for ASL signs
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract (x, y) coordinates for each landmark
            coords = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark])

            # Normalize coordinates:
            # Subtract the minimum x and y coordinates of the hand's bounding box
            # from all landmark coordinates. This makes the landmarks relative
            # to the top-left corner of the hand, providing some translation invariance.
            x_min, y_min = coords.min(axis=0)
            coords[:, 0] -= x_min  # Normalize x-coordinates
            coords[:, 1] -= y_min  # Normalize y-coordinates

            # Flatten the array of (x, y) coordinates into a single list
            landmarks_flat = coords.flatten().tolist()

            # Ensure the processed landmarks meet the expected length
            if len(landmarks_flat) == EXPECTED_LANDMARK_LENGTH:
                return landmarks_flat
    return None  # Return None if no hand detected or landmarks are not as expected

# Record start time for performance measurement
start_time = time.time()

# Calculate the total number of images to process for the progress bar
total_images = 0
for root, dirs, files in os.walk(DATA_DIR):
    # Count only image files in relevant directories
    if any(os.path.join(root, d).startswith(os.path.join(DATA_DIR, chr(65+i))) for d in dirs for i in range(26)) or \
       os.path.basename(root) in [chr(65+i) for i in range(26)]: # Check current root too
        total_images += len([file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))])


# Initialize tqdm progress bar
with tqdm(total=total_images, desc="Processing images", dynamic_ncols=True) as pbar:
    # Iterate through each subdirectory in the DATA_DIR (each representing a class/label)
    for label in os.listdir(DATA_DIR):
        label_path = os.path.join(DATA_DIR, label)

        if os.path.isdir(label_path):  # Ensure it's a directory
            # Iterate through each image file in the class directory
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                # Check if the file is an image
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = cv2.imread(img_path)  # Load the image
                    if img is None:
                        print(f"Error loading image: {img_path}")
                        if total_images > 0 : pbar.update(1) # Still update progress if one image fails
                        continue

                    # Process the original image
                    landmarks = process_image(img)
                    if landmarks:
                        data.append(landmarks)
                        labels.append(label)

                    # Data Augmentation: Flip the image horizontally
                    img_flipped = cv2.flip(img, 1)  # 1 for horizontal flip
                    landmarks_flipped = process_image(img_flipped)
                    if landmarks_flipped:
                        data.append(landmarks_flipped)
                        labels.append(label)

                    pbar.update(1)  # Update progress bar after processing one original image (and its flip)

# Save the processed data and labels to a pickle file
if data and labels:
    with open('data_signs.pickle', 'wb') as f:
        pickle.dump({'data': np.array(data), 'labels': np.array(labels)}, f)
    print(f"\nProcessed {len(data)} items (original and flipped images).")
    print(f"Dataset saved to data_signs.pickle")
else:
    print("\nNo data was processed. Please check your data directory and image files.")

# Record end time and print total processing time
end_time = time.time()
print(f"Total processing time: {end_time - start_time:.2f} seconds.")
