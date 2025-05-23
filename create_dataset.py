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

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

DATA_DIR = './data'
data = []
labels = []

EXPECTED_LANDMARK_LENGTH = 42  # 21 hand landmarks, each with (x, y)

def process_image(img):
    """Extract normalized hand landmarks from an image using MediaPipe."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            coords = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark])
            x_min, y_min = coords.min(axis=0)
            coords[:, 0] -= x_min
            coords[:, 1] -= y_min
            landmarks.extend(coords.flatten())
        if len(landmarks) == EXPECTED_LANDMARK_LENGTH:
            return landmarks
    return None

start_time = time.time()
total_images = sum(
    len([file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))])
    for r, d, files in os.walk(DATA_DIR)
)

with tqdm(total=total_images, desc="Processing images", dynamic_ncols=True) as pbar:
    for label in os.listdir(DATA_DIR):
        label_path = os.path.join(DATA_DIR, label)
        if os.path.isdir(label_path):
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Error loading image: {img_path}")
                        continue

                    # Process original image
                    landmarks = process_image(img)
                    if landmarks:
                        data.append(landmarks)
                        labels.append(label)

                    # Process horizontally flipped image (augmentation)
                    img_flipped = cv2.flip(img, 1)
                    landmarks_flipped = process_image(img_flipped)
                    if landmarks_flipped:
                        data.append(landmarks_flipped)
                        labels.append(label)

                    pbar.update(1)

# Save dataset if data is available
if data and labels:
    with open('data_signs.pickle', 'wb') as f:
        pickle.dump({'data': np.array(data), 'labels': np.array(labels)}, f)
    print(f"\nProcessed {len(data)} images (including flipped).")
else:
    print("\nNo data processed. Please check your data directory and image files.")

end_time = time.time()
print(f"Total processing time: {end_time - start_time:.2f} seconds.")
