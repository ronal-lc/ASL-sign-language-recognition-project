"""
ASL Alphabet Real-Time Recognition

This script provides a GUI for real-time American Sign Language (ASL) hand sign recognition using a webcam.
It uses MediaPipe for hand landmark detection and a trained Keras model for classification.
Includes a learning mode for practice.

Usage:
    - Ensure 'model.keras' exists (trained model).
    - Place alphabet images in the 'alfabeto' folder for learning mode.
    - Run this script to start the GUI.

Dependencies:
    - OpenCV
    - MediaPipe
    - NumPy
    - Pillow
    - TensorFlow
    - tkinter

Author: @ronal-lc
"""

import os
import random
import time
import tkinter as tk
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk
from PIL.Image import Resampling
from tensorflow.keras.models import load_model 

WINDOW_NAME = 'ASL Alphabet'
ICON_PATH = 'icono.ico'

# Load trained model
model = load_model('./model.keras')

# Open webcam
for camera_index in range(3):
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        break
else:
    exit(1)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.9
)

# Label dictionary (A-Z + STOP)
labels_dict = {i: chr(65 + i) for i in range(26)}
labels_dict[26] = "STOP"

# State variables
learning_label = None
tk_image = None
continuous_detection_start = 0
detected_phrase = ""
last_detected_letter = ""
current_letter = ""
last_detection_time = 0
stop_detection_time = 0
stop_detected = False
text_color = "black"
learning_mode = False
learning_window = None
current_learning_image = None
current_learning_letter = ""
alphabet_path = "alphabet_examples"

# --- GUI Components ---

def toggle_switch(parent, text, variable, command=None):
    """Custom toggle switch for tkinter."""
    switch_frame = tk.Frame(parent, bg="#f0f0f0")
    label = tk.Label(switch_frame, text=text, font=("Helvetica", 12), bg="#f0f0f0")
    label.pack(side=tk.LEFT, padx=(0, 10))
    toggle = tk.Canvas(switch_frame, width=50, height=25, bg="#f0f0f0", highlightthickness=0)
    toggle.pack(side=tk.RIGHT)

    def redraw_toggle():
        toggle.delete("all")
        if variable.get():
            toggle.create_rectangle(0, 0, 50, 25, fill="#4CAF50", outline="")
            toggle.create_oval(30, 5, 45, 20, fill="#FFFFFF", outline="")
        else:
            toggle.create_rectangle(0, 0, 50, 25, fill="#CCCCCC", outline="")
            toggle.create_oval(5, 5, 20, 20, fill="#FFFFFF", outline="")

    def on_toggle(_event=None):
        variable.set(not variable.get())
        redraw_toggle()
        if command:
            command()

    toggle.bind("<Button-1>", on_toggle)
    redraw_toggle()
    return switch_frame

def on_learning_window_close():
    """Handle closing of the learning mode window."""
    global learning_window
    if learning_window is not None and isinstance(learning_window, tk.Toplevel):
        learning_window.destroy()
        learning_window = None
        learning_var.set(False)
        redraw_learning_toggle()

def redraw_learning_toggle():
    """Redraw the learning mode toggle switch."""
    canvas = learning_toggle.children.get("!canvas")
    if isinstance(canvas, tk.Canvas):
        canvas.delete("all")
        if learning_var.get():
            canvas.create_rectangle(0, 0, 50, 25, fill="#4CAF50", outline="")
            canvas.create_oval(30, 5, 45, 20, fill="#FFFFFF", outline="")
        else:
            canvas.create_rectangle(0, 0, 50, 25, fill="#CCCCCC", outline="")
            canvas.create_oval(5, 5, 20, 20, fill="#FFFFFF", outline="")

def toggle_learning_mode():
    """Enable or disable learning mode."""
    global learning_window, last_detection_time, current_learning_letter
    if learning_var.get():
        if learning_window is None or not tk.Toplevel.winfo_exists(learning_window):
            learning_window = tk.Toplevel(root)
            learning_window.title("Learning Mode")
            learning_window.geometry("350x350")
            learning_window.resizable(False, False)
            learning_window.configure(bg="#f0f0f0")
            learning_window.iconbitmap(ICON_PATH)
            global learning_label
            learning_label = tk.Label(learning_window, bg="#f0f0f0")
            learning_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
            load_random_image()
            last_detection_time = time.time()
            current_learning_letter = ""
            learning_window.protocol("WM_DELETE_WINDOW", on_learning_window_close)
    else:
        if learning_window is not None:
            learning_window.destroy()
            learning_window = None
            last_detection_time = 0
            current_learning_letter = ""

def load_random_image(exclude_letter=None):
    """Load a random letter image for learning mode."""
    global current_learning_image, current_learning_letter
    try:
        images = [f for f in os.listdir(alphabet_path) if os.path.isfile(os.path.join(alphabet_path, f))]
        if not images:
            return
        if exclude_letter:
            images = [img for img in images if os.path.splitext(img)[0].upper() != exclude_letter]
        if not images:
            return
        selected_image = random.choice(images)
        image_path = os.path.join(alphabet_path, selected_image)
        img = Image.open(image_path)
        img = img.resize((300, 300), Resampling.LANCZOS)
        current_learning_image = ImageTk.PhotoImage(img)
        learning_label.config(image=current_learning_image)
        current_learning_letter = os.path.splitext(selected_image)[0].upper()
    except Exception as e:
        print(f"Error loading image: {e}")

# --- Main GUI setup ---

root = tk.Tk()
root.title(WINDOW_NAME)
root.iconbitmap(ICON_PATH)
root.configure(bg="#f0f0f0")
root.geometry("900x700")
root.resizable(False, False)

canvas = tk.Canvas(root, width=800, height=600, bg="#ffffff", highlightthickness=0)
canvas.pack(pady=20)

reset_image = Image.open("reset.png")
reset_image = reset_image.resize((50, 50), Resampling.LANCZOS)
reset_photo = ImageTk.PhotoImage(reset_image)

def reset_detected_text():
    """Reset detected phrase and state."""
    global detected_phrase, stop_detected, text_color
    detected_phrase = ""
    stop_detected = False
    text_color = "black"

learning_var = tk.BooleanVar()
learning_toggle = toggle_switch(root, "Learning Mode", learning_var, toggle_learning_mode)
learning_toggle.place(x=20, y=630)

reset_button = tk.Button(
    root,
    image=reset_photo,
    command=reset_detected_text,
    bg="#ffffff",
    activebackground="#d9534f",
    borderwidth=3,
    relief="raised",
    highlightthickness=0,
    cursor="hand2"
)
reset_button.image = reset_photo
reset_button.place(x=750, y=630)

# --- Main Loop ---

def update_frame():
    """Main loop: capture frame, process hand, predict letter, update GUI."""
    global tk_image, detected_phrase, last_detected_letter, current_letter, last_detection_time
    global stop_detection_time, stop_detected, text_color, current_learning_letter, continuous_detection_start

    ret, frame = cap.read()
    if not ret:
        print("Error capturing camera frame.")
        root.after(10, update_frame)
        return

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if 'continuous_detection_start' not in globals():
        continuous_detection_start = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            x_, y_ = [], []
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)
            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(x_[i] - min(x_))
                data_aux.append(y_[i] - min(y_))
            data_aux = data_aux[:42]
            prediction = model.predict(np.array([data_aux]), verbose=0)
            max_prob = np.max(prediction[0])
            predicted_index = np.argmax(prediction[0])
            predicted_character = labels_dict[predicted_index] if max_prob >= 0.9 else '?'
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            cv2.putText(frame, f'{predicted_character} ({max_prob * 100:.2f}%)', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)
            current_time = time.time()

            # STOP detection logic
            if predicted_character == "STOP" and max_prob >= 0.99:
                if stop_detection_time == 0:
                    stop_detection_time = current_time
                elif current_time - stop_detection_time >= 2:
                    stop_detected = True
                    text_color = "red"
                    print("STOP detected. Blocking further detections.")
            else:
                stop_detection_time = 0

            # Normal and learning mode logic
            if not stop_detected:
                if predicted_character != "STOP" and max_prob >= 0.99:
                    if learning_var.get():
                        if not current_learning_letter:
                            load_random_image()
                        if predicted_character == current_learning_letter:
                            if continuous_detection_start == 0:
                                continuous_detection_start = current_time
                            if current_time - continuous_detection_start >= 3:
                                detected_phrase += predicted_character
                                last_detected_letter = predicted_character
                                last_detection_time = current_time
                                print(f"Stored letter: {predicted_character}")
                                load_random_image(exclude_letter=current_learning_letter)
                                current_learning_letter = ""
                                continuous_detection_start = 0
                        else:
                            continuous_detection_start = 0
                    else:
                        if current_time - last_detection_time >= 1:
                            detected_phrase += predicted_character
                            last_detected_letter = predicted_character
                            last_detection_time = current_time
                            print(f"Stored letter: {predicted_character}")
            else:
                print("STOP active. No more letters can be added.")

    # Draw footer and update GUI
    frame_with_footer = np.zeros((H + 90, W, 3), dtype=np.uint8)
    frame_with_footer[:H, :] = frame
    footer_color = (211, 211, 211)
    if stop_detected:
        footer_color = (0, 0, 128)
    frame_with_footer[H:] = footer_color

    cv2.putText(frame_with_footer, f'Detected: {detected_phrase}', (20, H + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    img = Image.fromarray(cv2.cvtColor(frame_with_footer, cv2.COLOR_BGR2RGB))
    tk_image = ImageTk.PhotoImage(image=img)
    canvas.create_image(canvas.winfo_width() // 2, canvas.winfo_height() // 2, anchor=tk.CENTER, image=tk_image)

    root.after(10, update_frame)

update_frame()
root.mainloop()
cap.release()
cv2.destroy_all_windows()
hands.close()
