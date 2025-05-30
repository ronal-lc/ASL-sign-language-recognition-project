"""ASL Alphabet Real-Time Recognition

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

import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import random
import time
import cv2
import mediapipe as mp # Moved to global scope
import numpy as np
# from tensorflow.keras.models import load_model # Moved to CameraProcessingThread

from PySide6.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                               QVBoxLayout, QHBoxLayout, QWidget, QTabWidget,
                               QComboBox, QSlider, QGroupBox, QFormLayout, QSizePolicy) # Added QSizePolicy
from PySide6.QtGui import QImage, QPixmap, QIcon, QFont
from PySide6.QtCore import QTimer, Qt, QSize, Slot, QSettings

# Import PIL.Image explicitly for image loading in learning mode
from PIL import Image

from PySide6.QtCore import QThread, Signal # Added for QThread

WINDOW_NAME = 'ASL Alphabet Recognition'
ICON_PATH = 'icono.ico'
ALPHABET_PATH = "alphabet_examples"
MODEL_PATH = './model.keras' # Added for model path

# --- Theme Stylesheets --- (Can be moved to separate files if they grow large)
LIGHT_THEME_STYLESHEET = """
    QMainWindow {
        background-color: #f0f0f0;
    }
    QTabWidget::pane {
        border-top: 1px solid #c2c7cb;
    }
    QTabBar::tab {
        background: #e1e1e1;
        border: 1px solid #c2c7cb;
        border-bottom-color: #c2c7cb; 
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        min-width: 8ex;
        padding: 5px;
        color: #333;
    }
    QTabBar::tab:selected {
        background: #f0f0f0;
        border-color: #c2c7cb;
        border-bottom-color: #f0f0f0; /* Make selected tab blend with pane */
        color: #000;
    }
    QTabBar::tab:!selected:hover {
        background: #dcdcdc;
    }
    QWidget { /* Default for widgets inside tabs if not overridden */
        background-color: #f0f0f0;
        color: #333333;
    }
    QLabel {
        color: #333333;
    }
    QPushButton {
        background-color: #d9534f;
        color: white;
        border-radius: 5px;
        padding: 8px 15px;
        font-size: 12px; /* Base size, will be scaled */
    }
    QPushButton:hover {
        background-color: #c9302c;
    }
    QPushButton:pressed {
        background-color: #ac2925;
    }
    QComboBox {
        border: 1px solid #c2c7cb;
        border-radius: 3px;
        padding: 3px 18px 3px 5px;
        min-width: 6em;
        background-color: white;
        color: #333;
    }
    QComboBox:editable {
        background: white;
    }
    QComboBox:!editable, QComboBox::drop-down:editable {
         background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                     stop: 0 #E1E1E1, stop: 0.4 #DDDDDD,
                                     stop: 0.5 #D8D8D8, stop: 1.0 #D3D3D3);
    }
    QComboBox:!editable:on, QComboBox::drop-down:editable:on {
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                    stop: 0 #D3D3D3, stop: 0.4 #D8D8D8,
                                    stop: 0.5 #DDDDDD, stop: 1.0 #E1E1E1);
    }
    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 15px;
        border-left-width: 1px;
        border-left-color: darkgray;
        border-left-style: solid;
        border-top-right-radius: 3px;
        border-bottom-right-radius: 3px;
    }
    QComboBox::down-arrow {
        image: url(down_arrow.png); /* Needs a down_arrow.png or use a unicode char */
    }
    QGroupBox {
        background-color: #e8e8e8;
        border: 1px solid #c2c7cb;
        border-radius: 5px;
        margin-top: 1ex; /* leave space at the top for the title */
        font-weight: bold;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left; /* position at the top left */
        padding: 0 3px;
        color: #333;
    }
    /* Specific label styling if needed, e.g., for video_label border */
    #VideoLabel { /* Use objectName for specific styling */
        background-color: black;
        border: 1px solid #CCCCCC;
    }
    #LearningImageLabel {
        background-color: transparent;
        border: none;
        font-size: 16px; /* Base size */
        color: #555;
    }
    #PhraseLabel {
        font-size: 16px; /* Base size */
        background-color: #e0e0e0; 
        color: black; 
        padding: 5px; 
        border-radius: 5px;
    }
"""

DARK_THEME_STYLESHEET = """
    QMainWindow {
        background-color: #2e2e2e;
    }
    QTabWidget::pane {
        border-top: 1px solid #4a4a4a;
    }
    QTabBar::tab {
        background: #3c3c3c;
        border: 1px solid #4a4a4a;
        border-bottom-color: #4a4a4a;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        min-width: 8ex;
        padding: 5px;
        color: #cfcfcf;
    }
    QTabBar::tab:selected {
        background: #2e2e2e;
        border-color: #4a4a4a;
        border-bottom-color: #2e2e2e;
        color: #ffffff;
    }
    QTabBar::tab:!selected:hover {
        background: #484848;
    }
    QWidget {
        background-color: #2e2e2e;
        color: #cfcfcf;
    }
    QLabel {
        color: #cfcfcf;
    }
    QPushButton {
        background-color: #d9534f; /* Keep button color for visibility, or adjust */
        color: white;
        border-radius: 5px;
        padding: 8px 15px;
        font-size: 12px; /* Base size */
    }
    QPushButton:hover {
        background-color: #c9302c;
    }
    QPushButton:pressed {
        background-color: #ac2925;
    }
    QComboBox {
        border: 1px solid #4a4a4a;
        border-radius: 3px;
        padding: 3px 18px 3px 5px;
        min-width: 6em;
        background-color: #3c3c3c;
        color: #cfcfcf;
        selection-background-color: #585858;
    }
    QComboBox:editable {
        background: #3c3c3c;
    }
    QComboBox:!editable, QComboBox::drop-down:editable {
         background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                     stop: 0 #484848, stop: 0.4 #404040,
                                     stop: 0.5 #3c3c3c, stop: 1.0 #383838);
    }
    QComboBox:!editable:on, QComboBox::drop-down:editable:on {
        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                    stop: 0 #383838, stop: 0.4 #3c3c3c,
                                    stop: 0.5 #404040, stop: 1.0 #484848);
    }
    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 15px;
        border-left-width: 1px;
        border-left-color: #4a4a4a;
        border-left-style: solid;
        border-top-right-radius: 3px;
        border-bottom-right-radius: 3px;
    }
    QComboBox::down-arrow {
        image: url(down_arrow_dark.png); /* Needs a dark-theme compatible arrow */
    }
    QGroupBox {
        background-color: #3c3c3c;
        border: 1px solid #4a4a4a;
        border-radius: 5px;
        margin-top: 1ex;
        font-weight: bold;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 3px;
        color: #cfcfcf;
    }
    #VideoLabel {
        background-color: black; 
        border: 1px solid #4a4a4a;
    }
    #LearningImageLabel {
        background-color: transparent;
        border: none;
        font-size: 16px; /* Base size */
        color: #aaa;
    }
    #PhraseLabel {
        font-size: 16px; /* Base size */
        background-color: #383838; 
        color: #cfcfcf; 
        padding: 5px; 
        border-radius: 5px;
    }
"""

class CameraProcessingThread(QThread):
    frame_ready = Signal(QImage)
    prediction_ready = Signal(str) # Reverted: only char, confidence on frame
    status_update = Signal(str)
    model_loaded = Signal(bool)
    mediapipe_loaded = Signal(bool)

    def __init__(self, model_path, alphabet_path, labels_dict):
        super().__init__()
        self.model_path = model_path
        self.alphabet_path = alphabet_path # Though not used in run() as per current logic
        self.labels_dict = labels_dict
        self.cap = None
        self.hands_instance = None
        self.model = None
        # self.drawing_utils = mp.solutions.drawing_utils # Removed
        # self.drawing_styles = mp.solutions.drawing_styles # Removed
        self.hands_solution = mp.solutions.hands # Now uses global mp
        self.running = True

    def run(self):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles # Available if needed for custom styles

        # Initialization Phase
        try:
            from tensorflow.keras.models import load_model # Moved import here
            self.model = load_model(self.model_path)
            self.model_loaded.emit(True)
            self.status_update.emit("Keras model loaded successfully.")
        except Exception as e:
            self.model_loaded.emit(False)
            self.status_update.emit(f"Error loading Keras model: {e}")
            return

        try:
            # import mediapipe as mp # Moved to global scope
            self.hands_solution = mp.solutions.hands # Initialize using global mp
            self.hands_instance = self.hands_solution.Hands(
                static_image_mode=False, max_num_hands=1,
                min_detection_confidence=0.8, min_tracking_confidence=0.5)
            self.mediapipe_loaded.emit(True)
            self.status_update.emit("MediaPipe Hands initialized successfully.")
        except Exception as e:
            self.mediapipe_loaded.emit(False)
            self.status_update.emit(f"Failed to initialize MediaPipe Hands: {e}")
            return

        camera_indices_to_try = [0, 1, 2]
        for camera_index in camera_indices_to_try:
            cap_test = cv2.VideoCapture(camera_index)
            if cap_test.isOpened():
                self.cap = cap_test
                self.status_update.emit(f"Successfully opened camera with index: {camera_index}")
                break
            cap_test.release()
        
        if self.cap is None:
            self.status_update.emit("Error: Camera not found")
            return

        # Processing Loop
        while self.running:
            if not self.cap or not self.cap.isOpened():
                self.status_update.emit("Error: Camera disconnected or failed.")
                self.running = False # Stop the loop if camera fails
                break

            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.status_update.emit("Error capturing frame.")
                time.sleep(0.05) # Wait a bit before retrying
                continue

            H, W, _ = frame.shape
            
            # Hand landmark detection first (on a copy if needed, or on a specific format)
            frame_rgb_for_mediapipe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # MediaPipe prefers RGB
            frame_rgb_for_mediapipe.flags.writeable = False # To improve performance
            results = self.hands_instance.process(frame_rgb_for_mediapipe)
            print(f"DEBUG_MEDIAPIPE: results.multi_hand_landmarks = {results.multi_hand_landmarks}")
            
            # frame_rgb_for_mediapipe.flags.writeable = True # Set back if further processing on this RGB frame is needed

            predicted_char_to_emit = "" # Default to empty string
            confidence_to_emit = 0.0   # Default confidence

            # Draw landmarks on the original BGR 'frame' if hands are detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    data_aux = []
                    x_, y_ = [], []

                    # Usar estilos predeterminados de MediaPipe para landmarks y conexiones
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        mp.solutions.drawing_styles.get_default_hand_connections_style()
                    )

                    # Extraer coordenadas normalizadas
                    for lm in hand_landmarks.landmark:
                        x_.append(lm.x)
                        y_.append(lm.y)

                    # Normalizar coordenadas para predicción
                    for i in range(len(hand_landmarks.landmark)):
                        data_aux.append(x_[i] - min(x_))
                        data_aux.append(y_[i] - min(y_))
                    data_aux = data_aux[:42]  # Limitar a los primeros 42 valores

                    # Realizar predicción con el modelo
                    prediction = self.model.predict(np.array([data_aux]))
                    max_prob = np.max(prediction[0])
                    predicted_index = np.argmax(prediction[0])
                    predicted_character = self.labels_dict[predicted_index] if max_prob >= 0.9 else '?'

                    # Calcular posición del texto
                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10

                   
                    # Data extraction for prediction
                    data_aux = []
                    x_coords_norm = [lm.x for lm in hand_landmarks.landmark] # Normalized coords
                    y_coords_norm = [lm.y for lm in hand_landmarks.landmark] # Normalized coords
                    if not x_coords_norm or not y_coords_norm: continue

                    # For data_aux, use normalized relative coordinates
                    min_x_norm, min_y_norm = min(x_coords_norm), min(y_coords_norm)
                    for i in range(len(x_coords_norm)):
                        data_aux.append(x_coords_norm[i] - min_x_norm)
                        data_aux.append(y_coords_norm[i] - min_y_norm)

                    # For text placement, calculate absolute pixel coordinates
                    x_coords_abs = [int(x * W) for x in x_coords_norm]
                    y_coords_abs = [int(y * H) for y in y_coords_norm]
                    
                    # min_x, min_y = min(x_coords), min(y_coords) # Original calculation for data_aux
                    if self.model and len(data_aux) == self.model.input_shape[1]:
                        prediction_array = np.array([data_aux])
                        try:
                            prediction_result = self.model.predict(prediction_array, verbose=0)
                            max_prob = np.max(prediction_result[0])
                            predicted_index = np.argmax(prediction_result[0])
                            predicted_character = self.labels_dict.get(predicted_index, '?')
                            print(f"DEBUG_PREDICT: Raw prediction: '{predicted_character}', Confidence: {max_prob:.4f}")
                            
                            if (predicted_character == "STOP" and max_prob >= 0.99) or \
                               (predicted_character != "STOP" and 'A' <= predicted_character <= 'Z' and max_prob >= 0.95):
                                predicted_char_to_emit = predicted_character 
                                confidence_to_emit = float(max_prob)

                                # Draw text next to hand if a valid prediction is made for this hand
                                max_x_for_text = max(x_coords_abs) if x_coords_abs else W 
                                min_y_for_text = min(y_coords_abs) if y_coords_abs else 0
                                
                                text_x = max_x_for_text + 10
                                # Ensure y is within frame and not too high, considering text height (approx 20-30px for scale 0.7-1)
                                text_y = max(30, min_y_for_text) 
                                # Further ensure text_y is not too low if hand is at bottom of frame
                                text_y = min(text_y, H - 10) # Ensure it's not off the bottom of the screen

                                text_to_draw = f"{predicted_char_to_emit} ({confidence_to_emit:.0%})"
                                cv2.putText(frame, text_to_draw, (text_x, text_y), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                        except Exception as e_predict:
                            self.status_update.emit(f"Model prediction error: {e_predict}")
                
            # After iterating through all hands (max_num_hands=1, so this loop runs once if a hand is found)
            # The text drawing is now done inside the loop, associated with the hand.
            # The old fixed-position drawing (if predicted_char_to_emit was true after loop) is removed.
            
            if predicted_char_to_emit: # If a character met threshold from the hand
                self.prediction_ready.emit(predicted_char_to_emit)

            # After processing and potential drawing, convert the (possibly annotated) BGR frame to RGB for QImage
            frame_for_display_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            q_image = QImage(frame_for_display_rgb.data, W, H, frame_for_display_rgb.strides[0], QImage.Format_RGB888)
            self.frame_ready.emit(q_image.copy()) # Emit a copy

            # Small delay to control processing speed and allow GUI events
            time.sleep(0.01)

        # Cleanup Phase
        if self.cap:
            self.cap.release()
            self.status_update.emit("Camera released.")
        if self.hands_instance:
            self.hands_instance.close() # Assuming MediaPipe Hands has a close method or similar cleanup
            self.status_update.emit("MediaPipe Hands closed.")
        self.status_update.emit("CameraProcessingThread finished.")

    def stop(self):
        self.running = False
        self.status_update.emit("Stopping CameraProcessingThread...")
        # self.wait() # Wait for the run() method to complete. Consider timeout.


class ASLRecognitionApp(QMainWindow):
    # Font scaling UI and related variables removed
    TARGET_PRACTICE_IMAGE_SIZE = (200, 200) # Uniform size for practice images

    def __init__(self):
        super().__init__()
        self.current_practice_mode = "Random" # Default practice mode
        self.manual_target_letter = "A"      # Default for manual selection
        self.setWindowTitle(WINDOW_NAME)
        if os.path.exists(ICON_PATH):
            try: self.setWindowIcon(QIcon(ICON_PATH))
            except Exception as e: print(f"Could not load main window icon '{ICON_PATH}': {e}")
        
        self.setGeometry(100, 100, 900, 800)

        self.detected_phrase = ""
        self.last_detected_letter = ""
        self.last_detection_time = 0
        self.continuous_detection_start = 0
        self.stop_detection_time = 0
        self.stop_detected = False
        self.current_learning_letter = ""
        self.learning_mode_active = False
        
        self.labels_dict = {i: chr(65 + i) for i in range(26)}
        self.labels_dict[26] = "STOP"

        self.settings = QSettings("ASLApp", "Preferences")
        self._load_settings()

        # self._setup_resources() # Delegated to thread
        self._setup_ui() # Setup UI first
        
        self.apply_theme(self.current_theme_name)
        # Call to apply_font_scale removed

        # Camera and Processing Thread Setup
        self.camera_thread = CameraProcessingThread(MODEL_PATH, ALPHABET_PATH, self.labels_dict)
        self.camera_thread.frame_ready.connect(self.display_video_frame)
        self.camera_thread.prediction_ready.connect(self.update_prediction) # Signature already (str)
        self.camera_thread.status_update.connect(self.handle_status_update)
        self.camera_thread.model_loaded.connect(self.on_model_loaded)
        self.camera_thread.mediapipe_loaded.connect(self.on_mediapipe_loaded)

        # Connect new practice mode signals
        if hasattr(self, 'practice_mode_combo'): # Check if UI setup was successful
            self.practice_mode_combo.currentTextChanged.connect(self.on_practice_mode_changed)
        if hasattr(self, 'letter_select_combo'):
            self.letter_select_combo.currentTextChanged.connect(self.on_manual_letter_selected)
        
        self.camera_thread.start()
        print("ASLRecognitionApp initialized and camera thread started.")

    @Slot(QImage)
    def display_video_frame(self, q_image):
        pixmap = QPixmap.fromImage(q_image)
        current_tab_widget = self.tabs.currentWidget()
        if current_tab_widget == self.recognition_tab:
            if hasattr(self, 'main_video_label'): # Check if renamed
                 self.main_video_label.setPixmap(pixmap.scaled(self.main_video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            elif hasattr(self, 'video_label'): # Fallback to old name if rename failed / not yet applied
                 self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        elif current_tab_widget == self.practice_tab:
            if hasattr(self, 'practice_video_label'):
                 self.practice_video_label.setPixmap(pixmap.scaled(self.practice_video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @Slot(str) # Reverted: only char
    def update_prediction(self, predicted_character): # Reverted: only char
        current_time = time.time()
        # current_prediction_display using confidence_score removed for phrase_label

        # Logic adapted from old update_gui_frame
        if predicted_character == "STOP":
            if self.stop_detection_time == 0: self.stop_detection_time = current_time
            elif current_time - self.stop_detection_time >= 2.0:
                if not self.stop_detected:
                    self.stop_detected = True
                    print("INFO: STOP detected and held.")
                    self.phrase_label.setStyleSheet(f"font-size: 16px; background-color: darkblue; color: white; padding: 5px; border-radius: 5px;")
        else: # Reset stop detection if other char or no char
            self.stop_detection_time = 0
            # If stop was previously detected and now it's not, reset style
            if self.stop_detected and predicted_character != "STOP": # Check if it was a real character
                 self.stop_detected = False # Reset if a valid non-STOP character is detected
                 self.phrase_label.setStyleSheet(f"font-size: 16px; background-color: {'#383838' if self.current_theme_name == 'Dark' else '#e0e0e0'}; color: {'#cfcfcf' if self.current_theme_name == 'Dark' else 'black'}; padding: 5px; border-radius: 5px;")


        if not self.stop_detected and predicted_character and predicted_character != "STOP":
            # Confidence for A-Z check is assumed to be handled in thread before emitting signal
            if self.learning_mode_active:
                if self.current_learning_letter == "DONE": pass
                elif not self.current_learning_letter: self.load_learning_image_pyside()

                if predicted_character == self.current_learning_letter:
                    if self.continuous_detection_start == 0: self.continuous_detection_start = current_time
                    if current_time - self.continuous_detection_start >= 2.0:
                        print(f"INFO: Correctly practiced letter: {predicted_character}.")
                        self.load_learning_image_pyside(exclude_letter=self.current_learning_letter)
                        self.continuous_detection_start = 0
                else:
                    self.continuous_detection_start = 0
                # Update label for learning mode (without confidence)
                self.phrase_label.setText(f"Target: {self.current_learning_letter} | You: {predicted_character}")
            else: # Normal recognition mode
                if predicted_character != self.last_detected_letter or (current_time - self.last_detection_time >= 1.5):
                    self.detected_phrase += predicted_character
                    self.last_detected_letter = predicted_character
                    self.last_detection_time = current_time
                    # INFO log no longer includes confidence_score directly from here
                    print(f"INFO: Stored: {predicted_character}. Phrase: '{self.detected_phrase}'")
                # Update phrase_label (without confidence)
                self.phrase_label.setText(f"Sign: {predicted_character} | Phrase: {self.detected_phrase}")
        elif self.stop_detected: # If STOP was detected and is still active
             self.phrase_label.setStyleSheet(f"font-size: 16px; background-color: darkblue; color: white; padding: 5px; border-radius: 5px;") # Ensure style remains
             self.phrase_label.setText(f"Sign: STOP | Phrase: {self.detected_phrase}") 
        else: # Not learning, not stop_detected, but also not adding to phrase (e.g. debounce)
             # Update phrase_label (without confidence)
            self.phrase_label.setText(f"Sign: {predicted_character} | Phrase: {self.detected_phrase}")


    @Slot(str)
    def handle_status_update(self, message):
        print(f"THREAD_STATUS: {message}")
        # Optionally, display important status messages in the UI, e.g., self.phrase_label or a status bar
        if "Error" in message or "Failed" in message :
             self.phrase_label.setText(message) # Show critical errors on phrase_label

    @Slot(bool)
    def on_model_loaded(self, loaded):
        if loaded:
            print("Model successfully loaded by thread.")
        else:
            print("Model loading failed in thread.")
            error_message = "Error: Keras model failed to load. Check logs."
            if hasattr(self, 'main_video_label'): self.main_video_label.setText(error_message)
            elif hasattr(self, 'video_label'): self.video_label.setText(error_message) # Fallback
            if hasattr(self, 'practice_video_label'): self.practice_video_label.setText(error_message)

    @Slot(bool)
    def on_mediapipe_loaded(self, loaded):
        if loaded:
            print("MediaPipe successfully loaded by thread.")
        else:
            print("MediaPipe loading failed in thread.")
            error_message = "Error: MediaPipe failed to load. Check logs."
            if hasattr(self, 'main_video_label'): self.main_video_label.setText(error_message)
            elif hasattr(self, 'video_label'): self.video_label.setText(error_message) # Fallback
            if hasattr(self, 'practice_video_label'): self.practice_video_label.setText(error_message)


    def _load_settings(self):
        self.current_theme_name = self.settings.value("theme", "Light") # Default to Light
        # Font scale loading removed
        print(f"Loaded settings: Theme='{self.current_theme_name}'")


    def _save_settings(self):
        self.settings.setValue("theme", self.current_theme_name)
        # Font scale saving removed
        print(f"Saved settings: Theme='{self.current_theme_name}'")

    def _setup_ui(self):
        print("Setting up UI with tabs and theme/font controls.")
        
        # Create a main vertical layout for the central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.tabs = QTabWidget()
        # Add tabs first to the main layout
        main_layout.addWidget(self.tabs)


        # --- Recognition Tab ---
        self.recognition_tab = QWidget()
        recognition_layout = QVBoxLayout(self.recognition_tab)
        self.main_video_label = QLabel("Initializing Camera...") # Renamed from self.video_label
        self.main_video_label.setObjectName("MainVideoLabel") # Updated object name
        self.main_video_label.setAlignment(Qt.AlignCenter)
        self.main_video_label.setMinimumSize(800, 520) 
        recognition_layout.addWidget(self.main_video_label)

        self.phrase_label = QLabel("Initializing components...")
        self.phrase_label.setObjectName("PhraseLabel")
        self.phrase_label.setFixedHeight(30) # Keep fixed height
        recognition_layout.addWidget(self.phrase_label)
        
        recognition_controls_layout = QHBoxLayout()
        recognition_controls_layout.addStretch()
        self.reset_button = QPushButton()
        # ... (reset_button icon logic as before) ...
        if os.path.exists("reset.png"):
            try:
                self.reset_button.setIcon(QIcon("reset.png"))
                self.reset_button.setIconSize(QSize(32, 32)); self.reset_button.setFixedSize(QSize(48,48))
            except Exception as e_icon:
                 print(f"Failed to load reset.png as icon: {e_icon}."); self.reset_button.setText("Reset")
        else:
            self.reset_button.setText("Reset"); print("reset.png not found.")
        self.reset_button.clicked.connect(self.reset_detected_text_action)
        recognition_controls_layout.addWidget(self.reset_button)
        recognition_layout.addLayout(recognition_controls_layout)

        self.tabs.addTab(self.recognition_tab, "Recognition")

        # --- Practice Tab ---
        self.practice_tab = QWidget()
        practice_tab_main_v_layout = QVBoxLayout(self.practice_tab) # Main layout is Vertical

        # Controls layout (Horizontal)
        practice_controls_layout = QHBoxLayout()
        practice_controls_layout.addWidget(QLabel("Mode:"))
        self.practice_mode_combo = QComboBox()
        self.practice_mode_combo.addItems(["Random", "Manual"])
        practice_controls_layout.addWidget(self.practice_mode_combo)

        self.letter_select_combo = QComboBox()
        self.letter_select_combo.addItems([chr(65 + i) for i in range(26)]) # A-Z
        self.letter_select_combo.setEnabled(False) # Disabled for Random mode initially
        practice_controls_layout.addWidget(self.letter_select_combo)
        practice_controls_layout.addStretch() # Push controls to the left
        
        practice_tab_main_v_layout.addLayout(practice_controls_layout) # Add controls HBox to main VBox

        # Video and Image layout (Horizontal) - this is the existing QHBoxLayout
        video_image_layout = QHBoxLayout()
        
        self.practice_video_label = QLabel("Camera Feed") 
        self.practice_video_label.setObjectName("PracticeVideoLabel") 
        self.practice_video_label.setAlignment(Qt.AlignCenter)
        self.practice_video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.practice_video_label.setScaledContents(False)
        video_image_layout.addWidget(self.practice_video_label, 85) # Video on left

        self.learning_image_display_label = QLabel("Select 'Practice' tab to start learning.")
        self.learning_image_display_label.setObjectName("LearningImageLabel")
        self.learning_image_display_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.learning_image_display_label.setMinimumSize(QSize(250, 250))
        video_image_layout.addWidget(self.learning_image_display_label, 3) # Image on right
        
        practice_tab_main_v_layout.addLayout(video_image_layout) # Add video/image HBox to main VBox
        self.practice_tab.setLayout(practice_tab_main_v_layout)

        self.tabs.addTab(self.practice_tab, "Practice")
        self.tabs.currentChanged.connect(self.handle_tab_change)

        # --- Settings/Controls GroupBox ---
        settings_groupbox = QGroupBox("Settings")
        settings_layout = QFormLayout(settings_groupbox) # QFormLayout is good for label-field pairs

        # Theme Selector
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark"])
        self.theme_combo.setCurrentText(self.current_theme_name) # Set initial from loaded settings
        self.theme_combo.currentTextChanged.connect(self.apply_theme_from_combo)
        settings_layout.addRow(QLabel("Theme:"), self.theme_combo)

        # Font Scaler UI elements removed
        
        # Add settings groupbox to the main layout, below the tabs
        main_layout.addWidget(settings_groupbox)
        
        print("UI setup with tabs and theme/font controls complete.")

    # _setup_resources is now largely handled by CameraProcessingThread
    def _setup_resources(self):
        # This method can be removed or repurposed if there are other non-thread resources.
        # For now, we'll leave it commented out or minimal.
        print("Resource setup delegated to CameraProcessingThread.")
        pass


    @Slot(str)
    def apply_theme_from_combo(self, theme_name):
        self.apply_theme(theme_name)

    def apply_theme(self, theme_name):
        print(f"DEBUG: apply_theme called with theme_name: {theme_name}")
        print(f"Applying theme: {theme_name}")
        self.current_theme_name = theme_name # Store current theme name
        if theme_name == "Dark":
            QApplication.instance().setStyleSheet(DARK_THEME_STYLESHEET)
        else: # Default to Light theme
            QApplication.instance().setStyleSheet(LIGHT_THEME_STYLESHEET)
        # Re-apply font scaling as stylesheet might overwrite general font settings
        self._save_settings()

    # adjust_font_scale method removed
    # apply_font_scale method removed

    @Slot(int)
    def handle_tab_change(self, index):
        current_tab_text = self.tabs.tabText(index)
        print(f"Switched to tab: '{current_tab_text}' (Index: {index})")
        if current_tab_text == "Practice":
            self.learning_mode_active = True
            # self.load_learning_image_pyside() # Old call
            self.on_practice_mode_changed(self.practice_mode_combo.currentText()) # Update image based on mode
            self.last_detection_time = time.time() 
            # self.current_learning_letter will be set by on_practice_mode_changed via load_learning_image_pyside
            self.detected_phrase = "" 
            self.phrase_label.setText("Detected: ") # Update UI
            print("Learning mode activated (Practice tab).")
        else: # Recognition tab or any other tab
            self.learning_mode_active = False
            self.current_learning_letter = "" # Clear learning letter
            # Optionally, clear learning image display if it's separate and visible
            if hasattr(self, 'learning_image_display_label'):
                 self.learning_image_display_label.setText("Switch to 'Practice' tab to start learning.")
                 self.learning_image_display_label.setPixmap(QPixmap()) # Clear image
            print("Learning mode deactivated (Switched to Recognition or other tab).")


    # Slot methods for practice mode
    @Slot(str)
    def on_practice_mode_changed(self, mode):
        self.current_practice_mode = mode
        print(f"INFO: Practice mode changed to: {mode}")
        if mode == "Manual":
            self.letter_select_combo.setEnabled(True)
            current_manual_letter = self.letter_select_combo.currentText()
            self.manual_target_letter = current_manual_letter
            self.load_learning_image_pyside(force_letter=current_manual_letter)
        else: # Random mode
            self.letter_select_combo.setEnabled(False)
            self.load_learning_image_pyside() # Load a random image

    @Slot(str)
    def on_manual_letter_selected(self, letter):
        if self.current_practice_mode == "Manual": # Ensure this only acts if in manual mode
            self.manual_target_letter = letter
            print(f"INFO: Manual letter selected: {letter}")
            self.load_learning_image_pyside(force_letter=letter)

    def load_learning_image_pyside(self, exclude_letter=None, force_letter=None):
        print(f"DEBUG_PRACTICE: load_learning_image_pyside called. Exclude: {exclude_letter}, Force: {force_letter}, Mode: {self.current_practice_mode}")
        print(f"DEBUG_PRACTICE: ALPHABET_PATH: {ALPHABET_PATH}")
        if not hasattr(self, 'learning_image_display_label'):
            print("ERROR: learning_image_display_label not found. Cannot load learning image.")
            return

        if not os.path.exists(ALPHABET_PATH) or not os.path.isdir(ALPHABET_PATH):
            print(f"ERROR: Alphabet examples directory '{ALPHABET_PATH}' not found.")
            self.learning_image_display_label.setText("Alphabet images not found.")
            return
        try:
            images = [f for f in os.listdir(ALPHABET_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and os.path.isfile(os.path.join(ALPHABET_PATH, f))]
            print(f"DEBUG_PRACTICE: Found images: {images}") # This line might be problematic if images isn't defined yet
            
            selected_image_name = None
            if force_letter:
                self.current_learning_letter = force_letter.upper()
                possible_extensions = ['.jpg', '.jpeg', '.png']
                for ext in possible_extensions:
                    potential_file = f"{self.current_learning_letter}{ext}"
                    if os.path.isfile(os.path.join(ALPHABET_PATH, potential_file)):
                        selected_image_name = potential_file
                        break
                if not selected_image_name:
                    print(f"ERROR_PRACTICE: Image for forced letter '{self.current_learning_letter}' not found in {ALPHABET_PATH}")
                    self.learning_image_display_label.setText(f"Img for {self.current_learning_letter} not found.")
                    return
            elif self.current_practice_mode == "Manual":
                self.current_learning_letter = self.manual_target_letter.upper()
                possible_extensions = ['.jpg', '.jpeg', '.png']
                for ext in possible_extensions:
                    potential_file = f"{self.current_learning_letter}{ext}"
                    if os.path.isfile(os.path.join(ALPHABET_PATH, potential_file)):
                        selected_image_name = potential_file
                        break
                if not selected_image_name:
                    print(f"ERROR_PRACTICE: Image for manual letter '{self.current_learning_letter}' not found.")
                    self.learning_image_display_label.setText(f"Img for {self.current_learning_letter} not found.")
                    return
            else: # Random mode
                images = [f for f in os.listdir(ALPHABET_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and os.path.isfile(os.path.join(ALPHABET_PATH, f))]
                if not images:
                    print(f"WARNING: No images found in '{ALPHABET_PATH}'.")
                    self.learning_image_display_label.setText("No images in alphabet folder.")
                    return
                if exclude_letter:
                    images = [img for img in images if os.path.splitext(img)[0].upper() != exclude_letter.upper()]
                if not images:
                    print(f"INFO: No new images after excluding '{exclude_letter}'. All letters practiced?")
                    self.learning_image_display_label.setText("All letters practiced! Reset or switch tabs.")
                    self.current_learning_letter = "DONE" 
                    return
                selected_image_name = random.choice(images)
                self.current_learning_letter = os.path.splitext(selected_image_name)[0].upper()

            print(f"DEBUG_PRACTICE: Selected image name: {selected_image_name}")
            image_full_path = os.path.join(ALPHABET_PATH, selected_image_name)
            print(f"DEBUG_PRACTICE: Full image path: {image_full_path}")
            
            try:
                pil_img = Image.open(image_full_path)
                pil_img = pil_img.resize(self.TARGET_PRACTICE_IMAGE_SIZE, Image.Resampling.LANCZOS)
                
                if pil_img.mode == "RGB":
                    qimage_format = QImage.Format_RGB888
                    num_channels = 3
                elif pil_img.mode == "RGBA":
                    qimage_format = QImage.Format_RGBA8888
                    num_channels = 4
                else: 
                    pil_img = pil_img.convert("RGB")
                    qimage_format = QImage.Format_RGB888
                    num_channels = 3 # RGB after conversion
                print(f"DEBUG_PRACTICE: PIL Image mode: {pil_img.mode}, width: {pil_img.width}, height: {pil_img.height}, num_channels: {num_channels}")

                img_data = pil_img.tobytes("raw", pil_img.mode) 
                q_img = QImage(img_data, pil_img.width, pil_img.height, pil_img.width * num_channels, qimage_format)
            
            except Exception as e_img_load:
                print(f"ERROR_PRACTICE: Failed to load or create QImage for {image_full_path}: {e_img_load}")
                self.learning_image_display_label.setText(f"Error loading: {os.path.basename(image_full_path)}")
                return # Exit if image can't be processed

            q_pixmap = QPixmap.fromImage(q_img)

            label_size = self.learning_image_display_label.size()
            target_w, target_h = self.TARGET_PRACTICE_IMAGE_SIZE
            
            if label_size.width() < 50 or label_size.height() < 50: # Arbitrary small threshold
                current_size = QSize(target_w, target_h)
                # print(f"WARNING_PRACTICE: learning_image_display_label size is very small ({label_size.width()}x{label_size.height()}). Falling back to TARGET_PRACTICE_IMAGE_SIZE {self.TARGET_PRACTICE_IMAGE_SIZE}.") # Debug
            else:
                current_size = label_size
            
            scaled_pixmap = q_pixmap.scaled(current_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            self.learning_image_display_label.setPixmap(scaled_pixmap)
            self.current_learning_letter = os.path.splitext(selected_image_name)[0].upper()
            print(f"INFO: Practice Tab: Displaying image for letter '{self.current_learning_letter}'.")
        except Exception as e:
            print(f"ERROR: Error loading learning image for Practice tab: {e}")
            self.learning_image_display_label.setText("Error loading image.")

    @Slot()
    def reset_detected_text_action(self):
        self.detected_phrase = ""

        # Reset to a generic message, or specific if theme is known
        base_text = "Detected: "
        if hasattr(self, 'current_theme_name') and self.current_theme_name == "Dark":
             self.phrase_label.setStyleSheet(f"font-size: 16px; background-color: #383838; color: #cfcfcf; padding: 5px; border-radius: 5px;")
        else:
             self.phrase_label.setStyleSheet(f"font-size: 16px; background-color: #e0e0e0; color: black; padding: 5px; border-radius: 5px;")
        self.phrase_label.setText(base_text) 
        print("INFO: Detected text and STOP state reset.")

    # update_gui_frame is removed; its logic is now in CameraProcessingThread and new slots.

    def closeEvent(self, event):
        print("INFO: Close event triggered for main window.")
        if hasattr(self, 'camera_thread') and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread.wait() # Wait for thread to finish
        
        # cv2.destroyAllWindows() # Should not be needed if cv2 is only in thread
        print("INFO: Application resources released. Exiting.")
        event.accept()
        QApplication.instance().quit()


if __name__ == "__main__":
    # setup_logging removed
    
    app = QApplication(sys.argv)
    main_window = ASLRecognitionApp() # __init__ now starts the thread
    
    # The old critical component check might need adjustment or removal,
    # as components are loaded in the thread. Status signals will update UI.
    # For now, we assume the app will start and the thread will report status.
    # If initial critical failures (like model path not existing even before thread start)
    # are a concern, some checks could remain or be adapted.
    
    # Example of a pre-thread check that could be added:
    if not os.path.exists(MODEL_PATH):
         print(f"CRITICAL: Model file '{MODEL_PATH}' not found. The application might not function correctly.")
         # Optionally, show a message box and exit
         try:
            from PySide6.QtWidgets import QMessageBox
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setWindowTitle("Application Error")
            msg_box.setText(f"Critical component '{MODEL_PATH}' is missing.\nThe application may not work as expected.")
            msg_box.exec()
         except Exception as e_msgbox:
            print(f"ERROR: Failed to show critical error message box: {e_msgbox}")
         # Depending on severity, could sys.exit(-1) here

    main_window.show()
    
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("INFO: Application interrupted by user (Ctrl+C from console).")
        if hasattr(main_window, 'closeEvent'):
             main_window.close()
        sys.exit(0)
    except Exception as e_main_exec:
        print(f"CRITICAL: Unhandled exception in application exec loop: {e_main_exec}")
        if hasattr(main_window, 'closeEvent'): # Attempt graceful shutdown
             main_window.close()
        sys.exit(-1)