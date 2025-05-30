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

import sys
import os
import random
import time
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

from PySide6.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                               QVBoxLayout, QHBoxLayout, QWidget, QTabWidget,
                               QComboBox, QSlider, QGroupBox, QFormLayout) # Added for new controls
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
        background-color: #e8e8e8;
        border: 2px dashed #c0c0c0;
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
        background-color: #3c3c3c;
        border: 2px dashed #5a5a5a;
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
    prediction_ready = Signal(str)
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
        self.drawing_utils = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles
        self.hands_solution = mp.solutions.hands
        self.running = True

    def run(self):
        # Initialization Phase
        try:
            self.model = load_model(self.model_path)
            self.model_loaded.emit(True)
            self.status_update.emit("Keras model loaded successfully.")
        except Exception as e:
            self.model_loaded.emit(False)
            self.status_update.emit(f"Error loading Keras model: {e}")
            return

        try:
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
            
            # Emit raw camera frame for display
            frame_rgb_for_qimage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            q_image = QImage(frame_rgb_for_qimage.data, W, H, frame_rgb_for_qimage.strides[0], QImage.Format_RGB888)
            self.frame_ready.emit(q_image.copy()) # Emit a copy

            # Hand landmark detection
            frame_rgb_for_mediapipe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Separate conversion for mediapipe
            frame_rgb_for_mediapipe.flags.writeable = False
            results = self.hands_instance.process(frame_rgb_for_mediapipe)
            # frame_rgb_for_mediapipe.flags.writeable = True # Not strictly needed as we draw on a copy or not at all here

            predicted_character_for_signal = "" # Default to empty string

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Optionally, one could draw landmarks here on a copy of the frame
                    # and emit that as a separate signal if needed, or let main thread draw.
                    # For simplicity, this example focuses on prediction.

                    data_aux = []
                    x_coords = [lm.x for lm in hand_landmarks.landmark]
                    y_coords = [lm.y for lm in hand_landmarks.landmark]
                    if not x_coords or not y_coords: continue

                    min_x, min_y = min(x_coords), min(y_coords)
                    for i in range(len(x_coords)):
                        data_aux.append(x_coords[i] - min_x)
                        data_aux.append(y_coords[i] - min_y)
                    
                    if self.model and len(data_aux) == self.model.input_shape[1]:
                        prediction_array = np.array([data_aux])
                        try:
                            prediction_result = self.model.predict(prediction_array, verbose=0)
                            max_prob = np.max(prediction_result[0])
                            predicted_index = np.argmax(prediction_result[0])
                            predicted_character = self.labels_dict.get(predicted_index, '?')
                            
                            # For now, only emit if confidence is high for non-STOP, or very high for STOP
                            # This logic can be adjusted or moved to main thread based on `prediction_ready`
                            if (predicted_character == "STOP" and max_prob >= 0.99) or \
                               (predicted_character != "STOP" and 'A' <= predicted_character <= 'Z' and max_prob >= 0.95):
                                predicted_character_for_signal = predicted_character # Store it to emit after loop

                        except Exception as e_predict:
                            self.status_update.emit(f"Model prediction error: {e_predict}")
                            # Continue, don't break loop for one prediction error
                
                if predicted_character_for_signal: # Emit if a valid prediction was made
                    self.prediction_ready.emit(predicted_character_for_signal)

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
    # Define base font size and scaling limits
    BASE_FONT_SIZE = 10 # Default base size for font scaling logic
    MIN_FONT_SCALE_FACTOR = 0.8
    MAX_FONT_SCALE_FACTOR = 1.5
    CURRENT_FONT_SCALE_FACTOR = 1.0

    def __init__(self):
        super().__init__()
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
        self.apply_font_scale(self.CURRENT_FONT_SCALE_FACTOR, initial_setup=True)

        # Camera and Processing Thread Setup
        self.camera_thread = CameraProcessingThread(MODEL_PATH, ALPHABET_PATH, self.labels_dict)
        self.camera_thread.frame_ready.connect(self.display_video_frame)
        self.camera_thread.prediction_ready.connect(self.update_prediction)
        self.camera_thread.status_update.connect(self.handle_status_update)
        self.camera_thread.model_loaded.connect(self.on_model_loaded)
        self.camera_thread.mediapipe_loaded.connect(self.on_mediapipe_loaded)
        
        self.camera_thread.start()
        print("ASLRecognitionApp initialized and camera thread started.")

    @Slot(QImage)
    def display_video_frame(self, q_image):
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @Slot(str)
    def update_prediction(self, predicted_character):
        current_time = time.time()
        # Logic adapted from old update_gui_frame
        if predicted_character == "STOP": # Assuming high confidence check is done in thread for STOP
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
            else: # Normal recognition mode
                if predicted_character != self.last_detected_letter or (current_time - self.last_detection_time >= 1.5):
                    self.detected_phrase += predicted_character
                    self.last_detected_letter = predicted_character
                    self.last_detection_time = current_time
                    self.phrase_label.setText(f"Detected: {self.detected_phrase}")
                    print(f"INFO: Stored: {predicted_character}. Phrase: '{self.detected_phrase}'")
        
        # Update phrase label if not in learning mode or if phrase needs to be cleared
        if not self.learning_mode_active :
            self.phrase_label.setText(f"Detected: {self.detected_phrase}")


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
            # Enable UI elements that depend on the model
        else:
            print("Model loading failed in thread.")
            self.video_label.setText("Error: Keras model failed to load. Check logs.")
            # Disable UI elements

    @Slot(bool)
    def on_mediapipe_loaded(self, loaded):
        if loaded:
            print("MediaPipe successfully loaded by thread.")
            # Enable UI elements that depend on MediaPipe
        else:
            print("MediaPipe loading failed in thread.")
            self.video_label.setText("Error: MediaPipe failed to load. Check logs.")
            # Disable UI elements


    def _load_settings(self):
        self.current_theme_name = self.settings.value("theme", "Light") # Default to Light
        try:
            self.CURRENT_FONT_SCALE_FACTOR = float(self.settings.value("font_scale", 1.0))
        except ValueError:
            self.CURRENT_FONT_SCALE_FACTOR = 1.0
        print(f"Loaded settings: Theme='{self.current_theme_name}', FontScale={self.CURRENT_FONT_SCALE_FACTOR}")


    def _save_settings(self):
        self.settings.setValue("theme", self.current_theme_name)
        self.settings.setValue("font_scale", self.CURRENT_FONT_SCALE_FACTOR)
        print(f"Saved settings: Theme='{self.current_theme_name}', FontScale={self.CURRENT_FONT_SCALE_FACTOR}")

    def _setup_ui(self):
        # This method remains largely the same, but video_label initial text might change.
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
        # ... (video_label, phrase_label, reset_button setup as before) ...
        self.video_label = QLabel("Initializing Camera...")
        self.video_label.setObjectName("VideoLabel") # For specific styling
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 520) 
        recognition_layout.addWidget(self.video_label)

        self.phrase_label = QLabel("Initializing components...") # Updated initial text
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
        practice_layout = QVBoxLayout(self.practice_tab)
        # ... (learning_image_display_label setup as before) ...
        self.learning_image_display_label = QLabel("Select 'Practice' tab to start learning.")
        self.learning_image_display_label.setObjectName("LearningImageLabel")
        self.learning_image_display_label.setAlignment(Qt.AlignCenter)
        self.learning_image_display_label.setFixedSize(350, 350)
        practice_layout.addWidget(self.learning_image_display_label, alignment=Qt.AlignCenter) # Center the label
        self.practice_tab.setLayout(practice_layout) # Set layout for practice_tab

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

        # Font Scaler
        font_control_layout = QHBoxLayout()
        self.decrease_font_button = QPushButton("-")
        self.decrease_font_button.setFixedWidth(40)
        self.decrease_font_button.clicked.connect(lambda: self.adjust_font_scale(decrease=True))
        font_control_layout.addWidget(self.decrease_font_button)

        self.font_scale_label = QLabel(f"Font Scale: {self.CURRENT_FONT_SCALE_FACTOR:.1f}x") # Display current scale
        self.font_scale_label.setAlignment(Qt.AlignCenter)
        font_control_layout.addWidget(self.font_scale_label)
        
        self.increase_font_button = QPushButton("+")
        self.increase_font_button.setFixedWidth(40)
        self.increase_font_button.clicked.connect(lambda: self.adjust_font_scale(increase=True))
        font_control_layout.addWidget(self.increase_font_button)
        settings_layout.addRow(QLabel("Font Size:"), font_control_layout)
        
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
        print(f"Applying theme: {theme_name}")
        self.current_theme_name = theme_name # Store current theme name
        if theme_name == "Dark":
            QApplication.instance().setStyleSheet(DARK_THEME_STYLESHEET)
        else: # Default to Light theme
            QApplication.instance().setStyleSheet(LIGHT_THEME_STYLESHEET)
        # Re-apply font scaling as stylesheet might overwrite general font settings
        self.apply_font_scale(self.CURRENT_FONT_SCALE_FACTOR) 
        self._save_settings()


    def adjust_font_scale(self, increase=False, decrease=False):
        if increase:
            self.CURRENT_FONT_SCALE_FACTOR = min(self.MAX_FONT_SCALE_FACTOR, self.CURRENT_FONT_SCALE_FACTOR + 0.1)
        elif decrease:
            self.CURRENT_FONT_SCALE_FACTOR = max(self.MIN_FONT_SCALE_FACTOR, self.CURRENT_FONT_SCALE_FACTOR - 0.1)
        self.apply_font_scale(self.CURRENT_FONT_SCALE_FACTOR)
        self._save_settings()

    def apply_font_scale(self, scale_factor, initial_setup=False):
        self.CURRENT_FONT_SCALE_FACTOR = round(scale_factor,1) # Keep it to one decimal place
        self.font_scale_label.setText(f"Font Scale: {self.CURRENT_FONT_SCALE_FACTOR:.1f}x")
        
        # Create a new font based on the application's default font
        default_font = QApplication.font() # Get a copy of the default app font
        
        # Calculate new size based on original point size and scale factor
        # This assumes default_font.pointSize() gives a sensible base.
        # If pointSize is -1 (pixelSize is used), this logic might need adjustment.
        original_point_size = default_font.pointSize()
        if original_point_size <=0: # If pointSize is not reliable, use a fixed base
            original_point_size = self.BASE_FONT_SIZE 
            # For more robustness, could also check default_font.pixelSize()

        new_size = int(original_point_size * self.CURRENT_FONT_SCALE_FACTOR)
        
        scaled_font = QFont(default_font) # Create a new font instance
        scaled_font.setPointSize(new_size)
        
        QApplication.setFont(scaled_font) # Set global font
        
        # Update stylesheet for elements where font-size is explicitly set in stylesheets
        # This is a simplified approach; a more robust one might involve parsing and
        # regenerating the stylesheet or using QSS variables if supported/practical.
        # For now, we re-apply the whole theme which might contain scaled font sizes if they
        # are defined using relative units or if we modify the stylesheet strings here.
        # The current stylesheets use fixed px values for font-size, so global font change is primary.
        # If stylesheets had `font-size: @baseFontSize * @scaleFactor;` this would be easier.
        
        # Re-applying the theme can help ensure all elements pick up changes if some
        # elements don't dynamically update from QApplication.setFont() alone,
        # especially if their initial styles were set by the stylesheet.
        if not initial_setup: # Avoid re-applying during initial setup if already handled
             self.apply_theme(self.current_theme_name) 
        
        print(f"Applied font scale: {self.CURRENT_FONT_SCALE_FACTOR:.1f}x, New base point size: {new_size}")


    @Slot(int)
    def handle_tab_change(self, index):
        current_tab_text = self.tabs.tabText(index)
        print(f"Switched to tab: '{current_tab_text}' (Index: {index})")
        if current_tab_text == "Practice":
            self.learning_mode_active = True
            self.load_learning_image_pyside() # Load initial image for practice
            self.last_detection_time = time.time() # Reset timer for practice mode
            self.current_learning_letter = self.current_learning_letter # Keep or reset based on desired logic
            self.detected_phrase = "" # Clear phrase when switching to practice
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


    def load_learning_image_pyside(self, exclude_letter=None):
        # This method now updates self.learning_image_display_label in the "Practice" tab
        if not hasattr(self, 'learning_image_display_label'):
            print("ERROR: learning_image_display_label not found. Cannot load learning image.")
            return

        if not os.path.exists(ALPHABET_PATH) or not os.path.isdir(ALPHABET_PATH):
            print(f"ERROR: Alphabet examples directory '{ALPHABET_PATH}' not found.")
            self.learning_image_display_label.setText("Alphabet images not found.")
            return
        try:
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
                self.current_learning_letter = "DONE" # Special state
                return

            selected_image_name = random.choice(images)
            image_full_path = os.path.join(ALPHABET_PATH, selected_image_name)
            
            pil_img = Image.open(image_full_path)
            
            if pil_img.mode == "RGB": qimage_format = QImage.Format_RGB888
            elif pil_img.mode == "RGBA": qimage_format = QImage.Format_RGBA8888
            else: pil_img = pil_img.convert("RGB"); qimage_format = QImage.Format_RGB888
            
            img_data = pil_img.tobytes("raw", pil_img.mode) 
            q_img = QImage(img_data, pil_img.width, pil_img.height, pil_img.width * pil_img.getbands(), qimage_format)
            
            q_pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = q_pixmap.scaled(self.learning_image_display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.learning_image_display_label.setPixmap(scaled_pixmap)
            self.current_learning_letter = os.path.splitext(selected_image_name)[0].upper()
            print(f"INFO: Practice Tab: Displaying image for letter '{self.current_learning_letter}'.")
        except Exception as e:
            print(f"ERROR: Error loading learning image for Practice tab: {e}")
            self.learning_image_display_label.setText("Error loading image.")

    @Slot()
    def reset_detected_text_action(self):
        self.detected_phrase = ""
        self.stop_detected = False
        self.phrase_label.setText("Detected: ") 
        self.phrase_label.setStyleSheet("font-size: 16px; background-color: #e0e0e0; color: black; padding: 5px; border-radius: 5px;")
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