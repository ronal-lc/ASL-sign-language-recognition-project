"""ASL Alphabet Real-Time Recognition

This script provides a GUI for real-time American Sign Language (ASL) hand sign recognition using a webcam.
It uses MediaPipe for hand landmark detection and a trained Keras model for classification.
Includes a learning mode for practice.

Usage:
    - Ensure 'model.keras' (trained model) and 'icon.ico' exist.
    - Place alphabet example images (e.g., A.png, B.png) in the 'alphabet_examples' folder for learning mode.
    - Run this script to start the GUI.

Dependencies:
    - OpenCV (cv2)
    - MediaPipe (mediapipe)
    - NumPy (numpy)
    - Pillow (PIL)
    - TensorFlow (tensorflow)
    - PySide6

Author: @ronal-lc
"""

import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
import random
import time
import cv2
import mediapipe as mp
import numpy as np

from PySide6.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                               QVBoxLayout, QHBoxLayout, QWidget, QTabWidget,
                               QComboBox, QGroupBox, QFormLayout, QSizePolicy, QMessageBox)
from PySide6.QtGui import QImage, QPixmap, QIcon
from PySide6.QtCore import QTimer, Qt, QSize, Slot, QSettings, QThread, Signal
from PIL import Image 

WINDOW_NAME = 'ASL Alphabet Recognition'
ICON_PATH = 'icon.ico'
ALPHABET_PATH = "alphabet_examples"
MODEL_PATH = './model.keras'

# --- Theme Stylesheets ---
LIGHT_THEME_STYLESHEET = """
    QMainWindow { background-color: #f0f0f0; }
    QTabWidget::pane { border-top: 1px solid #c2c7cb; }
    QTabBar::tab {
        background: #e1e1e1; border: 1px solid #c2c7cb;
        border-bottom-color: #c2c7cb; border-top-left-radius: 4px;
        border-top-right-radius: 4px; min-width: 8ex; padding: 5px; color: #333;
    }
    QTabBar::tab:selected {
        background: #f0f0f0; border-color: #c2c7cb;
        border-bottom-color: #f0f0f0; color: #000;
    }
    QTabBar::tab:!selected:hover { background: #dcdcdc; }
    QWidget { background-color: #f0f0f0; color: #333333; }
    QLabel { color: #333333; }
    QPushButton {
        background-color: #d9534f; color: white; border-radius: 5px;
        padding: 8px 15px; font-size: 12px;
    }
    QPushButton:hover { background-color: #c9302c; }
    QPushButton:pressed { background-color: #ac2925; }
    QComboBox {
        border: 1px solid #c2c7cb; border-radius: 3px; padding: 3px 18px 3px 5px;
        min-width: 6em; background-color: white; color: #333;
    }
    QComboBox:editable { background: white; }
    QComboBox:!editable, QComboBox::drop-down:editable {
         background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #E1E1E1, stop:0.4 #DDDDDD, stop:0.5 #D8D8D8, stop:1.0 #D3D3D3);
    }
    QComboBox:!editable:on, QComboBox::drop-down:editable:on {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #D3D3D3, stop:0.4 #D8D8D8, stop:0.5 #DDDDDD, stop:1.0 #E1E1E1);
    }
    QComboBox::drop-down {
        subcontrol-origin: padding; subcontrol-position: top right; width: 15px;
        border-left-width: 1px; border-left-color: darkgray; border-left-style: solid;
        border-top-right-radius: 3px; border-bottom-right-radius: 3px;
    }
    QComboBox::down-arrow { image: url(down_arrow.png); /* Ensure down_arrow.png exists or use unicode */ }
    QGroupBox {
        background-color: #e8e8e8; border: 1px solid #c2c7cb; border-radius: 5px;
        margin-top: 1ex; font-weight: bold;
    }
    QGroupBox::title {
        subcontrol-origin: margin; subcontrol-position: top left;
        padding: 0 3px; color: #333;
    }
    #VideoLabel { background-color: black; border: 1px solid #CCCCCC; } /* Used for main_video_label, practice_video_label */
    #LearningImageLabel { background-color: transparent; border: none; font-size: 16px; color: #555; }
    #PhraseLabel { font-size: 16px; background-color: #e0e0e0; color: black; padding: 5px; border-radius: 5px; }
"""

DARK_THEME_STYLESHEET = """
    QMainWindow { background-color: #2e2e2e; }
    QTabWidget::pane { border-top: 1px solid #4a4a4a; }
    QTabBar::tab {
        background: #3c3c3c; border: 1px solid #4a4a4a;
        border-bottom-color: #4a4a4a; border-top-left-radius: 4px;
        border-top-right-radius: 4px; min-width: 8ex; padding: 5px; color: #cfcfcf;
    }
    QTabBar::tab:selected {
        background: #2e2e2e; border-color: #4a4a4a;
        border-bottom-color: #2e2e2e; color: #ffffff;
    }
    QTabBar::tab:!selected:hover { background: #484848; }
    QWidget { background-color: #2e2e2e; color: #cfcfcf; }
    QLabel { color: #cfcfcf; }
    QPushButton {
        background-color: #d9534f; color: white; border-radius: 5px;
        padding: 8px 15px; font-size: 12px;
    }
    QPushButton:hover { background-color: #c9302c; }
    QPushButton:pressed { background-color: #ac2925; }
    QComboBox {
        border: 1px solid #4a4a4a; border-radius: 3px; padding: 3px 18px 3px 5px;
        min-width: 6em; background-color: #3c3c3c; color: #cfcfcf; selection-background-color: #585858;
    }
    QComboBox:editable { background: #3c3c3c; }
    QComboBox:!editable, QComboBox::drop-down:editable {
         background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #484848, stop:0.4 #404040, stop:0.5 #3c3c3c, stop:1.0 #383838);
    }
    QComboBox:!editable:on, QComboBox::drop-down:editable:on {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #383838, stop:0.4 #3c3c3c, stop:0.5 #404040, stop:1.0 #484848);
    }
    QComboBox::drop-down {
        subcontrol-origin: padding; subcontrol-position: top right; width: 15px;
        border-left-width: 1px; border-left-color: #4a4a4a; border-left-style: solid;
        border-top-right-radius: 3px; border-bottom-right-radius: 3px;
    }
    QComboBox::down-arrow { image: url(down_arrow_dark.png); /* Ensure down_arrow_dark.png exists or use unicode */ }
    QGroupBox {
        background-color: #3c3c3c; border: 1px solid #4a4a4a; border-radius: 5px;
        margin-top: 1ex; font-weight: bold;
    }
    QGroupBox::title {
        subcontrol-origin: margin; subcontrol-position: top left;
        padding: 0 3px; color: #cfcfcf;
    }
    #VideoLabel { background-color: black; border: 1px solid #4a4a4a; } /* Used for main_video_label, practice_video_label */
    #LearningImageLabel { background-color: transparent; border: none; font-size: 16px; color: #aaa; }
    #PhraseLabel { font-size: 16px; background-color: #383838; color: #cfcfcf; padding: 5px; border-radius: 5px; }
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
        self.alphabet_path = alphabet_path # Currently not used directly in run()
        self.labels_dict = labels_dict
        self.cap = None
        self.hands_instance = None
        self.model = None
        self.hands_solution = mp.solutions.hands
        self.running = True

    def run(self):
        mp_drawing = mp.solutions.drawing_utils
        # mp_drawing_styles = mp.solutions.drawing_styles # Available if needed for custom landmark styles

        try:
            from tensorflow.keras.models import load_model # Import here to keep TensorFlow ops in this thread
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

        camera_indices_to_try = [0, 1, 2] # Try common camera indices
        for camera_index in camera_indices_to_try:
            cap_test = cv2.VideoCapture(camera_index)
            if cap_test.isOpened():
                self.cap = cap_test
                self.status_update.emit(f"Successfully opened camera with index: {camera_index}")
                break
            cap_test.release()
        
        if self.cap is None:
            self.status_update.emit("Error: Camera not found. Please check connection.")
            return

        while self.running:
            if not self.cap or not self.cap.isOpened():
                self.status_update.emit("Error: Camera disconnected or failed.")
                self.running = False
                break

            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.status_update.emit("Error capturing frame.")
                time.sleep(0.05) 
                continue

            H, W, _ = frame.shape
            
            frame_rgb_for_mediapipe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb_for_mediapipe.flags.writeable = False # Performance improvement
            results = self.hands_instance.process(frame_rgb_for_mediapipe)
            
            predicted_char_to_emit = ""
            confidence_to_emit = 0.0

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks on the display frame
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        mp.solutions.drawing_styles.get_default_hand_connections_style()
                    )
                   
                    # Data extraction for prediction
                    data_aux = []
                    x_coords_norm = [lm.x for lm in hand_landmarks.landmark]
                    y_coords_norm = [lm.y for lm in hand_landmarks.landmark]
                    if not x_coords_norm or not y_coords_norm: continue

                    min_x_norm, min_y_norm = min(x_coords_norm), min(y_coords_norm)
                    for i in range(len(x_coords_norm)):
                        data_aux.append(x_coords_norm[i] - min_x_norm)
                        data_aux.append(y_coords_norm[i] - min_y_norm)

                    if self.model and len(data_aux) == self.model.input_shape[1]: # Ensure data matches model input
                        prediction_array = np.array([data_aux])
                        try:
                            prediction_result = self.model.predict(prediction_array, verbose=0)
                            max_prob = np.max(prediction_result[0])
                            predicted_index = np.argmax(prediction_result[0])
                            predicted_character = self.labels_dict.get(predicted_index, '?') # Use .get for safety
                            
                            if 'A' <= predicted_character <= 'Z' and max_prob >= 0.95:
                                predicted_char_to_emit = predicted_character 
                                confidence_to_emit = float(max_prob)

                                x_coords_abs = [int(x * W) for x in x_coords_norm]
                                y_coords_abs = [int(y * H) for y in y_coords_norm]
                                max_x_for_text = max(x_coords_abs) if x_coords_abs else W 
                                min_y_for_text = min(y_coords_abs) if y_coords_abs else 0
                                
                                text_x = max_x_for_text + 10
                                text_y = max(30, min_y_for_text) 
                                text_y = min(text_y, H - 10) 

                                text_to_draw = f"{predicted_char_to_emit} ({confidence_to_emit:.0%})"
                                cv2.putText(frame, text_to_draw, (text_x, text_y), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                        except Exception as e_predict:
                            self.status_update.emit(f"Model prediction error: {e_predict}")
            
            if predicted_char_to_emit:
                self.prediction_ready.emit(predicted_char_to_emit)

            frame_for_display_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            q_image = QImage(frame_for_display_rgb.data, W, H, frame_for_display_rgb.strides[0], QImage.Format_RGB888)
            self.frame_ready.emit(q_image.copy())

            time.sleep(0.01) # Control processing speed

        if self.cap:
            self.cap.release()
            self.status_update.emit("Camera released.")
        if self.hands_instance:
            try: self.hands_instance.close()
            except: pass # MediaPipe Hands might not have explicit close in all versions or contexts
            self.status_update.emit("MediaPipe Hands resources released.")
        self.status_update.emit("CameraProcessingThread finished.")

    def stop(self):
        self.running = False
        self.status_update.emit("Stopping CameraProcessingThread...")


class ASLRecognitionApp(QMainWindow):
    TARGET_PRACTICE_IMAGE_SIZE = (200, 200)

    def __init__(self):
        super().__init__()
        self.current_practice_mode = "Random"
        self.manual_target_letter = "A"
        self.setWindowTitle(WINDOW_NAME)
        if os.path.exists(ICON_PATH):
            try: self.setWindowIcon(QIcon(ICON_PATH))
            except Exception as e: print(f"Warning: Could not load main window icon '{ICON_PATH}': {e}")
        
        self.setGeometry(100, 100, 900, 800) # Default window size

        self.detected_phrase = ""
        self.last_detected_letter = ""
        self.last_detection_time = 0
        self.continuous_detection_start = 0
        self.current_learning_letter = ""
        self.learning_mode_active = False
        
        self.labels_dict = {i: chr(65 + i) for i in range(26)} # A-Z

        self.settings = QSettings("ASLApp", "Preferences")
        self._load_settings()
        self._setup_ui()
        self.apply_theme(self.current_theme_name)

        self.camera_thread = CameraProcessingThread(MODEL_PATH, ALPHABET_PATH, self.labels_dict)
        self.camera_thread.frame_ready.connect(self.display_video_frame)
        self.camera_thread.prediction_ready.connect(self.update_prediction)
        self.camera_thread.status_update.connect(self.handle_status_update)
        self.camera_thread.model_loaded.connect(self.on_model_loaded)
        self.camera_thread.mediapipe_loaded.connect(self.on_mediapipe_loaded)

        if hasattr(self, 'practice_mode_combo'):
            self.practice_mode_combo.currentTextChanged.connect(self.on_practice_mode_changed)
        if hasattr(self, 'letter_select_combo'):
            self.letter_select_combo.currentTextChanged.connect(self.on_manual_letter_selected)
        
        self.camera_thread.start()
        print("ASLRecognitionApp initialized and camera thread started.")

    @Slot(QImage)
    def display_video_frame(self, q_image):
        pixmap = QPixmap.fromImage(q_image)
        current_tab_widget = self.tabs.currentWidget()
        target_label = None
        if current_tab_widget == self.recognition_tab:
            target_label = self.main_video_label
        elif current_tab_widget == self.practice_tab:
            target_label = self.practice_video_label
        
        if target_label and hasattr(target_label, 'setPixmap'):
             target_label.setPixmap(pixmap.scaled(target_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @Slot(str)
    def update_prediction(self, predicted_character):
        current_time = time.time()

        # Ensure phrase_label style is set based on current theme
        default_bg_color = '#383838' if self.current_theme_name == 'Dark' else '#e0e0e0'
        default_text_color = '#cfcfcf' if self.current_theme_name == 'Dark' else 'black'
        self.phrase_label.setStyleSheet(
            f"font-size: 16px; background-color: {default_bg_color}; "
            f"color: {default_text_color}; padding: 5px; border-radius: 5px;"
        )

        if predicted_character: 
            # Confidence check for A-Z is handled in CameraProcessingThread
            if self.learning_mode_active:
                if self.current_learning_letter == "DONE": # All letters practiced in current random session
                    self.phrase_label.setText(f"Target: {self.current_learning_letter} | You: {predicted_character}")
                    return 
                elif not self.current_learning_letter: 
                    self.load_learning_image_pyside() # Load first image if none shown yet

                if predicted_character == self.current_learning_letter:
                    if self.continuous_detection_start == 0: self.continuous_detection_start = current_time
                    # Check if the correct letter has been held for the required duration
                    if current_time - self.continuous_detection_start >= 1.5: # Time in seconds
                        print(f"INFO: Correctly practiced letter: {predicted_character}.")
                        self.load_learning_image_pyside(exclude_letter=self.current_learning_letter)
                        self.continuous_detection_start = 0 # Reset for next letter
                else:
                    # Reset if the wrong letter is shown
                    self.continuous_detection_start = 0
                self.phrase_label.setText(f"Target: {self.current_learning_letter} | You: {predicted_character}")
            else: # Normal recognition mode
                # Add to phrase if different from last, or if enough time has passed (debounce)
                if predicted_character != self.last_detected_letter or \
                   (current_time - self.last_detection_time >= 1.5): # Time in seconds
                    self.detected_phrase += predicted_character
                    self.last_detected_letter = predicted_character
                    self.last_detection_time = current_time
                    print(f"INFO: Stored: {predicted_character}. Current Phrase: '{self.detected_phrase}'")
                self.phrase_label.setText(f"Sign: {predicted_character} | Phrase: {self.detected_phrase}")
        else: # No valid character predicted (e.g., empty string from thread)
            self.phrase_label.setText(f"Phrase: {self.detected_phrase}")


    @Slot(str)
    def handle_status_update(self, message):
        print(f"THREAD_STATUS: {message}")
        # Display critical errors or important status updates in the UI
        if "Error" in message or "Failed" in message or "not found" in message:
             self.phrase_label.setText(message) 

    @Slot(bool)
    def on_model_loaded(self, loaded):
        if loaded:
            print("Model successfully loaded by CameraProcessingThread.")
        else:
            print("Model loading failed in CameraProcessingThread.")
            error_message = "Error: Keras model failed to load. Check logs and model file."
            if hasattr(self, 'main_video_label'): self.main_video_label.setText(error_message)
            if hasattr(self, 'practice_video_label'): self.practice_video_label.setText(error_message)

    @Slot(bool)
    def on_mediapipe_loaded(self, loaded):
        if loaded:
            print("MediaPipe successfully loaded by CameraProcessingThread.")
        else:
            print("MediaPipe loading failed in CameraProcessingThread.")
            error_message = "Error: MediaPipe Hands failed to load. Check installation."
            if hasattr(self, 'main_video_label'): self.main_video_label.setText(error_message)
            if hasattr(self, 'practice_video_label'): self.practice_video_label.setText(error_message)

    def _load_settings(self):
        self.current_theme_name = self.settings.value("theme", "Light") # Default to Light theme
        print(f"Loaded settings: Theme='{self.current_theme_name}'")

    def _save_settings(self):
        self.settings.setValue("theme", self.current_theme_name)
        print(f"Saved settings: Theme='{self.current_theme_name}'")

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # --- Recognition Tab ---
        self.recognition_tab = QWidget()
        recognition_layout = QVBoxLayout(self.recognition_tab)
        self.main_video_label = QLabel("Initializing Camera...")
        self.main_video_label.setObjectName("VideoLabel") # For consistent styling
        self.main_video_label.setAlignment(Qt.AlignCenter)
        self.main_video_label.setMinimumSize(640, 480) 
        recognition_layout.addWidget(self.main_video_label)

        self.phrase_label = QLabel("Initializing components...")
        self.phrase_label.setObjectName("PhraseLabel")
        self.phrase_label.setFixedHeight(40) 
        self.phrase_label.setAlignment(Qt.AlignCenter)
        recognition_layout.addWidget(self.phrase_label)
        
        recognition_controls_layout = QHBoxLayout()
        recognition_controls_layout.addStretch() # Push button to the right
        self.reset_button = QPushButton()
        if os.path.exists("reset.png"): # Check for reset icon
            try:
                self.reset_button.setIcon(QIcon("reset.png"))
                self.reset_button.setIconSize(QSize(32, 32)); self.reset_button.setFixedSize(QSize(48,48))
            except Exception as e_icon:
                 print(f"Warning: Failed to load reset.png as icon: {e_icon}. Using text 'Reset'."); self.reset_button.setText("Reset")
        else:
            self.reset_button.setText("Reset"); print("Warning: reset.png for reset button not found. Using text 'Reset'.")
        self.reset_button.setToolTip("Reset the detected phrase")
        self.reset_button.clicked.connect(self.reset_detected_text_action)
        recognition_controls_layout.addWidget(self.reset_button)
        recognition_layout.addLayout(recognition_controls_layout)
        self.tabs.addTab(self.recognition_tab, "Recognition")

        # --- Practice Tab ---
        self.practice_tab = QWidget()
        practice_tab_main_v_layout = QVBoxLayout(self.practice_tab)

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
        practice_tab_main_v_layout.addLayout(practice_controls_layout)

        video_image_layout = QHBoxLayout()
        
        self.practice_video_label = QLabel("Camera Feed")
        self.practice_video_label.setObjectName("PracticeVideoLabel") # Restored specific object name
        self.practice_video_label.setAlignment(Qt.AlignCenter)
        self.practice_video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.practice_video_label.setScaledContents(False)
        video_image_layout.addWidget(self.practice_video_label, 85) # Video on left, stretch factor 85

        self.learning_image_display_label = QLabel("Select 'Practice' tab to start learning.")
        self.learning_image_display_label.setObjectName("LearningImageLabel")
        self.learning_image_display_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.learning_image_display_label.setMinimumSize(QSize(250, 250)) # Restored minimum size
        video_image_layout.addWidget(self.learning_image_display_label, 3) # Image on right, stretch factor 3
        
        practice_tab_main_v_layout.addLayout(video_image_layout)
        self.tabs.addTab(self.practice_tab, "Practice")
        self.tabs.currentChanged.connect(self.handle_tab_change)

        # --- Settings GroupBox ---
        settings_groupbox = QGroupBox("Settings")
        settings_layout = QFormLayout(settings_groupbox)
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark"])
        self.theme_combo.setCurrentText(self.current_theme_name)
        self.theme_combo.currentTextChanged.connect(self.apply_theme_from_combo)
        settings_layout.addRow(QLabel("Theme:"), self.theme_combo)
        main_layout.addWidget(settings_groupbox)
        
        print("UI setup complete.")

    def _setup_resources(self):
        # This method is kept for potential future use if other non-thread resources are needed.
        pass

    @Slot(str)
    def apply_theme_from_combo(self, theme_name):
        self.apply_theme(theme_name)

    def apply_theme(self, theme_name):
        print(f"Applying theme: {theme_name}")
        self.current_theme_name = theme_name
        if theme_name == "Dark":
            QApplication.instance().setStyleSheet(DARK_THEME_STYLESHEET)
        else: # Default to Light theme
            QApplication.instance().setStyleSheet(LIGHT_THEME_STYLESHEET)
        self._save_settings() # Save theme preference

    @Slot(int)
    def handle_tab_change(self, index):
        current_tab_text = self.tabs.tabText(index)
        print(f"Switched to tab: '{current_tab_text}' (Index: {index})")
        if current_tab_text == "Practice":
            self.learning_mode_active = True
            self.on_practice_mode_changed(self.practice_mode_combo.currentText()) # Load/update image based on mode
            self.last_detection_time = time.time() # Reset debounce timer
            self.detected_phrase = "" # Clear phrase from recognition mode
            self.phrase_label.setText("Detected Phrase: ") # Clear UI label
            print("Learning mode activated.")
        else: # Switched to Recognition tab or any other
            self.learning_mode_active = False
            self.current_learning_letter = "" # Clear current learning letter
            if hasattr(self, 'learning_image_display_label'):
                 self.learning_image_display_label.setText("Switch to 'Practice' tab for learning mode.")
                 self.learning_image_display_label.setPixmap(QPixmap()) # Clear example image
            print("Learning mode deactivated.")

    @Slot(str)
    def on_practice_mode_changed(self, mode):
        self.current_practice_mode = mode
        print(f"INFO: Practice mode changed to: {mode}")
        if mode == "Manual":
            self.letter_select_combo.setEnabled(True)
            self.manual_target_letter = self.letter_select_combo.currentText()
            self.load_learning_image_pyside(force_letter=self.manual_target_letter)
        else: # Random mode
            self.letter_select_combo.setEnabled(False)
            self.load_learning_image_pyside() # Load a random image

    @Slot(str)
    def on_manual_letter_selected(self, letter):
        # This slot is triggered when the user selects a letter in Manual mode
        if self.current_practice_mode == "Manual": 
            self.manual_target_letter = letter
            print(f"INFO: Manual letter selected: {letter}")
            self.load_learning_image_pyside(force_letter=letter)

    def load_learning_image_pyside(self, exclude_letter=None, force_letter=None):
        if not hasattr(self, 'learning_image_display_label'):
            print("ERROR: Learning image display label not found in UI.")
            return

        if not os.path.exists(ALPHABET_PATH) or not os.path.isdir(ALPHABET_PATH):
            print(f"ERROR: Alphabet examples directory '{ALPHABET_PATH}' not found.")
            self.learning_image_display_label.setText("Alphabet image folder not found.")
            return
        
        selected_image_name = None
        try:
            if force_letter: # Specific letter for Manual mode or initial load
                self.current_learning_letter = force_letter.upper()
            elif self.current_practice_mode == "Manual": # Re-assert manual letter if no force_letter
                self.current_learning_letter = self.manual_target_letter.upper()
            else: # Random mode
                image_files = [f for f in os.listdir(ALPHABET_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and os.path.isfile(os.path.join(ALPHABET_PATH, f))]
                if not image_files:
                    self.learning_image_display_label.setText("No images in alphabet folder."); self.current_learning_letter = ""; return
                if exclude_letter: # If practicing, exclude the letter just shown
                    image_files = [img for img in image_files if os.path.splitext(img)[0].upper() != exclude_letter.upper()]
                if not image_files: # All letters (excluding current one) have been shown
                    self.learning_image_display_label.setText("All letters practiced! Reset or change mode."); 
                    self.current_learning_letter = "DONE"; return
                selected_image_name = random.choice(image_files)
                self.current_learning_letter = os.path.splitext(selected_image_name)[0].upper()

            # If a letter is determined (not random choice), find its image file
            if not selected_image_name and self.current_learning_letter and self.current_learning_letter != "DONE":
                possible_extensions = ['.png', '.jpg', '.jpeg'] 
                for ext in possible_extensions:
                    potential_file = f"{self.current_learning_letter}{ext}"
                    if os.path.isfile(os.path.join(ALPHABET_PATH, potential_file)):
                        selected_image_name = potential_file; break
            
            if not selected_image_name and self.current_learning_letter != "DONE": # Check if image was found
                 print(f"ERROR: Image for letter '{self.current_learning_letter}' not found in '{ALPHABET_PATH}'.")
                 self.learning_image_display_label.setText(f"Image for '{self.current_learning_letter}' not found.")
                 self.current_learning_letter = ""; return # Reset if image not found

            if self.current_learning_letter == "DONE": return # Stop if all practiced

            image_full_path = os.path.join(ALPHABET_PATH, selected_image_name)
            
            pil_img = Image.open(image_full_path)
            pil_img = pil_img.resize(self.TARGET_PRACTICE_IMAGE_SIZE, Image.Resampling.LANCZOS)
            
            # Determine QImage format
            qimage_format = QImage.Format_RGB888 # Default
            if pil_img.mode == "RGBA": qimage_format = QImage.Format_RGBA8888
            elif pil_img.mode != "RGB": pil_img = pil_img.convert("RGB") # Convert other modes to RGB
            
            img_data = pil_img.tobytes("raw", pil_img.mode)
            num_channels = len(pil_img.getbands()) # Get number of channels from PIL image
            q_img = QImage(img_data, pil_img.width, pil_img.height, pil_img.width * num_channels, qimage_format)
            
            q_pixmap = QPixmap.fromImage(q_img)
            # Scale pixmap to fit the label while maintaining aspect ratio
            self.learning_image_display_label.setPixmap(q_pixmap.scaled(
                self.learning_image_display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            print(f"INFO: Practice Tab: Displaying image for letter '{self.current_learning_letter}'.")

        except FileNotFoundError:
            print(f"ERROR: Image file '{selected_image_name}' not found at '{image_full_path}'.")
            self.learning_image_display_label.setText(f"Image for '{self.current_learning_letter}' missing.")
            self.current_learning_letter = ""
        except Exception as e:
            print(f"ERROR: Could not load learning image '{selected_image_name}': {e}")
            self.learning_image_display_label.setText("Error loading image.")
            self.current_learning_letter = "" 

    @Slot()
    def reset_detected_text_action(self):
        self.detected_phrase = ""
        self.last_detected_letter = ""
        self.last_detection_time = 0
        self.continuous_detection_start = 0 # Also reset continuous detection for learning mode consistency
        
        # Reset phrase label to a default state reflecting current theme
        default_bg_color = '#383838' if self.current_theme_name == 'Dark' else '#e0e0e0'
        default_text_color = '#cfcfcf' if self.current_theme_name == 'Dark' else 'black'
        self.phrase_label.setStyleSheet(
            f"font-size: 16px; background-color: {default_bg_color}; "
            f"color: {default_text_color}; padding: 5px; border-radius: 5px;"
        )
        self.phrase_label.setText("Detected Phrase: ") 
        print("INFO: Detected phrase and related states reset.")

    def closeEvent(self, event):
        print("INFO: Close event triggered for main window.")
        if hasattr(self, 'camera_thread') and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread.wait(2000) # Wait up to 2 seconds for thread to finish gracefully
        
        # cv2.destroyAllWindows() # Generally not needed if OpenCV windows are managed within the thread
        print("INFO: Application resources released. Exiting.")
        event.accept()
        QApplication.instance().quit() # Ensure application quits properly


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Critical pre-check for model file
    if not os.path.exists(MODEL_PATH):
         print(f"CRITICAL ERROR: Model file '{MODEL_PATH}' not found. The application cannot function without it.")
         msg_box = QMessageBox()
         msg_box.setIcon(QMessageBox.Critical)
         msg_box.setWindowTitle("Application Error")
         msg_box.setText(f"Model file '{MODEL_PATH}' is missing.\nThe application requires this file to function and will now exit.")
         msg_box.setStandardButtons(QMessageBox.Ok)
         msg_box.exec()
         sys.exit(1) # Exit if model is missing; use specific exit code

    main_window = ASLRecognitionApp() # Initialization starts the camera thread
    main_window.show()
    
    exit_code = 0
    try:
        exit_code = app.exec()
    except KeyboardInterrupt:
        print("INFO: Application interrupted by user (Ctrl+C from console).")
        # main_window.close() will be called in finally block if window exists
    except Exception as e_main_exec:
        print(f"CRITICAL: Unhandled exception in application exec loop: {e_main_exec}")
        exit_code = 1 # Indicate error
    finally:
        # Ensure graceful shutdown if the main window object exists and is visible
        if 'main_window' in locals() and main_window and not main_window.isHidden():
             main_window.close() # This will trigger closeEvent if not already called
        sys.exit(exit_code)