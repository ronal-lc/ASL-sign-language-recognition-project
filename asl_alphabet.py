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
import logging
from utils import setup_logging

from PySide6.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                               QVBoxLayout, QHBoxLayout, QWidget, QTabWidget,
                               QComboBox, QSlider, QGroupBox, QFormLayout) # Added for new controls
from PySide6.QtGui import QImage, QPixmap, QIcon, QFont
from PySide6.QtCore import QTimer, Qt, QSize, Slot, QSettings

# Import PIL.Image explicitly for image loading in learning mode
from PIL import Image


WINDOW_NAME = 'ASL Alphabet Recognition'
ICON_PATH = 'icono.ico'
ALPHABET_PATH = "alphabet_examples"

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
            except Exception as e: logging.warning(f"Could not load main window icon '{ICON_PATH}': {e}")
        
        self.setGeometry(100, 100, 900, 800) # Increased height for theme/font controls
        # self.setFixedSize(900, 730) # Comment out or adjust if making resizable

        self.detected_phrase = ""
        self.last_detected_letter = ""
        self.last_detection_time = 0
        self.continuous_detection_start = 0
        self.stop_detection_time = 0
        self.stop_detected = False
        self.current_learning_letter = ""
        self.learning_mode_active = False
        
        self.cap = None
        self.hands_solution = None
        self.hands_instance = None
        self.drawing_utils = None
        self.drawing_styles = None
        self.model = None
        self.labels_dict = {i: chr(65 + i) for i in range(26)}
        self.labels_dict[26] = "STOP"

        self.settings = QSettings("ASLApp", "Preferences")
        self._load_settings() # Load theme and font scale

        self._setup_resources()
        self._setup_ui()
        self.apply_theme(self.current_theme_name) # Apply loaded theme
        self.apply_font_scale(self.CURRENT_FONT_SCALE_FACTOR, initial_setup=True) # Apply loaded font scale

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_gui_frame)
        if self.cap and self.cap.isOpened() and self.hands_instance and self.model:
            self.timer.start(30)
        else:
            logging.critical("Failed to initialize critical components. Video feed will not start.")
            if hasattr(self, 'video_label'):
                 self.video_label.setText("Error: Camera or necessary models failed to load. Check logs.")
            else:
                 print("CRITICAL: Camera or necessary models failed to load. Check logs.")
        logging.info("ASLRecognitionApp initialized.")

    def _load_settings(self):
        self.current_theme_name = self.settings.value("theme", "Light") # Default to Light
        try:
            self.CURRENT_FONT_SCALE_FACTOR = float(self.settings.value("font_scale", 1.0))
        except ValueError:
            self.CURRENT_FONT_SCALE_FACTOR = 1.0
        logging.info(f"Loaded settings: Theme='{self.current_theme_name}', FontScale={self.CURRENT_FONT_SCALE_FACTOR}")


    def _save_settings(self):
        self.settings.setValue("theme", self.current_theme_name)
        self.settings.setValue("font_scale", self.CURRENT_FONT_SCALE_FACTOR)
        logging.info(f"Saved settings: Theme='{self.current_theme_name}', FontScale={self.CURRENT_FONT_SCALE_FACTOR}")

    def _setup_ui(self):
        logging.debug("Setting up UI with tabs and theme/font controls.")
        
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
        self.video_label.setMinimumSize(800, 520) # Adjusted size for controls below
        recognition_layout.addWidget(self.video_label)

        self.phrase_label = QLabel("Detected: ")
        self.phrase_label.setObjectName("PhraseLabel")
        self.phrase_label.setFixedHeight(30)
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
                 logging.warning(f"Failed to load reset.png as icon: {e_icon}."); self.reset_button.setText("Reset")
        else:
            self.reset_button.setText("Reset"); logging.warning("reset.png not found.")
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
        
        logging.debug("UI setup with tabs and theme/font controls complete.")


    def _setup_resources(self):
        logging.info("Setting up camera, MediaPipe, and TensorFlow model.")
        model_path = './model.keras'
        if not os.path.exists(model_path):
            logging.critical(f"Model file '{model_path}' not found.")
            return
        try:
            self.model = load_model(model_path)
            logging.info("Keras model loaded successfully.")
        except Exception as e:
            logging.critical(f"Critical error loading Keras model '{model_path}': {e}", exc_info=True)
            return

        camera_indices_to_try = [0, 1, 2]
        for camera_index in camera_indices_to_try:
            cap_test = cv2.VideoCapture(camera_index)
            if cap_test.isOpened():
                self.cap = cap_test
                logging.info(f"Successfully opened camera with index: {camera_index}")
                break
            cap_test.release()
        
        if self.cap is None:
            logging.critical("Could not open webcam.")
            return

        try:
            self.hands_solution = mp.solutions.hands
            self.drawing_utils = mp.solutions.drawing_utils
            self.drawing_styles = mp.solutions.drawing_styles
            self.hands_instance = self.hands_solution.Hands(
                static_image_mode=False, max_num_hands=1,
                min_detection_confidence=0.8, min_tracking_confidence=0.5)
            logging.info("MediaPipe Hands initialized successfully.")
        except Exception as e:
            logging.critical(f"Failed to initialize MediaPipe Hands: {e}", exc_info=True)
            if self.cap: self.cap.release()
            self.cap = None 
            return
        logging.info("Resources setup complete.")

    @Slot(str)
    def apply_theme_from_combo(self, theme_name):
        self.apply_theme(theme_name)

    def apply_theme(self, theme_name):
        logging.info(f"Applying theme: {theme_name}")
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
        
        logging.info(f"Applied font scale: {self.CURRENT_FONT_SCALE_FACTOR:.1f}x, New base point size: {new_size}")


    @Slot(int)
    def handle_tab_change(self, index):
        current_tab_text = self.tabs.tabText(index)
        logging.info(f"Switched to tab: '{current_tab_text}' (Index: {index})")
        if current_tab_text == "Practice":
            self.learning_mode_active = True
            self.load_learning_image_pyside() # Load initial image for practice
            self.last_detection_time = time.time() # Reset timer for practice mode
            self.current_learning_letter = self.current_learning_letter # Keep or reset based on desired logic
            self.detected_phrase = "" # Clear phrase when switching to practice
            self.phrase_label.setText("Detected: ") # Update UI
            logging.info("Learning mode activated (Practice tab).")
        else: # Recognition tab or any other tab
            self.learning_mode_active = False
            self.current_learning_letter = "" # Clear learning letter
            # Optionally, clear learning image display if it's separate and visible
            if hasattr(self, 'learning_image_display_label'):
                 self.learning_image_display_label.setText("Switch to 'Practice' tab to start learning.")
                 self.learning_image_display_label.setPixmap(QPixmap()) # Clear image
            logging.info("Learning mode deactivated (Switched to Recognition or other tab).")


    def load_learning_image_pyside(self, exclude_letter=None):
        # This method now updates self.learning_image_display_label in the "Practice" tab
        if not hasattr(self, 'learning_image_display_label'):
            logging.error("learning_image_display_label not found. Cannot load learning image.")
            return

        if not os.path.exists(ALPHABET_PATH) or not os.path.isdir(ALPHABET_PATH):
            logging.error(f"Alphabet examples directory '{ALPHABET_PATH}' not found.")
            self.learning_image_display_label.setText("Alphabet images not found.")
            return
        try:
            images = [f for f in os.listdir(ALPHABET_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and os.path.isfile(os.path.join(ALPHABET_PATH, f))]
            if not images:
                logging.warning(f"No images found in '{ALPHABET_PATH}'.")
                self.learning_image_display_label.setText("No images in alphabet folder.")
                return
            if exclude_letter:
                images = [img for img in images if os.path.splitext(img)[0].upper() != exclude_letter.upper()]
            if not images:
                logging.info(f"No new images after excluding '{exclude_letter}'. All letters practiced?")
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
            logging.info(f"Practice Tab: Displaying image for letter '{self.current_learning_letter}'.")
        except Exception as e:
            logging.error(f"Error loading learning image for Practice tab: {e}", exc_info=True)
            self.learning_image_display_label.setText("Error loading image.")

    @Slot()
    def reset_detected_text_action(self):
        self.detected_phrase = ""
        self.stop_detected = False
        self.phrase_label.setText("Detected: ") 
        self.phrase_label.setStyleSheet("font-size: 16px; background-color: #e0e0e0; color: black; padding: 5px; border-radius: 5px;")
        logging.info("Detected text and STOP state reset.")

    @Slot()
    def update_gui_frame(self):
        if not self.cap or not self.cap.isOpened(): return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            logging.warning("Error capturing frame for GUI update.")
            return
        
        H, W, _ = frame.shape
        frame_rgb_for_mediapipe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb_for_mediapipe.flags.writeable = False
        results = self.hands_instance.process(frame_rgb_for_mediapipe)
        # frame_rgb_for_mediapipe.flags.writeable = True # Not strictly needed as we draw on 'frame'

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, self.hands_solution.HAND_CONNECTIONS,
                    self.drawing_styles.get_default_hand_landmarks_style(),
                    self.drawing_styles.get_default_hand_connections_style())

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
                    except Exception as e_predict:
                        logging.error(f"Model prediction error: {e_predict}", exc_info=True)
                        continue

                    max_prob = np.max(prediction_result[0])
                    predicted_index = np.argmax(prediction_result[0])
                    predicted_character = self.labels_dict.get(predicted_index, '?')
                    logging.debug(f"Prediction: {predicted_character}, Confidence: {max_prob:.2f}")
                    
                    text_x, text_y = int(min(x_coords) * W) - 10, int(min(y_coords) * H) - 20
                    cv2.putText(frame, f'{predicted_character} ({max_prob * 100:.1f}%)', (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2, cv2.LINE_AA)
                    
                    current_time = time.time()
                    if predicted_character == "STOP" and max_prob >= 0.99:
                        if self.stop_detection_time == 0: self.stop_detection_time = current_time
                        elif current_time - self.stop_detection_time >= 2.0:
                            if not self.stop_detected:
                                self.stop_detected = True
                                logging.info("STOP detected and held.")
                    else: self.stop_detection_time = 0

                    if not self.stop_detected:
                        if predicted_character != "STOP" and 'A' <= predicted_character <= 'Z' and max_prob >= 0.95:
                            if self.learning_mode_active: # Check if practice tab is active
                                if self.current_learning_letter == "DONE": # All letters practiced
                                     pass # Do nothing until tab is switched or app reset
                                elif not self.current_learning_letter: # Load first image if none
                                    self.load_learning_image_pyside()

                                if predicted_character == self.current_learning_letter:
                                    if self.continuous_detection_start == 0: self.continuous_detection_start = current_time
                                    if current_time - self.continuous_detection_start >= 2.0: # Hold for 2s
                                        # In practice mode, we don't add to detected_phrase
                                        # We just log and load next image
                                        logging.info(f"Correctly practiced letter: {predicted_character}.")
                                        self.load_learning_image_pyside(exclude_letter=self.current_learning_letter)
                                        self.continuous_detection_start = 0
                                else: # Predicted wrong letter or not the one to practice
                                    self.continuous_detection_start = 0
                            else: # Normal recognition mode (Recognition Tab)
                                if predicted_character != self.last_detected_letter or (current_time - self.last_detection_time >= 1.5):
                                    self.detected_phrase += predicted_character
                                    self.last_detected_letter = predicted_character
                                    self.last_detection_time = current_time
                                    logging.info(f"Stored: {predicted_character}. Phrase: '{self.detected_phrase}'")
        
        # Update UI elements that are always visible or need updating based on overall state
        self.phrase_label.setText(f"Detected: {self.detected_phrase}")
        self.phrase_label.setStyleSheet(f"font-size: 16px; background-color: {'darkblue' if self.stop_detected else '#e0e0e0'}; color: {'white' if self.stop_detected else 'black'}; padding: 5px; border-radius: 5px;")

        # Update video feed display (always, as it's part of the core loop)
        frame_display_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_display_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(frame_display_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        # Check if recognition tab is active for displaying video, or handle as needed
        # For simplicity, video_label is part of recognition tab, so it updates if that tab is visible.
        # If video needs to be shown in other contexts, this logic might need adjustment.
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


    def closeEvent(self, event):
        logging.info("Close event triggered for main window.")
        self.timer.stop()
        if self.cap and self.cap.isOpened():
            logging.debug("Releasing camera resource.")
            self.cap.release()
        if self.hands_instance:
            logging.debug("Closing MediaPipe Hands resource.")
            self.hands_instance.close()
        cv2.destroyAllWindows()
        logging.info("Application resources released. Exiting.")
        event.accept()
        QApplication.instance().quit() # Ensure application quits properly


if __name__ == "__main__":
    setup_logging(log_level=logging.INFO, log_file="asl_alphabet_pyside.log")
    
    app = QApplication(sys.argv)
    main_window = ASLRecognitionApp()
    
    if main_window.model is None or \
       (main_window.cap is None or not main_window.cap.isOpened()) or \
       main_window.hands_instance is None:
        logging.critical("Application failed to initialize critical components during __init__. Exiting.")
        # Attempt to show a message box if QApplication is available
        try:
            from PySide6.QtWidgets import QMessageBox
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setWindowTitle("Application Error")
            msg_box.setText("Failed to initialize critical components (Camera, AI Model, or Hand Tracking).\nPlease check logs for details.\nThe application will now exit.")
            msg_box.exec()
        except Exception as e_msgbox:
            logging.error(f"Failed to show critical error message box: {e_msgbox}")
        sys.exit(-1)
        
    main_window.show()
    
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        logging.info("Application interrupted by user (Ctrl+C from console).")
        # Ensure cleanup is called if window might not have been fully closed
        if hasattr(main_window, 'closeEvent'): # Check if closeEvent can be called
             main_window.close() # Trigger closeEvent for cleanup
        else: # Manual cleanup if closeEvent not available/triggered
            if main_window.cap and main_window.cap.isOpened(): main_window.cap.release()
            if main_window.hands_instance: main_window.hands_instance.close()
            cv2.destroyAllWindows()
        sys.exit(0)
    except Exception as e_main_exec:
        logging.critical(f"Unhandled exception in application exec loop: {e_main_exec}", exc_info=True)
        sys.exit(-1)