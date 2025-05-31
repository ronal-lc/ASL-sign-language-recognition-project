# ✋ ASL Alphabet Recognizer

[ASL Recognition Demo]!
![asl](https://github.com/user-attachments/assets/52151929-5d3f-41c3-aa53-f02604b9567d)


An interactive application that recognizes American Sign Language (ASL) alphabet letters in real-time using computer vision and deep learning. Perfect for learning ASL or building accessible interfaces!

## 🌟 Features

- **Real-time Recognition**: Instantly identifies ASL letters (A-Z) from webcam input
- **Learning Mode**: Practice ASL with instant feedback on your gestures
- **Modern UI**: Clean interface with light/dark theme support
- **Complete Pipeline**: Includes data collection, processing, and model training tools
- **Cross-Platform**: Works on Windows, macOS, and Linux

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Webcam
- GPU recommended for optimal performance

### Installation
```bash
# Clone repository
git clone https://github.com/ronal-lc/ASL-sign-language-recognition-project.git
cd ASL-sign-language-recognition-project

# Create virtual environment
python -m venv venv

# Activate environment
# Linux/macOS:
source venv/bin/activate
# Windows:
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🛠️ Project Structure

| File/Folder             | Purpose                                                                 |
|-------------------------|-------------------------------------------------------------------------|
| `asl_alphabet.py`       | Main application with GUI interface                                     |
| `collector_images.py`   | Collects training images for each ASL letter                           |
| `create_dataset.py`     | Processes images into landmark datasets                                |
| `data_classify.py`      | Trains the neural network model                                       |
| `data/`                 | Collected image data (organized by letter)                             |
| `alphabet_examples/`    | Example images for practice mode                                      |
| `model.keras`           | Trained Keras model for recognition                                   |
| `data_signs.pickle`     | Processed dataset of hand landmarks                                   |

## 🖥️ Usage

### Launch Main Application
```bash
python asl_alphabet.py
```

**Application Features:**
- **Recognition Tab**: Real-time ASL letter detection from webcam
- **Practice Tab**: Learn ASL with guided exercises and instant feedback
- **Theme Selector**: Switch between light/dark mode

### Build Your Own Model (Optional)
1. **Collect training images**:
   ```bash
   python collector_images.py
   ```

2. **Create dataset from images**:
   ```bash
   python create_dataset.py
   ```

3. **Train new model**:
   ```bash
   python data_classify.py
   ```

## 💡 Tips for Best Results
- Use a plain background with good lighting
- Position your hand clearly in frame
- Start with stationary letters before trying fluid signing
- In practice mode, hold gestures for 1-2 seconds for accurate recognition

## ⚙️ Technical Details
- **Computer Vision**: MediaPipe for hand landmark detection
- **Machine Learning**: TensorFlow/Keras neural network classifier
- **GUI Framework**: PySide6 (Qt for Python)
- **Data Augmentation**: Horizontal flipping for improved model robustness

## ⚠️ Version Compatibility
Tested with:
- Python 3.9
- TensorFlow 2.16.1
- MediaPipe 0.10.21
- OpenCV 4.9.0
- PySide6 6.5.0

## 📜 License
This project is currently unlicensed.
