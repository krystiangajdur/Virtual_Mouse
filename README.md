# AI Virtual Mouse with Computer Vision

A professional, object-oriented desktop control system that uses computer vision and Machine Learning (SVM) to translate hand gestures captured by a webcam into real-time mouse movements and clicks. 

The project has been fully refactored from a single-script experiment into a clean, modular, and production-ready **Object-Oriented Programming (OOP)** architecture.

---

## Features & Architecture

The application is split into specialized, decoupled modules adhering to the **Single Responsibility Principle**:

*   **Hand Tracking (`hand_detector.py`):** Leverages Google MediaPipe to extract 21 3D hand landmarks (42 geometric features) from the video stream in real time.
*   **AI Classification (`virtual_mouse_app.py`):** Uses a trained **Support Vector Machine (SVM)** model to classify hand states (`opened` for moving, `closed` for clicking) based on extracted coordinate features.
*   **System Control & Smoothing (`smooth_mouse.py`):** Translates normalized camera coordinates into pixel-perfect screen positions using `PyAutoGUI`. It implements a custom **Exponential Moving Average (EMA)** filter to completely eliminate hand tremor.

### Signal Smoothing Math
To prevent cursor shaking, the movement is smoothed using the following tracking interpolation:

$$X_{new} = X_{prev} + \frac{X_{target} - X_{prev}}{\alpha}$$

Where $\alpha$ is the smoothing factor (set to 3). This creates an incredibly fluid, lag-free user experience.

---

## Project Structure

```text
├── collecting_data.py       # Utility: Captures hand landmarks and saves them to CSV
├── train_svm.py             # Utility: Trains the SVM model and exports the pipeline
├── test_model.py            # Utility: Live model tester via terminal/OpenCV window
│
├── hand_detector.py         # Module: MediaPipe wrapper & feature extractor
├── smooth_mouse.py          # Module: Screen scaler, clicker, and EMA filter
├── virtual_mouse_app.py     # Module: Core application orchestrator
└── virtual_mouse_main.py    # Entry Point: Clean main script to boot the application
```

### Requirements & Installation
Clone the repository and navigate to the project directory.

Install the required dependencies:

pip install opencv-python mediapipe numpy pyautogui joblib scikit-learn
### How to Run
To start the virtual mouse system, simply run the main entry point:

python virtual_mouse_main.py