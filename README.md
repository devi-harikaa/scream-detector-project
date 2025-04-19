# Scream Detection

This project implements a deep learning model to detect screams in audio, distinguishing them from ambient sounds and conversations using TensorFlow. The model achieves ~85% accuracy on a multi-class classification task (Screams, Ambient, Conversations) and includes interpretability analysis with SHAP.

## Project Structure

scream-detection/
├── data/
│   ├── ambient/                # Raw ambient audio files
│   ├── ambient_converted/      # Processed ambient audio (500 WAV files)
│   ├── screams/                # Scream audio (74 WAV files)
│   ├── conversations/          # Conversation audio (~70 WAV files)
├── models/
│   ├── scream_model.h5         # Trained model
├── src/
│   ├── train.py               # Script to train the model
│   ├── evaluate.py            # Script to evaluate model performance
│   ├── interpret.py           # Script for model interpretability with SHAP
│   ├── preprocess.py          # Data preprocessing utilities
├── confusion_matrix.png        # Evaluation confusion matrix
├── shap_class_0.png           # SHAP plot for Ambient class
├── shap_class_1.png           # SHAP plot for Screams class
├── shap_class_2.png           # SHAP plot for Conversations class
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
└── venv/                      # Virtual environment (excluded from repo)

## Dataset

- **Total**: 644 WAV files (16 kHz, mono).
- **Classes**:
  - Screams: 74 samples
  - Ambient: ~500 samples
  - Conversations: 70 samples
- **Features**: MFCCs, shape `(128, 94)`, reshaped to `(128, 94, 1)` for the model.

## Model Performance

- **Training**: ~80.41% test accuracy after 6 epochs (early stopping).
- **Evaluation**: 85% accuracy:
  - F1-score: Screams (0.88), Ambient (0.91), Conversations (0.03)
  - Confusion matrix saved as `confusion_matrix.png`
- **Interpretation**: SHAP `GradientExplainer` generates feature importance plots (`shap_class_0.png`, `shap_class_1.png`, `shap_class_2.png`).

## Requirements

- Python 3.8+
- Dependencies (see `requirements.txt`):
  - tensorflow
  - librosa
  - numpy
  - scikit-learn
  - matplotlib
  - shap
  - lime
  - ipython (optional, for interactive SHAP visualizations)
  - pydub

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/scream-detection.git
   cd scream-detection

Create Virtual Environment:
bash

python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

Install Dependencies:
bash

pip install -r requirements.txt

Prepare Dataset:
Place 16 kHz, mono WAV files in data/screams, data/ambient_converted, and data/conversations.

Usage
Train the Model:
bash

python src/train.py

Saves trained model to models/scream_model.h5.

Evaluate the Model:
bash

python src/evaluate.py

Outputs classification report and saves confusion_matrix.png.

Interpret the Model:
bash

python src/interpret.py

Generates SHAP plots (shap_class_0.png, shap_class_1.png, shap_class_2.png) for feature importance.

**Notes
Class Imbalance: The Conversations class has low performance (F1: 0.03). Consider adding more conversation audio or using class weights in train.py.

SHAP Compatibility: Uses GradientExplainer to support TensorFlow models with batch normalization layers.

CPU/GPU: Runs on CPU by default. For GPU support, install CUDA 11.0, cuDNN 8.0, and tensorflow-gpu.

IPython: Optional for interactive SHAP plots. Install with pip install ipython if needed.

Future Improvements
Balance dataset by adding more Screams and Conversations audio.

Enhance model with dropout, regularization, or class weights to improve Conversations performance.

Implement real-time scream detection with live audio input.

License
MIT License
Contact
For questions, contact Neeharika at [harikadevi414@gmail.com]

