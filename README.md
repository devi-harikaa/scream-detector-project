# Scream Detection

This project implements a deep learning model to detect screams in audio, distinguishing them from ambient sounds and conversations using TensorFlow. The model achieves ~85% accuracy on a multi-class classification task (Screams, Ambient, Conversations) and includes interpretability analysis with SHAP.

📁 Project Structure
data/ – All audio datasets

ambient/ – Raw ambient audio (from UrbanSound8K)

ambient_converted/ – Preprocessed ambient audio (16 kHz, mono)

screams/ – Scream audio samples (74 WAV files)

conversations/ – Conversation audio (~70 WAV files)

images/ – Visual outputs for evaluation and interpretation

confusion_matrix.png – Confusion matrix from evaluation

shap_class_0.png – SHAP plot for Ambient class

shap_class_1.png – SHAP plot for Screams class

shap_class_2.png – SHAP plot for Conversations class

models/ – Trained model files

scream_model.h5 – Final trained model

src/ – Source code and helper scripts

train.py – Script to train the model

evaluate.py – Model evaluation script

interpret.py – SHAP interpretability tool

preprocess.py – Audio preprocessing utilities

convert_audio.py – Audio format conversion script

filter_urbansound8k.py – Filter audio from UrbanSound8K

model.py – CNN model architecture

UrbanSound8K/

FREESOUNDCREDITS.txt – License and attribution info

metadata/UrbanSound8K.csv – UrbanSound8K metadata

venv/ – Python virtual environment (excluded from version control)

requirements.txt – Python dependencies list

README.md – Project overview and documentation


---

## 🎧 Dataset

- **Total**: 644 WAV files (mono, 16 kHz)
- **Classes**:
  - Screams: 74 samples
  - Ambient: ~500 samples (many from UrbanSound8K)
  - Conversations: 70 samples
- **Features**: MFCCs with shape `(128, 94)` reshaped to `(128, 94, 1)` for CNN input

> 📦 **Source**: Ambient audio partially derived from the [UrbanSound8K dataset](https://urbansounddataset.weebly.com/urbansound8k.html). After downloading, place relevant files under `data/ambient/`. See `src/UrbanSound8K/FREESOUNDCREDITS.txt` for attributions.

---

## 📊 Model Performance

- **Training Accuracy**: ~80.41% (with early stopping at 6 epochs)
- **Test Accuracy**: 85%
- **F1-Scores**:
  - Screams: 0.88
  - Ambient: 0.91
  - Conversations: 0.03 ⚠️ *(due to class imbalance)*

### Confusion Matrix
![Confusion Matrix](images/confusion_matrix.png)

### SHAP Interpretability
- **Ambient**:
  ![SHAP Ambient](images/shap_class_0.png)
- **Screams**:
  ![SHAP Screams](images/shap_class_1.png)
- **Conversations**:
  ![SHAP Conversations](images/shap_class_2.png)

---

## 🧰 Requirements

- Python 3.8+
- Dependencies (see `requirements.txt`):
  - `tensorflow`
  - `librosa`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `shap`
  - `lime`
  - `pydub`
  - `ipython` *(optional, for SHAP visualizations)*

---

## ⚙️ Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/devi-harikaa/scream-detector-project.git
   cd scream-detector-project
Create Virtual Environment

bash
Copy
Edit
python -m venv venv
source venv/bin/activate       # For Linux/Mac
.\venv\Scripts\activate        # For Windows
Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
Prepare Dataset

Download UrbanSound8K

Organize files into:

data/screams/

data/ambient_converted/

data/conversations/

Ensure all WAV files are 16 kHz, mono.

Use src/convert_audio.py and src/filter_urbansound8k.py as needed.

🧪 Usage
Train the Model
bash
Copy
Edit
python src/train.py
Model saved to models/scream_model.h5

Evaluate the Model
bash
Copy
Edit
python src/evaluate.py
Outputs evaluation metrics and saves images/confusion_matrix.png

Interpret the Model with SHAP
bash
Copy
Edit
python src/interpret.py
Generates SHAP plots for all three classes in images/
![shap_class_2](https://github.com/user-attachments/assets/25d5b8f8-096b-4f81-93e6-d00925b50fde)
![shap_class_1](https://github.com/user-attachments/assets/eb2d0ec9-785c-4aee-97d8-69e60c8ca1b1)
![shap_class_0](https://github.com/user-attachments/assets/367bf89d-7168-463a-852d-c1cd6331f129)
![confusion_matrix](https://github.com/user-attachments/assets/8a5cde2d-217d-451e-9bb7-02f4257a5c5b)

⚠️ Notes
Class Imbalance: Low F1 for Conversations is due to fewer training samples. Consider oversampling or using class weights in train.py.

SHAP Compatibility: Uses GradientExplainer to support TensorFlow models with batch normalization layers.

CPU/GPU: Runs on CPU by default. For GPU support, install CUDA 11.0, cuDNN 8.0, and tensorflow-gpu.

🔮 Future Improvements
Augment dataset with more Screams and Conversations

Improve generalization with dropout/regularization

Use class weights for better balance during training

Extend project to support real-time scream detection from live microphone input

📝 License
MIT License

📬 Contact
For queries or suggestions, contact Neeharika at: harikadevi414@gmail.com
