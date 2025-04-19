import os
import numpy as np
import librosa
import scipy.io.wavfile

# Function to load and preprocess audio into Mel-spectrograms
def load_audio(file_path, sr=16000, duration=3):
    try:
        # Load audio file
        audio, _ = librosa.load(file_path, sr=sr, duration=duration)
        # Ensure fixed length (pad or trim)
        target_length = sr * duration
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = audio[:target_length]
        # Compute Mel-spectrogram
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Function to prepare dataset
def prepare_dataset(data_dir, classes=['screams', 'ambient', 'conversations'], sr=16000):
    X, y = [], []
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        for file in os.listdir(class_dir):
            if file.endswith('.wav'):
                file_path = os.path.join(class_dir, file)
                mel_spec = load_audio(file_path, sr=sr)
                if mel_spec is not None:
                    X.append(mel_spec)
                    y.append(label)
    X = np.array(X)
    y = np.array(y)
    # Normalize Mel-spectrograms
    X = (X - np.mean(X)) / np.std(X)
    return X, y

# Data augmentation (optional, adds noise)
def augment_audio(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    augmented = audio + noise_factor * noise
    return augmented

if __name__ == "__main__":
    # Example usage (for testing)
    data_dir = "../data"
    X, y = prepare_dataset(data_dir)
    print(f"Dataset shape: {X.shape}, Labels: {y.shape}")