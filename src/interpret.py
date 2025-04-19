import numpy as np
import tensorflow as tf
import shap
from preprocess import prepare_dataset
import matplotlib.pyplot as plt

def interpret_model(model_path, data_dir):
    # Load model and data
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    print("Loading dataset...")
    X, y = prepare_dataset(data_dir)
    X = X[..., np.newaxis]  # Shape: (644, 128, 94, 1)
    print(f"Dataset shape: {X.shape}, Labels shape: {y.shape}")

    # Select background data for SHAP (e.g., 100 random samples)
    np.random.seed(42)
    background_indices = np.random.choice(X.shape[0], 100, replace=False)
    X_background = X[background_indices]
    print(f"X_background shape: {X_background.shape}")

    # Select a few samples for explanation (e.g., 3 per class)
    indices = []
    for label in range(3):
        class_indices = np.where(y == label)[0]
        if len(class_indices) < 3:
            print(f"Warning: Not enough samples for class {label}. Found {len(class_indices)}.")
            indices.extend(class_indices)
        else:
            indices.extend(np.random.choice(class_indices, 3, replace=False))
    X_sample = X[indices]
    print(f"X_sample shape: {X_sample.shape}")

    # Explain predictions with SHAP
    print("Computing SHAP explanations...")
    explainer = shap.GradientExplainer(model, X_background)
    shap_values = explainer.shap_values(X_sample)
    print(f"SHAP values computed: {len(shap_values)} classes")

    # Visualize SHAP for one sample per class
    class_names = ['Ambient', 'Screams', 'Conversations']
    for i, label in enumerate([0, 1, 2]):
        if i < len(shap_values):  # Ensure shap_values has data for the class
            print(f"Generating SHAP plot for {class_names[label]}...")
            shap.image_plot(shap_values[label][i:i+1], X_sample[i:i+1], show=False)
            plt.savefig(f'shap_class_{label}.png')
            plt.close()
            print(f"SHAP plot for {class_names[label]} saved as 'shap_class_{label}.png'")
        else:
            print(f"No SHAP values for {class_names[label]}.")

if __name__ == "__main__":
    model_path = "../models/scream_model.h5"
    data_dir = 'data'
    interpret_model(model_path, data_dir)