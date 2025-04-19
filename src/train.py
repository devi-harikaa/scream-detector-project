import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from preprocess import prepare_dataset
from model import build_model
import os

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def train_model(data_dir, model_save_path="../models/scream_model.h5", epochs=30):
    # Load and preprocess data
    print("Loading dataset...")
    X, y = prepare_dataset(data_dir)
    print(f"Dataset loaded: {X.shape}, {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1765, random_state=42)  # 0.15/0.85 ≈ 0.1765
    
    # Reshape for CNN (add channel dimension)
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    
    # Build model
    model = build_model(input_shape=X_train.shape[1:])
    
    # Train model
    print("Training model...")
    history = model.fit(X_train, y_train, 
                        validation_data=(X_val, y_val),
                        epochs=epochs, 
                        batch_size=32,
                        callbacks=[
                            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
                        ])
    
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    
    # Save model
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    return history, test_acc

if __name__ == "__main__":
    data_dir = 'data'
    history, test_acc = train_model(data_dir)