import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from preprocess import prepare_dataset

def evaluate_model(model_path, data_dir):
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Load test data
    X, y = prepare_dataset(data_dir)
    X = X[..., np.newaxis]  # Add channel dimension
    
    # Predict
    print("Evaluating model...")
    y_pred = model.predict(X)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Metrics
    print("Classification Report:")
    print(classification_report(y, y_pred_classes, target_names=['Screams', 'Ambient', 'Conversations']))
    
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Screams', 'Ambient', 'Conversations'],
                yticklabels=['Screams', 'Ambient', 'Conversations'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("Confusion matrix saved as 'confusion_matrix.png'")

if __name__ == "__main__":
    model_path = "../models/scream_model.h5"
    data_dir = 'data'
    evaluate_model(model_path, data_dir)