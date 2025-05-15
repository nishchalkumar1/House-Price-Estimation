import os
import pandas as pd
import joblib

def load_data(file_path):
    """Loads the dataset from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    return pd.read_csv(file_path)

def save_model(model, file_path):
    """Saves a trained model to disk using joblib."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")

def load_model(file_path):
    """Loads a trained model from disk."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model not found at {file_path}")
    return joblib.load(file_path)

def ensure_dir(directory):
    """Ensures that a directory exists."""
    os.makedirs(directory, exist_ok=True)
