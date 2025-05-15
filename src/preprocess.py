import pandas as pd
from sklearn.model_selection import train_test_split

def clean_data(df):
    """
    Cleans the dataset by handling missing values and dropping duplicates.
    """
    # Drop duplicates if any
    df = df.drop_duplicates()
    
    # Handle missing values (e.g., dropping rows with missing targets or filling features)
    if df.isnull().sum().any():
        df = df.dropna()  # Minimal approach: drop rows with missing values
    
    return df

def preprocess_data(df):
    """
    Preprocesses the data:
    1. One-hot encoding for categorical variables
    2. Separating features (X) and target (y)
    """
    # One-hot encoding for the 'Location' column
    df_encoded = pd.get_dummies(df, columns=['Location'], drop_first=True)
    
    # Ensure boolean columns are converted to int (True->1, False->0) to match baseline if needed
    for col in df_encoded.columns:
        if df_encoded[col].dtype == bool:
            df_encoded[col] = df_encoded[col].astype(int)
    
    # Separate features and target
    X = df_encoded.drop("Price", axis=1)
    y = df_encoded["Price"]
    
    return X, y, X.columns.tolist()

def get_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Splits the data into training and testing sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
