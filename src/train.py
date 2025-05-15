import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from src.utils import ensure_dir

def train_linear_regression(X_train, y_train):
    """Trains a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """Trains a Random Forest Regressor."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluates a model and returns MSE and R2 score."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n--- {model_name} Evaluation ---")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    return {"Model": model_name, "MSE": mse, "R2": r2}

def generate_correlation_heatmap(df, output_path="models/correlation_heatmap.png"):
    """Generates and saves a correlation heatmap."""
    ensure_dir(os.path.dirname(output_path))
    plt.figure(figsize=(10, 8))
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=['number'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Correlation heatmap saved to {output_path}")

def generate_feature_importance(model, feature_names, output_path="models/feature_importance.png"):
    """Generates and saves a feature importance plot (for tree-based models)."""
    if hasattr(model, "feature_importances_"):
        ensure_dir(os.path.dirname(output_path))
        importances = model.feature_importances_
        feature_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        feature_df = feature_df.sort_values(by="Importance", ascending=False)
        
        plt.figure(figsize=(8, 6))
        sns.barplot(x="Importance", y="Feature", data=feature_df, palette="viridis", hue="Feature", legend=False)
        plt.title("Feature Importance (Random Forest)")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Feature importance plot saved to {output_path}")
    else:
        print("Model does not support feature importances.")
