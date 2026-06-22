import pandas as pd
from src.utils import load_data, save_model
from src.preprocess import clean_data, preprocess_data, get_train_test_split
from src.train import (
    train_linear_regression,
    train_random_forest,
    evaluate_model,
    generate_correlation_heatmap,
    generate_feature_importance
)
from src.predict import HousePricePredictor

def main():
    print("=== House Price Prediction Pipeline ===")
    
    # 1. Load Data
    print("\nLoading data..")
    df = load_data("data/house_prices_extended.csv")
    print(df.head())
    
    # 2. Preprocess Data
    print("\nPreprocessing data..")
    df_clean = clean_data(df)
    
    # Generate Heatmap before encoding (or after, depending on preference)
    generate_correlation_heatmap(df_clean)
    
    X, y, feature_names = preprocess_data(df_clean)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    
    print(f"Features: {feature_names}")
    
    # 3. Train & Evaluate Models
    print("\nTraining Linear Regression Model...")
    lr_model = train_linear_regression(X_train, y_train)
    lr_results = evaluate_model(lr_model, X_test, y_test, "Linear Regression")
    
    print("\nTraining Random Forest Model...")
    rf_model = train_random_forest(X_train, y_train)
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    
    # Generate Feature Importance for Random Forest
    generate_feature_importance(rf_model, feature_names)
    
    # Compare Models
    print("\n--- Model Comparison ---")
    comparison_df = pd.DataFrame([lr_results, rf_results])
    print(comparison_df.to_string(index=False))
    
    # 4. Save Models
    print("\nSaving Models...")
    save_model(lr_model, "models/linear_regression.pkl")
    save_model(rf_model, "models/random_forest.pkl")
    
    # 5. Make Sample Predictions
    print("\n--- Sample Predictions (Linear Regression) ---")
    predictor = HousePricePredictor("models/linear_regression.pkl", expected_features=feature_names)
    
    test_cases = [
        {"area": 3000, "bedrooms": 4, "location": "Delhi"},
        {"area": 5000, "bedrooms": 5, "location": "Mumbai"},
        {"area": 4000, "bedrooms": 3, "location": "Bangalore"}
    ]
    
    for case in test_cases:
        price = predictor.predict(**case)
        print(f"Predicted Price for {case['area']} sqft, {case['bedrooms']} BHK in {case['location']}: Rs {price:,.0f}")

if __name__ == "__main__":
    main()
