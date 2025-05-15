import pandas as pd
from src.utils import load_model

class HousePricePredictor:
    def __init__(self, model_path="models/linear_regression_model.pkl", expected_features=None):
        """Initializes the predictor by loading the trained model."""
        self.model = load_model(model_path)
        self.expected_features = expected_features or []

    def predict(self, area, bedrooms, location):
        """Predicts the house price based on inputs."""
        # Create a base dictionary with 0s for all expected features
        input_data = {feature: [0] for feature in self.expected_features}
        
        # Populate known values
        if 'Area' in input_data:
            input_data['Area'][0] = area
        if 'Bedrooms' in input_data:
            input_data['Bedrooms'][0] = bedrooms
            
        # Handle the one-hot encoded location
        location_col = f'Location_{location}'
        if location_col in input_data:
            input_data[location_col][0] = 1
        elif location != "Bangalore" and any(col.startswith('Location_') for col in self.expected_features):
            # If Bangalore is the dropped feature, we just pass (all Location_ cols are 0).
            # Otherwise, check if location is valid
            print(f"Warning: Location '{location}' might not be fully supported by the model's training data.")
            
        input_df = pd.DataFrame(input_data)
        
        # Ensure column order matches training data
        input_df = input_df[self.expected_features]
        
        predicted_price = self.model.predict(input_df)[0]
        return predicted_price
