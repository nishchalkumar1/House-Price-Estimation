# House Price Estimation

A professional Machine Learning portfolio project I built to predict house prices based on features such as area, number of bedrooms, and location.

## What I Did in This Project
I designed and implemented an end-to-end Machine Learning pipeline from scratch. Specifically, I:
1. **Cleaned & Preprocessed the Data**: I handled missing values, dealt with categorical variables via one-hot encoding, and split the data into training and test sets.
2. **Built the Models**: I trained a baseline Linear Regression model and then developed a Random Forest Regressor to capture non-linear relationships. 
3. **Evaluated Performance**: I compared my models using Mean Squared Error (MSE) and R² Scores, pushing the Random Forest model to achieve over 98% accuracy on the test set.
4. **Visualized the Data**: I generated a Feature Correlation Heatmap to understand the underlying relationships in the dataset and plotted Feature Importances to show which factors impacted the house price the most.
5. **Created an Inference Pipeline**: I wrapped the best model in a clean, reusable `HousePricePredictor` class so the system can easily predict prices given an area, number of bedrooms, and location.

## Dataset Description
The dataset I used contains the following features:
- **Area**: Total square footage of the house.
- **Bedrooms**: Total number of bedrooms (BHK).
- **Location**: Categorical variable (e.g., Bangalore, Delhi, Mumbai).
- **Price**: Target variable representing the cost of the house (in ₹).

## Technologies I Used
- **Python**: Core programming language.
- **Pandas & NumPy**: Data manipulation and numerical operations.
- **Scikit-Learn**: Machine learning modeling, preprocessing, and evaluation.
- **Matplotlib & Seaborn**: Data visualization.
- **Joblib**: Model serialization.

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repo_url>
   cd House-Price-Estimation
   ```

2. **Create a virtual environment** (Optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Main Pipeline**:
   ```bash
   python main.py
   ```
   This will train the models, output the evaluation metrics, save my plots to the `models/` folder, and run some sample predictions.

## Future Improvements I Plan to Add
- Integrate an interactive Web App using Streamlit or Flask so users can input values easily.
- Gather more features (e.g., proximity to city center, age of the property).
- Implement hyperparameter tuning (GridSearchCV/RandomizedSearchCV) for the Random Forest model.
