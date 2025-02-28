import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
#This is a test commit to trigger CI/CD
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define file paths
TRAIN_DATA_PATH = 'data/train.csv'
TEST_DATA_PATH = 'data/test.csv'
MODEL_PATH = 'model.pkl'

def load_and_preprocess_data(train_path=TRAIN_DATA_PATH, test_path=TEST_DATA_PATH):
    """
    Load and preprocess the car price data
    """
    # Load the data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Display basic information about the data
    print("Train data columns:", train_data.columns.tolist())
    print("Test data columns:", test_data.columns.tolist())
    
    # Check for missing values
    print("Missing values in train data:")
    print(train_data.isnull().sum())
    
    # Replace '-' with NaN in both datasets
    train_data = train_data.replace('-', np.nan)
    test_data = test_data.replace('-', np.nan)
    
    # Extract features and target variable from training data
    X_train = train_data.drop('Price', axis=1)
    y_train = train_data['Price']
    
    # For test data, we might not have the target variable
    if 'Price' in test_data.columns:
        X_test = test_data.drop('Price', axis=1)
        y_test = test_data['Price']
    else:
        X_test = test_data
        y_test = None
    
    return X_train, y_train, X_test, y_test

def create_preprocessing_pipeline():
    """
    Create a preprocessing pipeline for the car price data
    """
    # Identify numeric and categorical columns
    numeric_features = ['Levy', 'Prod. year', 'Engine volume', 'Mileage', 'Cylinders', 'Airbags']
    categorical_features = ['Manufacturer', 'Model', 'Category', 'Leather interior', 
                           'Fuel type', 'Gear box type', 'Drive wheels', 'Doors', 'Wheel', 'Color']
    
    # Preprocessing for numeric features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def clean_data(X):
    """
    Clean the data by handling special cases and converting data types
    """
    # Make a copy to avoid modifying the original
    X_cleaned = X.copy()
    
    # Convert 'Prod. year' to numeric
    X_cleaned['Prod. year'] = pd.to_numeric(X_cleaned['Prod. year'], errors='coerce')
    
    # Clean 'Engine volume' - extract numeric part
    try:
        if X_cleaned['Engine volume'].dtype == 'object' or isinstance(X_cleaned['Engine volume'].iloc[0], str):
            # For string values like '2.5' or '2.5 L'
            X_cleaned['Engine volume'] = X_cleaned['Engine volume'].astype(str).str.extract(r'(\d+\.?\d*)').squeeze()
            X_cleaned['Engine volume'] = pd.to_numeric(X_cleaned['Engine volume'], errors='coerce')
        else:
            # Already numeric
            X_cleaned['Engine volume'] = pd.to_numeric(X_cleaned['Engine volume'], errors='coerce')
    except Exception as e:
        print(f"Error processing Engine volume: {e}")
        X_cleaned['Engine volume'] = pd.to_numeric(X_cleaned['Engine volume'], errors='coerce')
    
    # Clean 'Mileage' - extract numeric part
    try:
        if X_cleaned['Mileage'].dtype == 'object' or isinstance(X_cleaned['Mileage'].iloc[0], str):
            # For string values like '50000 km'
            X_cleaned['Mileage'] = X_cleaned['Mileage'].astype(str).str.extract(r'(\d+)').squeeze()
            X_cleaned['Mileage'] = pd.to_numeric(X_cleaned['Mileage'], errors='coerce')
        else:
            # Already numeric
            X_cleaned['Mileage'] = pd.to_numeric(X_cleaned['Mileage'], errors='coerce')
    except Exception as e:
        print(f"Error processing Mileage: {e}")
        X_cleaned['Mileage'] = pd.to_numeric(X_cleaned['Mileage'], errors='coerce')
    
    # Convert 'Cylinders' to numeric
    X_cleaned['Cylinders'] = pd.to_numeric(X_cleaned['Cylinders'], errors='coerce')
    
    # Convert 'Airbags' to numeric
    X_cleaned['Airbags'] = pd.to_numeric(X_cleaned['Airbags'], errors='coerce')
    
    # Convert 'Levy' to numeric
    X_cleaned['Levy'] = pd.to_numeric(X_cleaned['Levy'], errors='coerce')
    
    # Handle 'Doors' - convert to categorical
    X_cleaned['Doors'] = X_cleaned['Doors'].astype(str)
    
    return X_cleaned

def train_model():
    """
    Train a regression model for car price prediction
    """
    # Load and preprocess data
    X_train, y_train, X_test, y_test = load_and_preprocess_data()
    
    # Clean the data
    X_train_cleaned = clean_data(X_train)
    X_test_cleaned = clean_data(X_test) if X_test is not None else None
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline()
    
    # Create and train the model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Fit the model
    model.fit(X_train_cleaned, y_train)
    
    # Evaluate the model if test data is available
    if X_test_cleaned is not None and y_test is not None:
        # Drop rows with NaN in y_test
        valid_indices = ~y_test.isna()
        if valid_indices.sum() > 0:
            y_test_valid = y_test[valid_indices]
            X_test_cleaned_valid = X_test_cleaned.loc[valid_indices]
            
            y_pred = model.predict(X_test_cleaned_valid)
            mse = mean_squared_error(y_test_valid, y_pred)
            r2 = r2_score(y_test_valid, y_pred)
            print(f"Mean Squared Error: {mse}")
            print(f"R^2 Score: {r2}")
    
    # Save the model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {MODEL_PATH}")
    
    return model

def load_model():
    """
    Load the trained model from disk
    """
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        return model
    else:
        print("Model file not found. Training a new model...")
        return train_model()

# API endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        print(f"Received data: {data}")
        
        # Convert to DataFrame
        input_data = pd.DataFrame([data])
        print(f"Input data types: {input_data.dtypes}")
        
        # Clean the data
        try:
            input_data_cleaned = clean_data(input_data)
            print(f"Cleaned data: {input_data_cleaned}")
        except Exception as e:
            print(f"Error in clean_data: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        # Load model
        model = load_model()
        
        # Make prediction
        prediction = model.predict(input_data_cleaned)[0]
        print(f"Prediction: {prediction}")
        
        # Return prediction
        return jsonify({
            'prediction': float(prediction),
            'status': 'success'
        })
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

# Route for the home page
@app.route('/')
def home():
    return """
    <html>
        <head>
            <title>Car Price Prediction API</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }
                h1 {
                    color: #333;
                }
                pre {
                    background-color: #f4f4f4;
                    padding: 10px;
                    border-radius: 5px;
                }
            </style>
        </head>
        <body>
            <h1>Car Price Prediction API</h1>
            <p>This API predicts car prices based on various features.</p>
            <h2>How to use:</h2>
            <p>Send a POST request to <code>/predict</code> with the following JSON structure:</p>
            <pre>
{
  "Levy": 1000,
  "Manufacturer": "TOYOTA",
  "Model": "Camry",
  "Prod. year": 2015,
  "Category": "Sedan",
  "Leather interior": "Yes",
  "Fuel type": "Petrol",
  "Engine volume": "2.5",
  "Mileage": "50000 km",
  "Cylinders": 4,
  "Gear box type": "Automatic",
  "Drive wheels": "Front",
  "Doors": "04-May",
  "Wheel": "Left wheel",
  "Color": "Black",
  "Airbags": 8
}
            </pre>
        </body>
    </html>
    """

if __name__ == '__main__':
    # Train the model if it doesn't exist
    if not os.path.exists(MODEL_PATH):
        train_model()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=8080)
