# Car Price Prediction API

A Flask-based RESTful API that uses machine learning to predict car prices based on various features.

## Overview

This application provides a simple yet powerful API for predicting car prices using a Random Forest Regression model. It processes input data containing car specifications and returns a predicted price based on patterns learned from historical data.

## Features

- **Data Preprocessing**: Handles missing values, categorical features, and numerical scaling
- **Machine Learning Model**: Uses Random Forest Regression for accurate price predictions
- **RESTful API**: Simple HTTP interface for making predictions
- **Cross-Origin Resource Sharing (CORS)**: Supports cross-origin requests
- **Comprehensive Testing**: Includes unit tests to ensure functionality

## Project Structure

```
my_flask_app/
├── app.py                # Main Flask application
├── model.pkl             # Trained machine learning model (generated)
├── requirements.txt      # Project dependencies
├── README.md             # This file
├── test_api.py           # API testing script
├── data/                 # Data directory
│   ├── train.csv         # Training data
│   └── test.csv          # Test data
└── tests/                # Test directory
    └── test_app.py       # Unit tests
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup

1. Clone the repository:
   ```
   git clone <repository-url>
   cd my_flask_app
   ```

2. Create a virtual environment:
   ```
   python -m venv .venv
   ```

3. Activate the virtual environment:
   - On macOS/Linux:
     ```
     source .venv/bin/activate
     ```
   - On Windows:
     ```
     .venv\Scripts\activate
     ```

4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Starting the Server

Run the following command to start the Flask server:

```
python app.py
```

The server will start on `http://0.0.0.0:8080/`.

### Making Predictions

Send a POST request to the `/predict` endpoint with car specifications in JSON format:

```bash
curl -X POST -H "Content-Type: application/json" -d '{
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
}' http://localhost:8080/predict
```

### Example Response

```json
{
  "prediction": 27561.88,
  "status": "success"
}
```

## API Endpoints

- `GET /`: Home page
- `POST /predict`: Predicts car price based on provided features

## Data Preprocessing

The application performs several preprocessing steps:

1. **Cleaning**: Handles missing values and extracts numeric parts from text fields
2. **Encoding**: Converts categorical variables to numerical representations
3. **Scaling**: Normalizes numerical features to improve model performance

## Model Training

The model is trained using the following steps:

1. Load and preprocess the training data
2. Create preprocessing pipelines for numerical and categorical features
3. Train a Random Forest Regressor on the processed data
4. Evaluate the model on test data (if available)
5. Save the trained model to disk

## Dependencies

- **Flask**: Web framework
- **Flask-CORS**: Cross-Origin Resource Sharing support
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **pytest**: Testing framework

## Testing

Run the tests using:

```
python -m unittest discover -s tests
```

Or using pytest:

```
pytest
```

## CI/CD Pipeline

This project uses GitHub Actions for continuous integration and deployment. The workflow is defined in `.github/workflows/ci-cd.yml` and includes the following stages:

### CI (Continuous Integration)

1. **Linting**: Checks code style and syntax using flake8
2. **Testing**: Runs all tests using pytest
3. **Coverage**: Generates and uploads code coverage reports

### CD (Continuous Deployment)

When changes are pushed to the main branch, the workflow:

1. **Deploys**: Automatically deploys the application to Heroku

### Setting Up GitHub Actions for Heroku Deployment

1. Push your code to GitHub
2. Create a Heroku account and create a new app
3. Configure the following secrets in your GitHub repository settings:
   - Go to Settings > Secrets and variables > Actions
   - Add the following secrets:
     - `HEROKU_API_KEY`: Your Heroku API key (found in your Heroku account settings)
     - `HEROKU_APP_NAME`: The name of your Heroku application
     - `HEROKU_EMAIL`: The email address associated with your Heroku account

## Heroku Deployment

You can also manually deploy the application to Heroku:

### Prerequisites

- Heroku CLI installed
- Heroku account

### Steps

1. Login to Heroku:
   ```
   heroku login
   ```

2. Create a new Heroku app (if you haven't already):
   ```
   heroku create your-app-name
   ```

3. Push to Heroku:
   ```
   git push heroku main
   ```

4. Scale the web dyno:
   ```
   heroku ps:scale web=1
   ```

5. Open the app:
   ```
   heroku open
   ```

## Troubleshooting

### Common Issues

- **403 Forbidden**: Check CORS settings in the application
- **Model not found**: Ensure the model is trained before making predictions
- **Data format errors**: Verify that the input JSON matches the expected format

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
