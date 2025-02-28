import sys
import os
import unittest
import json
import pandas as pd
import numpy as np

# Add parent directory to path so we can import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, clean_data, create_preprocessing_pipeline

class TestApp(unittest.TestCase):
    """Test cases for the car price prediction app"""
    
    def setUp(self):
        """Set up test client"""
        self.app = app.test_client()
        self.app.testing = True
    
    def test_home_route(self):
        """Test the home route"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Car Price Prediction API', response.data)
    
    def test_predict_route(self):
        """Test the prediction route"""
        test_data = {
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
        
        response = self.app.post('/predict',
                                data=json.dumps(test_data),
                                content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('prediction', data)
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'success')
        self.assertIsInstance(data['prediction'], float)
    
    def test_clean_data(self):
        """Test the data cleaning function"""
        test_df = pd.DataFrame({
            'Prod. year': ['2015', '2010', np.nan],
            'Engine volume': ['2.5', '3.0 Turbo', '1.8'],
            'Mileage': ['50000 km', '100000 km', np.nan],
            'Cylinders': ['4', '6.0', np.nan],
            'Airbags': ['8', '12', np.nan],
            'Levy': ['1000', '-', '500'],
            'Doors': ['04-May', '02-Mar', '>5']
        })
        
        cleaned_df = clean_data(test_df)
        
        # Check data types
        self.assertEqual(cleaned_df['Prod. year'].dtype, np.float64)
        self.assertEqual(cleaned_df['Engine volume'].dtype, np.float64)
        self.assertEqual(cleaned_df['Mileage'].dtype, np.float64)
        self.assertEqual(cleaned_df['Cylinders'].dtype, np.float64)
        self.assertEqual(cleaned_df['Airbags'].dtype, np.float64)
        self.assertEqual(cleaned_df['Levy'].dtype, np.float64)
        
        # Check specific values
        self.assertEqual(cleaned_df['Prod. year'].iloc[0], 2015.0)
        self.assertEqual(cleaned_df['Engine volume'].iloc[1], 3.0)
        self.assertEqual(cleaned_df['Mileage'].iloc[0], 50000.0)
    
    def test_preprocessing_pipeline(self):
        """Test the preprocessing pipeline creation"""
        preprocessor = create_preprocessing_pipeline()
        self.assertIsNotNone(preprocessor)
        
        # Check that the pipeline has the expected transformers
        transformers = preprocessor.transformers
        transformer_names = [name for name, _, _ in transformers]
        self.assertIn('num', transformer_names)
        self.assertIn('cat', transformer_names)

if __name__ == '__main__':
    unittest.main()
