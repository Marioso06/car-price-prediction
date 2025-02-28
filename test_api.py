import requests
import json

def test_predict_api():
    url = "http://localhost:8080/predict"
    data = {
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
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, data=json.dumps(data), headers=headers)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        return response
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    test_predict_api()
