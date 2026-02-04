import requests
import json

def test_prediction():
    url = "http://localhost:8000/predict"
    
    # Sample customer data
    payload = {
        "Age": 45,
        "Gender": "Female",
        "Location": "Miami",
        "Subscription_Length_Months": 15,
        "Monthly_Bill": 75.5,
        "Total_Usage_GB": 300.0
    }
    
    print(f"Sending request to {url}...")
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Successfully received prediction:")
            print(json.dumps(response.json(), indent=4))
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Could not connect to API: {e}")
        print("Make sure the FastAPI server is running with: uvicorn app.api:app --port 8000")

if __name__ == "__main__":
    test_prediction()
