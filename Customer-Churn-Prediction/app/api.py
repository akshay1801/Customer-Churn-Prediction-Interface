from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="Customer Churn Prediction API")

# Input model
class CustomerData(BaseModel):
    Age: int
    Gender: str
    Location: str
    Subscription_Length_Months: int
    Monthly_Bill: float
    Total_Usage_GB: float

# Load artifacts
def load_artifacts():
    try:
        model = joblib.load('models/production_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        gender_encoder = joblib.load('models/gender_encoder.pkl')
        cols = joblib.load('models/column_names.pkl')
        return model, scaler, gender_encoder, cols
    except Exception as e:
        return None, None, None, None

model, scaler, gender_encoder, cols = load_artifacts()

@app.get("/")
def read_root():
    return {"message": "Customer Churn Prediction API is running. Use /predict for inference."}

@app.post("/predict")
def predict_churn(data: CustomerData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Run pipeline first.")
    
    # Prepare input
    df = pd.DataFrame([data.dict()])
    
    try:
        # Preprocessing
        df['Gender'] = gender_encoder.transform(df['Gender'])
        df = pd.get_dummies(df, columns=['Location'], prefix='Loc')
        
        # Ensure all columns from training are present
        feature_cols = [c for c in cols if c != 'Churn']
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        df = df[feature_cols]
        
        # Scaling
        num_cols = ['Age', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB']
        df[num_cols] = scaler.transform(df[num_cols])
        
        # Prediction
        prediction = int(model.predict(df)[0])
        probability = float(model.predict_proba(df)[0][1])
        
        return {
            "prediction": "Churn" if prediction == 1 else "No Churn",
            "probability": probability,
            "status_code": 200
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
