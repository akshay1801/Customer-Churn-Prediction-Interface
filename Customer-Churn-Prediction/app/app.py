import streamlit as st
import pandas as pd
import joblib
import os

# Page Config
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.title("📊 Customer Churn Prediction Interface")
st.markdown("""
This app predicts whether a customer is likely to churn based on the latest production-grade model.
""")

# Load artifacts
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('models/production_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        gender_encoder = joblib.load('models/gender_encoder.pkl')
        cols = joblib.load('models/column_names.pkl')
        return model, scaler, gender_encoder, cols
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None, None, None

model, scaler, gender_encoder, cols = load_artifacts()

if model:
    st.sidebar.header("Customer Input Features")
    
    age = st.sidebar.slider("Age", 18, 70, 35)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    location = st.sidebar.selectbox("Location", ["Chicago", "Houston", "Los Angeles", "Miami", "New York"])
    sub_length = st.sidebar.slider("Subscription Length (Months)", 1, 24, 12)
    bill = st.sidebar.number_input("Monthly Bill ($)", 30.0, 100.0, 65.0)
    usage = st.sidebar.number_input("Total Usage (GB)", 50.0, 500.0, 250.0)
    
    if st.button("Predict Churn"):
        # Prepare input data
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Location': [location],
            'Subscription_Length_Months': [sub_length],
            'Monthly_Bill': [bill],
            'Total_Usage_GB': [usage]
        })
        
        # Preprocessing (must match training logic)
        input_data['Gender'] = gender_encoder.transform(input_data['Gender'])
        input_data = pd.get_dummies(input_data, columns=['Location'], prefix='Loc')
        
        # Ensure all columns from training are present
        # Remove 'Churn' from expected columns if it exists
        feature_cols = [c for c in cols if c != 'Churn']
        for col in feature_cols:
            if col not in input_data.columns:
                input_data[col] = 0
        
        input_data = input_data[feature_cols]
        
        # Scaling
        num_cols = ['Age', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB']
        input_data[num_cols] = scaler.transform(input_data[num_cols])
        
        # Prediction
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]
        
        # Display Result
        st.subheader("Result")
        if prediction == 1:
            st.error(f"Prediction: **Churn** (Probability: {prob:.2%})")
        else:
            st.success(f"Prediction: **No Churn** (Probability: {prob:.2%})")
            
else:
    st.warning("Production model not found. Please run the pipeline (main.py) first.")
