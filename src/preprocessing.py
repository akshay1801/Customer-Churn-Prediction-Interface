import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

def preprocess_data(file_path, output_dir='data'):
    """
    Cleans and preprocesses the customer churn dataset.
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_excel(file_path)
    
    # Drop irrelevant columns
    cols_to_drop = ['CustomerID', 'Name']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # Handle categorical variables
    print("Encoding categorical variables...")
    le_gender = LabelEncoder()
    df['Gender'] = le_gender.fit_transform(df['Gender'])
    
    # One-hot encode Location
    df = pd.get_dummies(df, columns=['Location'], prefix='Loc')
    
    # Feature Scaling
    print("Scaling numerical features...")
    scaler = StandardScaler()
    numerical_cols = ['Age', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Save artifacts
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(le_gender, 'models/gender_encoder.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    # Save column names for one-hot encoding consistency
    joblib.dump(df.columns.tolist(), 'models/column_names.pkl')
    
    # Split data
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Save processed data
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
    
    print(f"Preprocessed data saved in {output_dir}/")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocess_data('data/customer_churn_large_dataset.xlsx')
