from src.preprocessing import preprocess_data
from src.model_training import train_model
from src.registry import register_model
import os

def run_pipeline(data_path, model_type='rf'):
    """
    Runs the end-to-end MLOps pipeline.
    """
    print("--- Starting MLOps Pipeline ---")
    
    # 1. Preprocessing
    X_train, X_test, y_train, y_test = preprocess_data(data_path)
    
    # 2. Training
    model, metrics, model_path = train_model(
        'data/X_train.csv', 
        'data/y_train.csv', 
        'data/X_test.csv', 
        'data/y_test.csv', 
        model_type=model_type
    )
    
    # 3. Registration & Logic Gate
    is_deployed = register_model(model_path, metrics, model_type)
    
    if is_deployed:
        print("--- Pipeline Completed: New model deployed to production! ---")
    else:
        print("--- Pipeline Completed: New model registered but not deployed. ---")

if __name__ == "__main__":
    DATA_PATH = 'data/customer_churn_large_dataset.xlsx'
    if os.path.exists(DATA_PATH):
        # Run with RF first
        run_pipeline(DATA_PATH, model_type='rf')
        # Then try to improve with XGB
        run_pipeline(DATA_PATH, model_type='xgb')
    else:
        print(f"Dataset not found at {DATA_PATH}. Please check the path.")
