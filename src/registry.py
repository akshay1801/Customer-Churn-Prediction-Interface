import joblib
import json
import os
import shutil
from datetime import datetime

REGISTRY_PATH = 'models/registry.json'

def get_registry():
    if not os.path.exists(REGISTRY_PATH):
        return {"production_model": None, "history": []}
    with open(REGISTRY_PATH, 'r') as f:
        return json.load(f)

def save_registry(registry):
    with open(REGISTRY_PATH, 'w') as f:
        json.dump(registry, f, indent=4)

def register_model(model_path, metrics, model_type):
    """
    Registers a new model and checks if it should be productionized.
    """
    registry = get_registry()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_filename = f"{model_type}_model_{timestamp}.pkl"
    versioned_path = f"models/{versioned_filename}"
    
    # Copy model to versioned path
    shutil.copy(model_path, versioned_path)
    
    new_entry = {
        "version": timestamp,
        "type": model_type,
        "path": versioned_path,
        "metrics": metrics
    }
    
    registry["history"].append(new_entry)
    
    # Logic Gate: Compare against production baseline
    production_model = registry.get("production_model")
    is_better = False
    
    if production_model is None:
        print("No production model found. Setting new model as production baseline.")
        registry["production_model"] = new_entry
        is_better = True
    else:
        current_f1 = production_model["metrics"]["f1"]
        new_f1 = metrics["f1"]
        print(f"Comparing New F1 ({new_f1:.4f}) with Production F1 ({current_f1:.4f})...")
        
        if new_f1 >= current_f1:
            print("New model meets or exceeds performance threshold. Updating production model.")
            registry["production_model"] = new_entry
            is_better = True
        else:
            print("New model does not exceed production performance. Model registered but not productionized.")
            
    save_registry(registry)
    
    if is_better:
        # Link current production model for app use
        shutil.copy(versioned_path, "models/production_model.pkl")
        
    return is_better

if __name__ == "__main__":
    # Test stub
    pass
