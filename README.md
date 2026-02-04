# Customer Churn Prediction: Production-Grade MLOps Pipeline

This project implements a modular MLOps lifecycle to automate the training, evaluation, and deployment of a customer churn prediction model. By transitioning from a static notebook to a structured pipeline, we ensure consistent preprocessing, artifact versioning, and performance-based model promotion. The system utilizes a mandatory logic gate that compares new model metrics (F1-score) against the current production baseline, only updating the registry if an improvement or equality is found. This architecture ensures high-reliability deployments and protects production environments from regressive updates.

### 🚀 Execution Instructions
1.  **Environment Setup**: Install all dependencies using `pip install -r requirements.txt`.
2.  **Pipeline Orchestration**: Run `python main.py` to trigger the end-to-end flow, including data cleaning, model training (RF/XGBoost), and registration.
3.  **User Interface**: Launch the interactive testing dashboard with `streamlit run app/app.py`.
4.  **Inference API**: Access the programmatic endpoint by running `uvicorn app.api:app --port 8000`.
5.  **Docker Deployment**: Use `docker build -t churn-pipeline .` and `docker run -p 8501:8501 -p 8000:8000 churn-pipeline` to run both services in a containerized environment.

### 📝 Assumptions and Limitations
*   **Data Integrity**: Continuous execution assumes the input `data/customer_churn_large_dataset.xlsx` remains consistent with the established schema (8 core features).
*   **Metric Logic**: The current logic gate is strictly F1-score dependent; while effective for balanced classes, it may need adjustment for highly skewed churn distributions.
*   **Storage**: The model registry is implemented as a local JSON-based system, which is ideal for single-node deployments but would require transition to a cloud-based registry (e.g., MLflow) for distributed scaling.
