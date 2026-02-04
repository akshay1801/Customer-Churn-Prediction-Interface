# Use official Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Run the pipeline once to ensure artifacts exist (optional if images are pre-trained)
# RUN python main.py

# Expose ports
EXPOSE 8501
EXPOSE 8000

# Create a startup script
RUN echo '#!/bin/bash\n\
uvicorn app.api:app --host 0.0.0.0 --port 8000 & \n\
streamlit run app/app.py --server.port=8501 --server.address=0.0.0.0\n\
' > /app/start.sh && chmod +x /app/start.sh

# Command to run both
CMD ["/app/start.sh"]
