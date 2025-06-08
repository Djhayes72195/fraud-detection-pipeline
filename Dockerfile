# Use official Python base image (lean slim version)
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements file first (Docker cache optimization)
COPY predict_service_requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r predict_service_requirements.txt

# Copy your app code — adjust as needed
COPY api/ ./api
COPY inference/ ./inference
COPY modeling/ ./modeling
COPY model_registry/ ./model_registry

# Expose FastAPI port
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
