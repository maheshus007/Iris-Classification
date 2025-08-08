# Use Python 3.12 slim as base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies for MLflow, DVC, etc.
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    gcc \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY ../ .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# DVC init without SCM (no git tracking inside Docker)
RUN dvc init --no-scm || echo "DVC already initialized"

# Create necessary directories
RUN mkdir -p /app/data /app/model /app/logs

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
