FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY conda.yaml . 

# Install Python dependencies
RUN pip install --no-cache-dir \
    mlflow>=2.10.0 \
    pandas>=2.0.0 \
    numpy>=1. 24.0 \
    scikit-learn>=1.3. 0 \
    matplotlib>=3.7.0 \
    seaborn>=0. 12.0

# Copy project files
COPY .  /app

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "modelling.py"]