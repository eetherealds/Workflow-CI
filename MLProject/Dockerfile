FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies with fixed versions
RUN pip install --no-cache-dir \
    mlflow==2.10. 0 \
    pandas==2.0.3 \
    numpy==1.24.4 \
    scikit-learn==1.3.2 \
    matplotlib==3.7.3 \
    seaborn==0.12.2

# Copy project files
COPY .  /app

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "modelling.py"]