# Dockerfile for Aladynoulli Training Environment
# This creates a reproducible environment with all dependencies

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    vim \
    r-base \
    r-base-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker caching)
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for AWS
RUN pip install --no-cache-dir \
    boto3 \
    awscli \
    s3fs

# Copy the entire project
COPY pyScripts/ /app/pyScripts/
COPY claudefile/ /app/claudefile/

# Create directories for data and results
RUN mkdir -p /data /results /logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/data/torch_cache
ENV HDF5_USE_FILE_LOCKING=FALSE

# Default command (can be overridden)
CMD ["python", "pyScripts/local_survival_training.py"]
