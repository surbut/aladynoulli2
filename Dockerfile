FROM python:3.9-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (pin in requirements.txt as you like)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
# Ensure these files exist in your repo root OR adjust as needed.
COPY pyScripts/ /app/pyScripts/

# Data mount points (EFS)
RUN mkdir -p /data/input /data/output

# Env
ENV PYTHONPATH=/app:/app/pyScripts \
    MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Leave entrypoint empty so Batch job definition controls the command
ENTRYPOINT []
CMD []
