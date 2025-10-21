#!/bin/bash
# Script to run training jobs with Docker
# Usage: ./run-training.sh [start_index] [end_index] [max_age_offset]

set -e

# Default values
START_INDEX=${1:-0}
END_INDEX=${2:-10000}
MAX_AGE_OFFSET=${3:-30}

# Data directory (adjust as needed)
DATA_DIR=${DATA_DIR:-"/home/ubuntu/aladynoulli2/data"}
RESULTS_DIR=${RESULTS_DIR:-"$(pwd)/results"}
LOGS_DIR=${LOGS_DIR:-"$(pwd)/logs"}

# Create directories if they don't exist
mkdir -p "$RESULTS_DIR"
mkdir -p "$LOGS_DIR"

echo "========================================="
echo "Aladynoulli Training Job"
echo "========================================="
echo "Start Index: $START_INDEX"
echo "End Index: $END_INDEX"
echo "Max Age Offset: $MAX_AGE_OFFSET"
echo "Data Directory: $DATA_DIR"
echo "Results Directory: $RESULTS_DIR"
echo "Logs Directory: $LOGS_DIR"
echo "========================================="
echo ""

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory does not exist: $DATA_DIR"
    echo "Please create it and copy your data files there."
    exit 1
fi

# Check for GPU
GPU_FLAG=""
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected, enabling GPU support..."
    GPU_FLAG="--gpus all"
else
    echo "No GPU detected, running on CPU..."
fi

# Generate timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOGS_DIR/training_${START_INDEX}_${END_INDEX}_${TIMESTAMP}.log"

echo "Starting training... (logs: $LOG_FILE)"
echo ""

# Run Docker container
docker run \
    $GPU_FLAG \
    --rm \
    -v "$DATA_DIR:/data" \
    -v "$RESULTS_DIR:/results" \
    -v "$LOGS_DIR:/logs" \
    -e START_INDEX=$START_INDEX \
    -e END_INDEX=$END_INDEX \
    -e MAX_AGE_OFFSET=$MAX_AGE_OFFSET \
    -e PYTHONUNBUFFERED=1 \
    aladynoulli-training \
    python pyScripts/local_survival_training.py \
    2>&1 | tee "$LOG_FILE"

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Training completed successfully!"
    echo "========================================="
    echo "Results saved to: $RESULTS_DIR"
    echo "Log saved to: $LOG_FILE"
else
    echo ""
    echo "========================================="
    echo "Training failed!"
    echo "========================================="
    echo "Check log file: $LOG_FILE"
    exit 1
fi

# Optional: Copy results to S3
read -p "Upload results to S3? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    read -p "Enter S3 bucket path (e.g., s3://my-bucket/results/): " S3_PATH
    echo "Uploading to $S3_PATH..."
    aws s3 sync "$RESULTS_DIR" "$S3_PATH"
    echo "Upload complete!"
fi
