#!/bin/bash
# Script to run multiple training jobs in parallel on different data chunks
# This is useful for running on large EC2 instances with multiple GPUs/cores

set -e

# Configuration
TOTAL_SAMPLES=400000
CHUNK_SIZE=10000
MAX_AGE_OFFSET=30
MAX_PARALLEL_JOBS=4  # Adjust based on your instance resources

# Data directory
DATA_DIR=${DATA_DIR:-"/home/ubuntu/aladynoulli2/data"}
RESULTS_DIR=${RESULTS_DIR:-"$(pwd)/results"}
LOGS_DIR=${LOGS_DIR:-"$(pwd)/logs"}

# Create directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$LOGS_DIR"

echo "========================================="
echo "Parallel Training Job Launcher"
echo "========================================="
echo "Total Samples: $TOTAL_SAMPLES"
echo "Chunk Size: $CHUNK_SIZE"
echo "Max Parallel Jobs: $MAX_PARALLEL_JOBS"
echo "========================================="
echo ""

# Calculate number of chunks
NUM_CHUNKS=$((TOTAL_SAMPLES / CHUNK_SIZE))

# Function to run a single chunk
run_chunk() {
    local START=$1
    local END=$2
    local GPU_ID=$3

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$LOGS_DIR/training_${START}_${END}_${TIMESTAMP}.log"

    echo "Starting chunk [$START-$END] on GPU $GPU_ID (log: $LOG_FILE)"

    # Set GPU device if available
    GPU_FLAG=""
    if [ -n "$GPU_ID" ] && command -v nvidia-smi &> /dev/null; then
        GPU_FLAG="--gpus device=$GPU_ID"
    fi

    docker run \
        $GPU_FLAG \
        --rm \
        -v "$DATA_DIR:/data" \
        -v "$RESULTS_DIR:/results" \
        -v "$LOGS_DIR:/logs" \
        -e START_INDEX=$START \
        -e END_INDEX=$END \
        -e MAX_AGE_OFFSET=$MAX_AGE_OFFSET \
        -e PYTHONUNBUFFERED=1 \
        --name "aladyn_${START}_${END}" \
        aladynoulli-training \
        python pyScripts/local_survival_training.py \
        > "$LOG_FILE" 2>&1

    echo "Completed chunk [$START-$END]"
}

# Export function for parallel execution
export -f run_chunk
export DATA_DIR RESULTS_DIR LOGS_DIR MAX_AGE_OFFSET

# Create array of chunk indices
CHUNKS=()
for i in $(seq 0 $((NUM_CHUNKS - 1))); do
    START=$((i * CHUNK_SIZE))
    END=$(((i + 1) * CHUNK_SIZE))
    if [ $END -gt $TOTAL_SAMPLES ]; then
        END=$TOTAL_SAMPLES
    fi
    CHUNKS+=("$START $END")
done

echo "Total chunks to process: ${#CHUNKS[@]}"
echo ""

# Process chunks in parallel
echo "Starting parallel processing..."
echo ""

# Simple parallel execution using background jobs
ACTIVE_JOBS=0
GPU_ID=0

for chunk in "${CHUNKS[@]}"; do
    read START END <<< "$chunk"

    # Wait if we've reached max parallel jobs
    while [ $ACTIVE_JOBS -ge $MAX_PARALLEL_JOBS ]; do
        sleep 10
        # Count running docker containers
        ACTIVE_JOBS=$(docker ps | grep -c aladyn_ || true)
    done

    # Launch job in background
    run_chunk $START $END $GPU_ID &

    # Increment counters
    ACTIVE_JOBS=$((ACTIVE_JOBS + 1))
    GPU_ID=$(((GPU_ID + 1) % MAX_PARALLEL_JOBS))

    sleep 2  # Small delay between launches
done

# Wait for all background jobs to complete
echo ""
echo "Waiting for all jobs to complete..."
wait

echo ""
echo "========================================="
echo "All parallel jobs completed!"
echo "========================================="
echo "Results saved to: $RESULTS_DIR"
echo "Logs saved to: $LOGS_DIR"
