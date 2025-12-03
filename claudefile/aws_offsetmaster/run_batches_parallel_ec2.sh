#!/bin/bash
# Run multiple batches in parallel on EC2 using setup_and_run_ec2.sh
# Usage: ./run_batches_parallel_ec2.sh [num_parallel] [start_batch] [max_batches]
# Example: ./run_batches_parallel_ec2.sh 2 0 40  # Run 2 at a time, batches 0-39

NUM_PARALLEL=${1:-2}  # Default: 2 batches in parallel
START_BATCH=${2:-0}   # Default: start from batch 0
MAX_BATCHES=${3:-40}  # Default: 40 batches total (batches 0-39)

LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Running batches in parallel"
echo "=========================================="
echo "Parallel batches: $NUM_PARALLEL"
echo "Start batch: $START_BATCH"
echo "Max batches: $MAX_BATCHES"
echo "Log dir: $LOG_DIR"
echo "=========================================="

# Function to run a single batch
run_batch() {
    local batch_num=$1
    local start_idx=$((batch_num * 10000))
    local end_idx=$((start_idx + 10000))
    
    echo "[$(date)] Starting batch $batch_num (samples $start_idx-$end_idx)"
    
    bash setup_and_run_ec2.sh $start_idx $end_idx > "$LOG_DIR/batch_${batch_num}.log" 2>&1
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[$(date)] ✓ Batch $batch_num completed successfully"
    else
        echo "[$(date)] ✗ Batch $batch_num failed with exit code $exit_code"
    fi
    return $exit_code
}

# Export function and variables so they can be used by parallel processes
export -f run_batch
export LOG_DIR

# Create array of batch numbers
batches=()
for ((i=$START_BATCH; i<$((START_BATCH + MAX_BATCHES)); i++)); do
    batches+=($i)
done

echo "Total batches to process: ${#batches[@]}"
echo "Running $NUM_PARALLEL batches in parallel..."
echo ""

# Run batches in parallel using xargs
# Use -P for parallelism, -I {} replaces {} with the batch number (one per line)
printf '%s\n' "${batches[@]}" | xargs -P $NUM_PARALLEL -I {} bash -c 'run_batch {}'

echo ""
echo "=========================================="
echo "All batches completed!"
echo "=========================================="
echo "Check logs: tail -f $LOG_DIR/batch_*.log"
echo "Count running: ps aux | grep setup_and_run_ec2 | grep -v grep | wc -l"
echo "Check completed: ls -1 /results/model_batch_*.pt 2>/dev/null | wc -l"
echo "=========================================="






