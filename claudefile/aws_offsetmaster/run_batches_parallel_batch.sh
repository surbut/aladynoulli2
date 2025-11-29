#!/bin/bash
# Run multiple batches in parallel on AWS using run_aladyn_batch.py
# Usage: ./run_batches_parallel_batch.sh [num_parallel] [start_batch] [max_batches]
# Example: ./run_batches_parallel_batch.sh 2 0 40  # Run 2 at a time, batches 0-39

NUM_PARALLEL=${1:-2}  # Default: 2 batches in parallel (use 2 for c8i.24xlarge)
START_BATCH=${2:-0}   # Default: start from batch 0
MAX_BATCHES=${3:-40}  # Default: 40 batches total (batches 0-39)

# Default paths for aladyn_project structure
PROJECT_DIR="${PROJECT_DIR:-$HOME/aladyn_project}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data_for_running}"

mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Running batches in parallel"
echo "=========================================="
echo "Parallel batches: $NUM_PARALLEL"
echo "Start batch: $START_BATCH"
echo "Max batches: $MAX_BATCHES"
echo "Output dir: $OUTPUT_DIR"
echo "Log dir: $LOG_DIR"
echo "Data dir: $DATA_DIR"
echo "=========================================="

# Function to run a single batch
run_batch() {
    local batch_num=$1
    local start_idx=$((batch_num * 10000))
    local end_idx=$((start_idx + 10000))
    
    echo "[$(date)] Starting batch $batch_num (samples $start_idx-$end_idx)"
    
    # Change to project directory
    cd "$PROJECT_DIR" || return 1
    
    # Activate virtual environment if it exists
    if [ -f "$PROJECT_DIR/aladyn_env/bin/activate" ]; then
        source "$PROJECT_DIR/aladyn_env/bin/activate"
    elif [ -f ~/aladyn_env/bin/activate ]; then
        source ~/aladyn_env/bin/activate
    fi
    
    # Find the script location
    if [ -f "$PROJECT_DIR/scripts/run_aladyn_batch.py" ]; then
        SCRIPT_PATH="$PROJECT_DIR/scripts/run_aladyn_batch.py"
        cd "$PROJECT_DIR/scripts" || return 1
    elif [ -f "$PROJECT_DIR/run_aladyn_batch.py" ]; then
        SCRIPT_PATH="$PROJECT_DIR/run_aladyn_batch.py"
        cd "$PROJECT_DIR" || return 1
    else
        echo "[$(date)] ✗ ERROR: Could not find run_aladyn_batch.py"
        return 1
    fi
    
    # Run the batch script
    python "$SCRIPT_PATH" \
        --start_index $start_idx \
        --end_index $end_idx \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --covariates_path "$DATA_DIR/baselinagefamh_withpcs.csv" \
        > "$LOG_DIR/batch_${batch_num}.log" 2>&1
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
export OUTPUT_DIR LOG_DIR DATA_DIR HOME PROJECT_DIR

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
echo "Count running: ps aux | grep run_aladyn_batch | grep -v grep | wc -l"
echo "Check completed: ls -1 $OUTPUT_DIR/enrollment_model_W*_batch_*.pt 2>/dev/null | wc -l"
echo "=========================================="

