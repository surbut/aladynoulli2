#!/bin/bash
# Run multiple batches in parallel on AWS
# Usage: ./run_batches_parallel.sh [num_parallel] [start_batch] [max_batches]
# Example: ./run_batches_parallel.sh 8 10 30  # Run 8 at a time, starting from batch 10, 30 batches total

NUM_PARALLEL=${1:-8}  # Default: 8 batches in parallel (adjust based on instance)
START_BATCH=${2:-10}  # Default: start from batch 10 (since 0-9 are done)
MAX_BATCHES=${3:-30}  # Default: 30 batches remaining (batches 10-39)

# Change to project directory
cd ~/aladyn_project || exit 1

OUTPUT_DIR="${OUTPUT_DIR:-./output/retrospective_pooled}"
LOG_DIR="${LOG_DIR:-./logs}"

mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Running batches in parallel"
echo "=========================================="
echo "Parallel batches: $NUM_PARALLEL"
echo "Start batch: $START_BATCH"
echo "Max batches: $MAX_BATCHES"
echo "Output dir: $OUTPUT_DIR"
echo "Log dir: $LOG_DIR"
echo "=========================================="

# Function to run a single batch
run_batch() {
    local batch_num=$1
    local start_idx=$((batch_num * 10000))
    local end_idx=$((start_idx + 10000))
    
    echo "[$(date)] Starting batch $batch_num (samples $start_idx-$end_idx)"
    
    cd ~/aladyn_project || return 1
    
    # Activate virtual environment if it exists
    if [ -f ~/aladyn_project/aladyn_env/bin/activate ]; then
        source ~/aladyn_project/aladyn_env/bin/activate
    fi
    
    # Script is in scripts/, so use scripts/run_aladyn_predict_with_master.py
    python scripts/run_aladyn_predict_with_master.py \
        --trained_model_path data_for_running/master_for_fitting_pooled_all_data.pt \
        --data_dir data_for_running/ \
        --output_dir "$OUTPUT_DIR" \
        --covariates_path data_for_running/baselinagefamh_withpcs.csv \
        --start_batch $batch_num \
        --max_batches 1 \
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
export OUTPUT_DIR LOG_DIR

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
echo "All batches submitted!"
echo "=========================================="
echo "Check logs: tail -f $LOG_DIR/batch_*.log"
echo "Count running: ps aux | grep run_aladyn_predict_with_master | grep -v grep | wc -l"
echo "Check completed: ls -1 $OUTPUT_DIR/model_enroll_fixedphi_sex_*.pt | wc -l"
echo "=========================================="

