#!/bin/bash
# Run all 40 batches sequentially for joint estimation (NO lambda_reg on gamma)

START_BATCH=0
MAX_BATCHES=40
OUTPUT_DIR=~/aladyn_project/results
LOG_DIR=~/aladyn_project/logs

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "=========================================="
echo "Running joint estimation batches (NOLR)"
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
    
    python scripts/run_aladyn_batch_vector_e_enrollment_joint_nolr.py \
        --start_index $start_idx \
        --end_index $end_idx \
        --data_dir ~/aladyn_project/data_for_running \
        --output_dir "$OUTPUT_DIR" \
        --covariates_path ~/aladyn_project/data_for_running/baselinagefamh_withpcs.csv \
        > "$LOG_DIR/batch_${batch_num}.log" 2>&1
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[$(date)] ✓ Batch $batch_num completed successfully"
    else
        echo "[$(date)] ✗ Batch $batch_num failed with exit code $exit_code"
    fi
    return $exit_code
}

# Run batches sequentially
for ((i=$START_BATCH; i<$((START_BATCH + MAX_BATCHES)); i++)); do
    run_batch $i
done

echo ""
echo "=========================================="
echo "All batches complete!"
echo "=========================================="
echo "Check logs:    tail -f $LOG_DIR/batch_*.log"
echo "Check output: ls -1 $OUTPUT_DIR/enrollment_joint_model_VECTORIZED_nolr_*.pt | wc -l"
echo "=========================================="
