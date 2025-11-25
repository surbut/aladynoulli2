#!/bin/bash
# Run multiple batches in parallel by launching separate processes
# Usage: ./run_parallel_batches.sh [num_parallel]

NUM_PARALLEL=${1:-4}  # Default: 4 batches in parallel
START_BATCH=6         # Start from batch 6 (0-5 are done)
TOTAL_BATCHES=34      # 34 batches remaining

OUTPUT_DIR="./output/retrospective_pooled"
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Running $NUM_PARALLEL batches in parallel"
echo "Starting from batch $START_BATCH"
echo "=========================================="

# Launch parallel processes
for ((i=0; i<$NUM_PARALLEL; i++)); do
    batch_num=$((START_BATCH + i))
    
    echo "Launching batch $batch_num in background..."
    
    nohup python scripts/run_aladyn_predict_with_master.py \
        --trained_model_path data_for_running/master_for_fitting_pooled_all_data.pt \
        --data_dir data_for_running/ \
        --output_dir "$OUTPUT_DIR" \
        --covariates_path data_for_running/baselinagefamh_withpcs.csv \
        --start_batch $batch_num \
        --max_batches 1 \
        > "$LOG_DIR/batch_${batch_num}.log" 2>&1 &
    
    echo "  → PID: $!"
    sleep 2  # Small delay between launches
done

echo ""
echo "Launched $NUM_PARALLEL batches in parallel"
echo "Monitor with: tail -f $LOG_DIR/batch_*.log"
echo "Check processes: ps aux | grep run_aladyn_predict_with_master"
echo ""
echo "To launch more batches as these complete, run:"
echo "  ./run_parallel_batches.sh $NUM_PARALLEL"


