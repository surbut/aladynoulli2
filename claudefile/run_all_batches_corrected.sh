#!/bin/bash
# Run all 40 batches (0-400k) with corrected E matrix and prevalence
# Outputs to censor_e_batchrun_vectorized/

BATCH_SIZE=10000
TOTAL_SAMPLES=400000
NUM_BATCHES=$((TOTAL_SAMPLES / BATCH_SIZE))

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/run_aladyn_batch_vector_e_censor.py"

OUTPUT_DIR="/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized"
LOG_DIR="$OUTPUT_DIR/logs"

# Create output and log directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "=========================================================="
echo "Running all $NUM_BATCHES batches (0 to $TOTAL_SAMPLES)"
echo "Output directory: $OUTPUT_DIR"
echo "Log directory: $LOG_DIR"
echo "=========================================================="
echo ""

# Run each batch
for i in $(seq 0 $((NUM_BATCHES - 1))); do
    START=$((i * BATCH_SIZE))
    END=$((START + BATCH_SIZE))
    
    echo "=========================================================="
    echo "Starting batch $((i+1))/$NUM_BATCHES: samples $START to $END"
    echo "=========================================================="
    
    LOG_FILE="$LOG_DIR/batch_${START}_${END}.log"
    
    python "$SCRIPT" \
        --start_index $START \
        --end_index $END \
        --output_dir "$OUTPUT_DIR" \
        > "$LOG_FILE" 2>&1
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Batch $((i+1))/$NUM_BATCHES completed successfully"
    else
        echo "✗ Batch $((i+1))/$NUM_BATCHES failed with exit code $EXIT_CODE"
        echo "  Check log: $LOG_FILE"
        # Continue with next batch instead of stopping
    fi
    
    echo ""
done

echo "=========================================================="
echo "All batches complete!"
echo "Check logs in: $LOG_DIR"
echo "Check outputs in: $OUTPUT_DIR"
echo "=========================================================="

