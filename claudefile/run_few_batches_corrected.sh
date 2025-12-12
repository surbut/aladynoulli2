#!/bin/bash
# Run a few batches (not all 40) with corrected E matrix and prevalence
# For testing PC vs no PC comparison

BATCH_SIZE=10000
# Run just batches 0, 1, 2 (samples 0-30000) for testing
BATCHES_TO_RUN=(0 1 2)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/run_aladyn_batch_vector_e_censor_npcs.py"

# NO PCs version
OUTPUT_DIR_NO_PCS="/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_noPCS"
LOG_DIR_NO_PCS="$OUTPUT_DIR_NO_PCS/logs"

# WITH PCs version  
OUTPUT_DIR_WITH_PCS="/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_withPCS"
LOG_DIR_WITH_PCS="$OUTPUT_DIR_WITH_PCS/logs"

# Create output and log directories
mkdir -p "$OUTPUT_DIR_NO_PCS"
mkdir -p "$LOG_DIR_NO_PCS"
mkdir -p "$OUTPUT_DIR_WITH_PCS"
mkdir -p "$LOG_DIR_WITH_PCS"

echo "=========================================================="
echo "Running ${#BATCHES_TO_RUN[@]} batches for PC vs no PC comparison"
echo "Batches: ${BATCHES_TO_RUN[@]}"
echo "=========================================================="
echo ""

# Run WITHOUT PCs
echo "=========================================================="
echo "Running WITHOUT PCs..."
echo "=========================================================="
for i in "${BATCHES_TO_RUN[@]}"; do
    START=$((i * BATCH_SIZE))
    END=$((START + BATCH_SIZE))
    
    echo "Starting batch: samples $START to $END (NO PCs)"
    
    LOG_FILE="$LOG_DIR_NO_PCS/batch_${START}_${END}.log"
    
    python "$SCRIPT" \
        --start_index $START \
        --end_index $END \
        --output_dir "$OUTPUT_DIR_NO_PCS" \
        --no_pcs \
        > "$LOG_FILE" 2>&1
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Batch $START-$END (NO PCs) completed successfully"
    else
        echo "✗ Batch $START-$END (NO PCs) failed with exit code $EXIT_CODE"
        echo "  Check log: $LOG_FILE"
    fi
    echo ""
done

# Run WITH PCs
echo "=========================================================="
echo "Running WITH PCs..."
echo "=========================================================="
for i in "${BATCHES_TO_RUN[@]}"; do
    START=$((i * BATCH_SIZE))
    END=$((START + BATCH_SIZE))
    
    echo "Starting batch: samples $START to $END (WITH PCs)"
    
    LOG_FILE="$LOG_DIR_WITH_PCS/batch_${START}_${END}.log"
    
    python "$SCRIPT" \
        --start_index $START \
        --end_index $END \
        --output_dir "$OUTPUT_DIR_WITH_PCS" \
        --include_pcs \
        > "$LOG_FILE" 2>&1
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Batch $START-$END (WITH PCs) completed successfully"
    else
        echo "✗ Batch $START-$END (WITH PCs) failed with exit code $EXIT_CODE"
        echo "  Check log: $LOG_FILE"
    fi
    echo ""
done

echo "=========================================================="
echo "All batches complete!"
echo "NO PCs outputs: $OUTPUT_DIR_NO_PCS"
echo "WITH PCs outputs: $OUTPUT_DIR_WITH_PCS"
echo "=========================================================="

