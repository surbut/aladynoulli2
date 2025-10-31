#!/bin/bash
# Run full retrospective characterization on all 400k samples WITHOUT PCs (sex only)
# Processes in 10k batches sequentially
# nohup ./run_full_retrospective_noPCS.sh > retrospective_no_pcs.log 2>&1 &
set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Configuration
BATCH_SIZE=10000
TOTAL_SAMPLES=400000
NUM_EPOCHS=200
INCLUDE_PCS=False

# Calculate number of batches
NUM_BATCHES=$((TOTAL_SAMPLES / BATCH_SIZE))

# Create logs directory if it doesn't exist
mkdir -p logs

echo "========================================"
echo "Full Retrospective Characterization"
echo "========================================"
echo "Total samples: $TOTAL_SAMPLES"
echo "Batch size: $BATCH_SIZE"
echo "Number of batches: $NUM_BATCHES"
echo "Epochs per batch: $NUM_EPOCHS"
echo "Include PCs: $INCLUDE_PCS"
echo "========================================"
echo ""

# Track start time
START_TIME=$(date +%s)

# Run batches sequentially
for ((i=0; i<NUM_BATCHES; i++)); do
    START=$((i * BATCH_SIZE))
    END=$(((i + 1) * BATCH_SIZE))

    echo ""
    echo "----------------------------------------"
    echo "Batch $((i+1))/$NUM_BATCHES: Samples $START to $END"
    echo "Started at: $(date)"
    echo "----------------------------------------"

    python run_aladyn_batch_sex_nopc.py \
        --start_index $START \
        --end_index $END \
        --num_epochs $NUM_EPOCHS \
        --output_dir /Users/sarahurbut/Library/CloudStorage/Dropbox/ret_full_nopc_withsex \
        2>&1 | tee "logs/retrospective_batch_${START}_${END}.log"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✓ Batch $((i+1))/$NUM_BATCHES completed successfully at $(date)"
    else
        echo "✗ Batch $((i+1))/$NUM_BATCHES FAILED!"
        echo "Check log: logs/retrospective_batch_${START}_${END}.log"
        exit 1
    fi

    # Brief pause to allow memory cleanup
    sleep 2
done

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo ""
echo "========================================"
echo "ALL BATCHES COMPLETE!"
echo "========================================"
echo "Total time: ${HOURS}h ${MINUTES}m"
echo "Finished at: $(date)"
echo "========================================"
