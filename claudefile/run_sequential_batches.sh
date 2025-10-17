#!/bin/bash
# Sequential batch runner for Aladyn training
# Runs batches one at a time to avoid memory conflicts

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Configuration
BATCH_SIZE=10000
TOTAL_SAMPLES=400000
NUM_EPOCHS=200

# Calculate number of batches
NUM_BATCHES=$((TOTAL_SAMPLES / BATCH_SIZE))

echo "========================================"
echo "Sequential Batch Training"
echo "========================================"
echo "Total samples: $TOTAL_SAMPLES"
echo "Batch size: $BATCH_SIZE"
echo "Number of batches: $NUM_BATCHES"
echo "Epochs per batch: $NUM_EPOCHS"
echo "========================================"
echo ""

# Run batches sequentially
for ((i=0; i<NUM_BATCHES; i++)); do
    START=$((i * BATCH_SIZE))
    END=$(((i + 1) * BATCH_SIZE))

    echo ""
    echo "----------------------------------------"
    echo "Batch $((i+1))/$NUM_BATCHES: Samples $START to $END"
    echo "----------------------------------------"

    python run_aladyn_batch.py \
        --start_index $START \
        --end_index $END \
        --num_epochs $NUM_EPOCHS \
        2>&1 | tee "logs/batch_${START}_${END}.log"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✓ Batch $((i+1))/$NUM_BATCHES completed successfully"
    else
        echo "✗ Batch $((i+1))/$NUM_BATCHES FAILED!"
        echo "Check log: logs/batch_${START}_${END}.log"
        exit 1
    fi

    # Brief pause to allow memory cleanup
    sleep 2
done

echo ""
echo "========================================"
echo "ALL BATCHES COMPLETE!"
echo "========================================"
