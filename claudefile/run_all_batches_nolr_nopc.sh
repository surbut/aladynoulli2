#!/bin/bash
# Run all 40 batches sequentially (no lambda_reg, no PCs, no sex version)
# This version uses PRS only (36 PRS), similar to March version but with corrected E matrix

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

BATCH_SIZE=10000
TOTAL_SAMPLES=400000
NUM_BATCHES=$((TOTAL_SAMPLES / BATCH_SIZE))

mkdir -p logs

echo "========================================"
echo "Running $NUM_BATCHES batches (NO LAMBDA_REG, NO PCs, NO SEX)"
echo "Configuration: PRS only (36 PRS), no lambda_reg penalty"
echo "This should be close to March version (but uses corrected E matrix)"
echo "========================================"

for ((i=0; i<NUM_BATCHES; i++)); do
    START=$((i * BATCH_SIZE))
    END=$(((i + 1) * BATCH_SIZE))

    echo ""
    echo "Batch $((i+1))/$NUM_BATCHES: Samples $START to $END"
    echo "----------------------------------------"

    python run_aladyn_batch_vector_e_censor_nolor_nosex_nopc.py \
        --start_index $START \
        --end_index $END \
        --num_epochs 200 \
        --learning_rate 0.1 \
        --K 20 \
        --W 0.0001 \
        2>&1 | tee "logs/batch_${START}_${END}_nolr_nopcs_nosex.log"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✓ Batch $((i+1)) completed"
    else
        echo "✗ Batch $((i+1)) FAILED!"
        exit 1
    fi
    
    sleep 2
done

echo ""
echo "========================================"
echo "ALL BATCHES COMPLETE!"
echo "Output directory: censor_e_batchrun_vectorized_nolr_nopcs_nosex"
echo "========================================"