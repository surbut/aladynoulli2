#!/bin/bash
# Run all 40 batches for reparam (run_aladyn_batch_vector_e_censor_nolor_reparam)
# Same pattern as run_all_batches_nolr.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

BATCH_SIZE=10000
TOTAL_SAMPLES=400000
NUM_BATCHES=$((TOTAL_SAMPLES / BATCH_SIZE))

mkdir -p logs

echo "========================================"
echo "Running $NUM_BATCHES batches (REPARAM)"
echo "Output: censor_e_batchrun_vectorized_REPARAM"
echo "========================================"

for ((i=0; i<NUM_BATCHES; i++)); do
    START=$((i * BATCH_SIZE))
    END=$(((i + 1) * BATCH_SIZE))

    echo ""
    echo "Batch $((i+1))/$NUM_BATCHES: Samples $START to $END"
    echo "----------------------------------------"

    python run_aladyn_batch_vector_e_censor_nolor_reparam.py \
        --start_index $START \
        --end_index $END \
        --num_epochs 200 \
        --learning_rate 0.1 \
        --K 20 \
        --W 0.0001 \
        2>&1 | tee "logs/batch_${START}_${END}_reparam.log"

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
echo "ALL REPARAM BATCHES COMPLETE!"
echo "Output: censor_e_batchrun_vectorized_REPARAM"
echo "Next: pool_phi_kappa_gamma_from_batches.py --model_type reparam --max_batches 39"
echo "========================================"
