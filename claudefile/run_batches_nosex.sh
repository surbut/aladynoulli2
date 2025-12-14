#!/bin/bash
# Run 8 batches sequentially (one at a time) without sex

BATCH_SIZE=10000
SCRIPT="claudefile/run_aladyn_batch_vector_e_censor_noSEX.py"
OUTPUT_DIR="/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_noSEX"

mkdir -p "$OUTPUT_DIR"

# Run each batch sequentially (wait for each to complete)
for i in {0..7}; do
    START=$((i * BATCH_SIZE))
    END=$((START + BATCH_SIZE))
    
    echo "=========================================="
    echo "Starting batch $i: samples $START to $END"
    echo "=========================================="
    
    python "$SCRIPT" \
        --start_index $START \
        --end_index $END \
        --num_epochs 200 \
        --learning_rate 0.1 \
        --lambda_reg 0.01 \
        --K 20 \
        --W 0.0001 \
        --include_pcs True \
        --output_dir "$OUTPUT_DIR" \
        2>&1 | tee "$OUTPUT_DIR/batch_${i}_${START}_${END}.log"
    
    echo "Batch $i completed!"
    echo ""
done

echo "All 8 batches completed!"