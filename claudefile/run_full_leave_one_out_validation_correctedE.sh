#!/bin/bash
# Full leave-one-out validation pipeline for corrected E data
# This script runs the complete validation: checkpoint creation, predictions, and AUC calculation

# Configuration
TOTAL_BATCHES=40
DATA_DIR="/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/"
BATCH_PATTERN="/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized/enrollment_model_W0.0001_batch_*_*.pt"
OUTPUT_BASE="/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/"

echo "========================================================================"
echo "Full Leave-One-Out Validation for Corrected E Data"
echo "========================================================================"
echo "Total batches: ${TOTAL_BATCHES}"
echo ""

# Step 1: Create leave-one-out checkpoints
echo "========================================================================"
echo "STEP 1: Creating leave-one-out checkpoints"
echo "========================================================================"
for BATCH_IDX in $(seq 0 $((TOTAL_BATCHES - 1))); do
    echo "Creating checkpoint excluding batch ${BATCH_IDX}..."
    python claudefile/create_leave_one_out_checkpoints_correctedE.py \
        --exclude_batch ${BATCH_IDX} \
        --data_dir "${DATA_DIR}" \
        --batch_pattern "${BATCH_PATTERN}" \
        --output_dir "${OUTPUT_BASE}" \
        --total_batches ${TOTAL_BATCHES}
    echo ""
done

echo "========================================================================"
echo "STEP 2: Running predictions on excluded batches"
echo "========================================================================"
echo "This will run predictions for all ${TOTAL_BATCHES} batches..."
echo "You can run this manually or use:"
echo "  python claudefile/run_leave_one_out_predictions_correctedE.py --all_batches"
echo ""

# Step 3: Calculate AUC
echo "========================================================================"
echo "STEP 3: Calculating 10-year AUC for each batch"
echo "========================================================================"
echo "After predictions are complete, run:"
echo "  python claudefile/calculate_leave_one_out_auc_correctedE.py --all_batches"
echo ""

# Step 4: Compare results
echo "========================================================================"
echo "STEP 4: Comparing leave-one-out AUCs to overall AUC"
echo "========================================================================"
echo "After AUC calculation is complete, run:"
echo "  python claudefile/compare_leave_one_out_auc_correctedE.py"
echo ""

echo "========================================================================"
echo "Pipeline setup complete!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "1. Checkpoints created (if successful above)"
echo "2. Run predictions: python claudefile/run_leave_one_out_predictions_correctedE.py --all_batches"
echo "3. Calculate AUC: python claudefile/calculate_leave_one_out_auc_correctedE.py --all_batches"
echo "4. Compare results: python claudefile/compare_leave_one_out_auc_correctedE.py"

