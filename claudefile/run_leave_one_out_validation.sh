#!/bin/bash
# Leave-one-out validation: Create checkpoints excluding test batches and run predictions

# Test batches to exclude (one at a time)
TEST_BATCHES=(39 38 37)

# Base paths
DATA_DIR="/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/"
RETROSPECTIVE_PATTERN="/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_retrospective_full/enrollment_model_W0.0001_batch_*_*.pt"
ENROLLMENT_PATTERN="/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_prediction_jointphi_sex_pcs/enrollment_model_W0.0001_batch_*_*.pt"
OUTPUT_BASE="/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/"

echo "========================================================================"
echo "Leave-One-Out Validation Setup"
echo "========================================================================"
echo "Testing batches: ${TEST_BATCHES[@]}"
echo ""

# Create leave-one-out checkpoints for each test batch
for EXCLUDE_BATCH in "${TEST_BATCHES[@]}"; do
    echo "----------------------------------------------------------------------"
    echo "Creating checkpoints excluding batch ${EXCLUDE_BATCH}"
    echo "----------------------------------------------------------------------"
    
    # Create retrospective checkpoint (excluding this batch)
    python claudefile/create_leave_one_out_checkpoints.py \
        --exclude_batch ${EXCLUDE_BATCH} \
        --analysis_type retrospective \
        --data_dir "${DATA_DIR}" \
        --retrospective_pattern "${RETROSPECTIVE_PATTERN}" \
        --output_dir "${OUTPUT_BASE}" \
        --total_batches 40
    
    echo ""
done

echo "========================================================================"
echo "Checkpoints created! Next steps:"
echo "========================================================================"
echo ""
echo "For each excluded batch, run predictions:"
echo ""
for EXCLUDE_BATCH in "${TEST_BATCHES[@]}"; do
    START_IDX=$((EXCLUDE_BATCH * 10000))
    END_IDX=$(((EXCLUDE_BATCH + 1) * 10000))
    
    echo "# Test batch ${EXCLUDE_BATCH} (samples ${START_IDX}-${END_IDX}):"
    echo "python claudefile/run_aladyn_predict_with_master.py \\"
    echo "    --trained_model_path ${OUTPUT_BASE}master_for_fitting_pooled_all_data_exclude_batch_${EXCLUDE_BATCH}.pt \\"
    echo "    --data_dir ${DATA_DIR} \\"
    echo "    --output_dir ${OUTPUT_BASE}leave_one_out_validation/batch_${EXCLUDE_BATCH}/ \\"
    echo "    --start_batch ${EXCLUDE_BATCH} \\"
    echo "    --max_batches 1"
    echo ""
done

