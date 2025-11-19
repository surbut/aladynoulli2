#!/bin/bash
# Run predictions using fixed phi from ENROLLMENT data (analysis #2)
# This uses phi estimated from enrollment data to make predictions on enrollment data

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Configuration
BATCH_SIZE=10000
NUM_EPOCHS=200
LEARNING_RATE=1e-1
LAMBDA_REG=1e-2
INCLUDE_PCS=True

# Paths
DATA_DIR="/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/"
COVARIATES_PATH="/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv"

# Fixed phi from enrollment data (using batch 0_10000, or could pool all batches)
# Option 1: Use single batch (recommended since batches are close)
ENROLLMENT_PHI_CHECKPOINT="/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_prediction_jointphi_sex_pcs/enrollment_model_W0.0001_batch_0_10000.pt"

# Option 2: If you want to use pooled phi, you'd need to create that first
# ENROLLMENT_PHI_CHECKPOINT="/path/to/pooled_enrollment_phi.pt"

# Output directory
OUTPUT_DIR="/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_ENROLLMENT_withpcs_batchrun/"

echo "========================================"
echo "Aladyn Predictions - Fixed Phi from Enrollment"
echo "========================================"
echo "Fixed phi checkpoint: $ENROLLMENT_PHI_CHECKPOINT"
echo "Output directory: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Epochs per batch: $NUM_EPOCHS"
echo "Include PCs: $INCLUDE_PCS"
echo "========================================"
echo ""

# Check if checkpoint exists
if [ ! -f "$ENROLLMENT_PHI_CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $ENROLLMENT_PHI_CHECKPOINT"
    exit 1
fi

# Run prediction script
python run_aladyn_predict.py \
    --trained_model_path "$ENROLLMENT_PHI_CHECKPOINT" \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --lambda_reg $LAMBDA_REG \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --covariates_path "$COVARIATES_PATH" \
    --include_pcs $INCLUDE_PCS

echo ""
echo "========================================"
echo "Predictions complete!"
echo "Output saved to: $OUTPUT_DIR"
echo "========================================"

