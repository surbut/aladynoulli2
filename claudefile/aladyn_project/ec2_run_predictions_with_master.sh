#!/bin/bash
# Run predictions using master checkpoint files (pooled phi)
# This script supports three analysis types:
# 1. Fixed phi from retrospective (all data) - use master_for_fitting_pooled_all_data.pt
# 2. Fixed phi from enrollment data - use master_for_fitting_pooled_enrollment_data.pt
# 3. Joint phi (uses different script - run_joint_ENROLL_joint.sh)

set -e

echo "=========================================="
echo "Aladyn Prediction Runner - Master Checkpoint"
echo "=========================================="

# Activate conda
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

# Activate the environment
echo "Activating conda environment..."
conda activate new_env_pyro2

# Navigate to project directory
cd ~/aladyn_project

# Download data from S3
echo "Downloading data from S3..."
aws s3 sync s3://sarah-research-aladynoulli/data_for_running/ ~/aladyn_project/data_for_running/ --no-progress

# Verify required files exist
echo "Verifying required files..."
REQUIRED_FILES=(
    "Y_tensor.pt"
    "E_enrollment_full.pt"
    "G_matrix.pt"
    "model_essentials.pt"
    "reference_trajectories.pt"
    "baselinagefamh_withpcs.csv"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$HOME/aladyn_project/data_for_running/$file" ]; then
        echo "ERROR: Required file not found: $file"
        exit 1
    fi
done

echo "All required files present!"

# Determine which analysis to run based on argument
ANALYSIS_TYPE="${1:-enrollment}"  # Default to enrollment
MAX_BATCHES="${2:-}"  # Optional max batches limit

case $ANALYSIS_TYPE in
    "retrospective")
        MASTER_CHECKPOINT="$HOME/aladyn_project/data_for_running/master_for_fitting_pooled_all_data.pt"
        OUTPUT_DIR="$HOME/aladyn_project/output/fixedphi_retrospective_pooled"
        LOG_FILE="$HOME/aladyn_project/logs/predict_retrospective_pooled.log"
        echo "Running: Fixed phi from pooled retrospective (all data)"
        ;;
    "enrollment")
        MASTER_CHECKPOINT="$HOME/aladyn_project/data_for_running/master_for_fitting_pooled_enrollment_data.pt"
        OUTPUT_DIR="$HOME/aladyn_project/output/fixedphi_enrollment_pooled"
        LOG_FILE="$HOME/aladyn_project/logs/predict_enrollment_pooled.log"
        echo "Running: Fixed phi from pooled enrollment data"
        ;;
    *)
        echo "ERROR: Unknown analysis type: $ANALYSIS_TYPE"
        echo "Usage: $0 [retrospective|enrollment] [max_batches]"
        exit 1
        ;;
esac

# Check if master checkpoint exists
if [ ! -f "$MASTER_CHECKPOINT" ]; then
    echo "ERROR: Master checkpoint not found at $MASTER_CHECKPOINT"
    echo "Please ensure the master checkpoint file was downloaded from S3"
    echo "Expected files:"
    echo "  - master_for_fitting_pooled_all_data.pt (for retrospective)"
    echo "  - master_for_fitting_pooled_enrollment_data.pt (for enrollment)"
    exit 1
fi

echo "Using master checkpoint: $MASTER_CHECKPOINT"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"

# Create output and log directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$HOME/aladyn_project/logs"

# Build command
CMD="python ~/aladyn_project/run_aladyn_predict_with_master.py \
    --trained_model_path \"$MASTER_CHECKPOINT\" \
    --data_dir ~/aladyn_project/data_for_running/ \
    --output_dir \"$OUTPUT_DIR\" \
    --covariates_path ~/aladyn_project/data_for_running/baselinagefamh_withpcs.csv \
    --batch_size 10000 \
    --num_epochs 200 \
    --learning_rate 0.1 \
    --lambda_reg 0.01"

# Add max_batches if specified
if [ -n "$MAX_BATCHES" ]; then
    CMD="$CMD --max_batches $MAX_BATCHES"
    echo "Limiting to $MAX_BATCHES batches"
fi

# Run the prediction script
echo "=========================================="
echo "Starting Aladyn predictions..."
echo "=========================================="

nohup $CMD > "$LOG_FILE" 2>&1 &

PID=$!
echo "Prediction script started with PID: $PID"
echo ""
echo "Monitor progress with: tail -f $LOG_FILE"
echo "Check if still running: ps aux | grep run_aladyn_predict_with_master"
echo "To upload results to S3: aws s3 sync $OUTPUT_DIR s3://sarah-research-aladynoulli/predictions/${ANALYSIS_TYPE}_pooled/"
echo ""
echo "To stop: kill $PID"

