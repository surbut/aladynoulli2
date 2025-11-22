#!/bin/bash
# Main script to run the age offset predictions on AWS
# Usage: ./run_aws.sh [start_index] [end_index] [s3_bucket] [max_age_offset]

set -e

echo "=========================================="
echo "Aladyn Age Offset Predictions - AWS Runner"
echo "=========================================="

# Configuration - set these or pass as arguments
START_INDEX="${1:-0}"
END_INDEX="${2:-10000}"
S3_BUCKET="${3:-s3://sarah-research-aladynoulli}"
MAX_AGE_OFFSET="${4:-10}"

# Directories
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$PROJECT_DIR/data_for_running"
OUTPUT_DIR="$PROJECT_DIR/output"
LOG_DIR="$PROJECT_DIR/logs"

# Create directories
mkdir -p "$DATA_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Log file with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/run_${START_INDEX}_${END_INDEX}_${TIMESTAMP}.log"

echo "Configuration:"
echo "  Start Index: $START_INDEX"
echo "  End Index: $END_INDEX"
echo "  S3 Bucket: $S3_BUCKET"
echo "  Max Age Offset: $MAX_AGE_OFFSET"
echo "  Data Directory: $DATA_DIR"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Log File: $LOG_FILE"
echo ""

# Step 1: Download data from S3
echo "=========================================="
echo "Step 1: Downloading data from S3"
echo "=========================================="
"$PROJECT_DIR/download_from_s3.sh" "$S3_BUCKET/data_for_running" "$DATA_DIR" 2>&1 | tee -a "$LOG_FILE"

# Step 2: Verify required files
echo ""
echo "=========================================="
echo "Step 2: Verifying required files"
echo "=========================================="
REQUIRED_FILES=(
    "Y_tensor.pt"
    "E_matrix.pt"
    "G_matrix.pt"
    "model_essentials.pt"
    "reference_trajectories.pt"
    "master_for_fitting_pooled_all_data.pt"
    "baselinagefamh_withpcs.csv"
)

ALL_FOUND=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$DATA_DIR/$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file - NOT FOUND"
        ALL_FOUND=false
    fi
done

if [ "$ALL_FOUND" = false ]; then
    echo "ERROR: Some required files are missing!"
    exit 1
fi

# Step 3: Check if we need to set up Python environment
echo ""
echo "=========================================="
echo "Step 3: Checking Python environment"
echo "=========================================="

# Try to activate conda if available
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    echo "Found conda installation"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    echo "Found conda installation"
fi

# Check if conda environment exists
if command -v conda &> /dev/null; then
    if conda env list | grep -q "new_env_pyro2"; then
        echo "Activating conda environment: new_env_pyro2"
        conda activate new_env_pyro2
    else
        echo "WARNING: Conda environment 'new_env_pyro2' not found"
        echo "You may need to create it from environment.yml"
    fi
fi

# Step 4: Check for required Python modules
echo ""
echo "=========================================="
echo "Step 4: Checking Python dependencies"
echo "=========================================="
python -c "import torch; import numpy; import pandas; import scipy; print('All basic dependencies found')" 2>&1 | tee -a "$LOG_FILE" || {
    echo "ERROR: Some Python dependencies are missing!"
    echo "Please install required packages or activate the correct conda environment"
    exit 1
}

# Step 5: Check for required Python files (utils.py, clust_huge_amp_fixedPhi.py)
echo ""
echo "=========================================="
echo "Step 5: Checking for required Python modules"
echo "=========================================="

# Check if we're in a directory with utils.py and clust_huge_amp_fixedPhi.py
PYTHON_PATH="$PROJECT_DIR"
if [ ! -f "$PYTHON_PATH/../pyScripts/utils.py" ] && [ ! -f "$PYTHON_PATH/utils.py" ]; then
    # Try to find them
    if [ -f "../../pyScripts/utils.py" ]; then
        PYTHON_PATH="$PROJECT_DIR/../../pyScripts"
    elif [ -f "../../../pyScripts/utils.py" ]; then
        PYTHON_PATH="$PROJECT_DIR/../../../pyScripts"
    else
        echo "WARNING: utils.py not found in expected locations"
        echo "You may need to copy utils.py and clust_huge_amp_fixedPhi.py to $PROJECT_DIR"
    fi
fi

export PYTHONPATH="$PYTHON_PATH:$PYTHONPATH"

# Step 6: Run the Python script
echo ""
echo "=========================================="
echo "Step 6: Running predictions"
echo "=========================================="
echo "This may take a while..."

cd "$PROJECT_DIR"

python "$PROJECT_DIR/forAWS_offsetmasterfix.py" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --start_index "$START_INDEX" \
    --end_index "$END_INDEX" \
    --max_age_offset "$MAX_AGE_OFFSET" \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "ERROR: Python script failed with exit code $EXIT_CODE"
    echo "Check the log file for details: $LOG_FILE"
    exit 1
fi

# Step 7: Upload results to S3
echo ""
echo "=========================================="
echo "Step 7: Uploading results to S3"
echo "=========================================="
RUN_NAME="age_offset_${START_INDEX}_${END_INDEX}_${TIMESTAMP}"
"$PROJECT_DIR/upload_to_s3.sh" "$OUTPUT_DIR" "$S3_BUCKET/results" "$RUN_NAME" 2>&1 | tee -a "$LOG_FILE"

echo ""
echo "=========================================="
echo "All done!"
echo "=========================================="
echo "Results uploaded to: $S3_BUCKET/results/$RUN_NAME/"
echo "Log file: $LOG_FILE"

