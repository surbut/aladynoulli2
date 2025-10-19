#!/bin/bash
# Run this script after ec2_setup.sh and uploading files
# This script downloads data and runs the Aladyn predictions

set -e

echo "=========================================="
echo "Aladyn Prediction Runner"
echo "=========================================="

# Activate conda
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

# Create environment from yml file
echo "Creating conda environment from environment.yml..."
cd ~/aladyn_project
conda env create -f environment.yml

# Activate the environment
echo "Activating environment..."
conda activate new_env_pyro2

# Download data from S3
echo "Downloading data from S3..."
aws s3 sync s3://sarah-research-aladynoulli/data_for_running/ ~/aladyn_project/data_for_running/ --no-progress

# Verify required files exist
echo "Verifying required files..."
REQUIRED_FILES=(
    "~/aladyn_project/data_for_running/Y_tensor.pt"
    "~/aladyn_project/data_for_running/E_enrollment_full.pt"
    "~/aladyn_project/data_for_running/G_matrix.pt"
    "~/aladyn_project/data_for_running/model_essentials.pt"
    "~/aladyn_project/data_for_running/reference_trajectories.pt"
    "~/aladyn_project/data_for_running/baselinagefamh_withpcs.csv"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "ERROR: Required file not found: $file"
        exit 1
    fi
done

echo "All required files present!"

# Get trained model path
TRAINED_MODEL="$HOME/aladyn_project/data_for_running/enrollment_model_W0.0001_fulldata_sexspecific.pt"

# Allow override from command line
if [ -n "$1" ]; then
    TRAINED_MODEL="$1"
    echo "Using provided model: $TRAINED_MODEL"
fi

# Check if trained model exists
if [ ! -f "$TRAINED_MODEL" ]; then
    echo "ERROR: Trained model not found at $TRAINED_MODEL"
    echo "Please ensure the model file was downloaded from S3"
    exit 1
fi

echo "Using trained model: $TRAINED_MODEL"

# Run the prediction script
echo "=========================================="
echo "Starting Aladyn predictions..."
echo "=========================================="

nohup python ~/aladyn_project/run_aladyn_predict.py \
    --trained_model_path "$TRAINED_MODEL" \
    --data_dir ~/aladyn_project/data_for_running/ \
    --output_dir ~/aladyn_project/output/ \
    --covariates_path ~/aladyn_project/data_for_running/baselinagefamh_withpcs.csv \
    --batch_size 10000 \
    --num_epochs 200 \
    --learning_rate 0.1 \
    --lambda_reg 0.01 \
    > ~/aladyn_project/logs/predict.log 2>&1 &

PID=$!
echo "Prediction script started with PID: $PID"
echo "Monitor progress with: tail -f ~/aladyn_project/logs/predict.log"
echo ""
echo "To check if still running: ps aux | grep run_aladyn_predict"
echo "To upload results to S3: aws s3 sync ~/aladyn_project/output/ s3://sarah-research-aladynoulli/predictions/"
