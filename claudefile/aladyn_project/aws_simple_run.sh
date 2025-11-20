#!/bin/bash
# Simple one-command script to run Aladyn predictions on EC2
# Usage: bash aws_simple_run.sh [retrospective|enrollment] [max_batches]

set -e

ANALYSIS="${1:-enrollment}"  # Default: enrollment
MAX_BATCHES="${2:-}"          # Optional: max batches

echo "=========================================="
echo "Running: $ANALYSIS analysis"
echo "=========================================="

# Activate conda
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate new_env_pyro2

# Create directories
mkdir -p ~/aladyn_project/data_for_running
mkdir -p ~/aladyn_project/output
mkdir -p ~/aladyn_project/logs

# Download data from S3 (if needed)
echo "Syncing data from S3..."
aws s3 sync s3://sarah-research-aladynoulli/data_for_running/ ~/aladyn_project/data_for_running/ --no-progress

# Set paths based on analysis type
if [ "$ANALYSIS" == "retrospective" ]; then
    MASTER="$HOME/aladyn_project/data_for_running/master_for_fitting_pooled_all_data.pt"
    OUTDIR="$HOME/aladyn_project/output/retrospective"
    LOG="$HOME/aladyn_project/logs/retrospective.log"
else
    MASTER="$HOME/aladyn_project/data_for_running/master_for_fitting_pooled_enrollment_data.pt"
    OUTDIR="$HOME/aladyn_project/output/enrollment"
    LOG="$HOME/aladyn_project/logs/enrollment.log"
fi

# Build command
CMD="python ~/aladyn_project/run_aladyn_predict_with_master.py \
    --trained_model_path $MASTER \
    --data_dir ~/aladyn_project/data_for_running/ \
    --output_dir $OUTDIR \
    --covariates_path ~/aladyn_project/data_for_running/baselinagefamh_withpcs.csv"

if [ -n "$MAX_BATCHES" ]; then
    CMD="$CMD --max_batches $MAX_BATCHES"
fi

# Run in background
echo "Starting prediction..."
echo "Log: $LOG"
nohup $CMD > $LOG 2>&1 &

echo "Started! PID: $!"
echo "Watch: tail -f $LOG"

