#!/bin/bash
# Download data from S3 bucket to local directory
# Usage: ./download_from_s3.sh [s3_bucket] [local_dir]

set -e

# Configuration
S3_BUCKET="${1:-s3://sarah-research-aladynoulli/data_for_running}"
LOCAL_DATA_DIR="${2:-./data_for_running}"

echo "=========================================="
echo "Downloading data from S3"
echo "=========================================="
echo "S3 Bucket: $S3_BUCKET"
echo "Local Directory: $LOCAL_DATA_DIR"
echo ""

# Create local directory if it doesn't exist
mkdir -p "$LOCAL_DATA_DIR"

# Download required files
echo "Downloading required files..."

REQUIRED_FILES=(
    "Y_tensor.pt"
    "E_matrix.pt"
    "G_matrix.pt"
    "model_essentials.pt"
    "reference_trajectories.pt"
    "master_for_fitting_pooled_all_data.pt"
    "baselinagefamh_withpcs.csv"
)

for file in "${REQUIRED_FILES[@]}"; do
    echo "  Downloading $file..."
    aws s3 cp "$S3_BUCKET/$file" "$LOCAL_DATA_DIR/$file" || {
        echo "ERROR: Failed to download $file"
        exit 1
    }
done

echo ""
echo "=========================================="
echo "Download complete!"
echo "=========================================="
echo "Files downloaded to: $LOCAL_DATA_DIR"
echo ""
echo "Verifying files..."
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$LOCAL_DATA_DIR/$file" ]; then
        size=$(du -h "$LOCAL_DATA_DIR/$file" | cut -f1)
        echo "  ✓ $file ($size)"
    else
        echo "  ✗ $file - NOT FOUND"
    fi
done

