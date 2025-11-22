#!/bin/bash
# Upload results from local directory to S3 bucket
# Usage: ./upload_to_s3.sh [local_dir] [s3_bucket] [run_name]

set -e

# Configuration
LOCAL_OUTPUT_DIR="${1:-./output}"
S3_BUCKET="${2:-s3://sarah-research-aladynoulli/results}"
RUN_NAME="${3:-$(date +%Y%m%d_%H%M%S)}"

echo "=========================================="
echo "Uploading results to S3"
echo "=========================================="
echo "Local Directory: $LOCAL_OUTPUT_DIR"
echo "S3 Bucket: $S3_BUCKET/$RUN_NAME"
echo ""

# Check if output directory exists
if [ ! -d "$LOCAL_OUTPUT_DIR" ]; then
    echo "ERROR: Output directory not found: $LOCAL_OUTPUT_DIR"
    exit 1
fi

# Check if there are any files to upload
if [ -z "$(ls -A $LOCAL_OUTPUT_DIR)" ]; then
    echo "WARNING: Output directory is empty. Nothing to upload."
    exit 0
fi

# Upload all .pt files (model checkpoints and predictions)
echo "Uploading results..."
aws s3 sync "$LOCAL_OUTPUT_DIR" "$S3_BUCKET/$RUN_NAME/" \
    --exclude "*" \
    --include "*.pt" \
    --no-progress

echo ""
echo "=========================================="
echo "Upload complete!"
echo "=========================================="
echo "Results uploaded to: $S3_BUCKET/$RUN_NAME/"
echo ""
echo "To download results later:"
echo "  aws s3 sync $S3_BUCKET/$RUN_NAME/ $LOCAL_OUTPUT_DIR/"

