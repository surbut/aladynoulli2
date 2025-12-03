#!/bin/bash
# Upload enrollment age offset model files to S3 and prepare for EC2 shutdown
# Run this script on the EC2 instance

set -e

echo "=========================================="
echo "Uploading enrollment age offset models to S3"
echo "=========================================="

# S3 bucket and path
S3_BUCKET="s3://sarah-research-aladynoulli"
S3_PATH="${S3_BUCKET}/predictions/enrollment_age_offset_fixedphi_withpcs_newrun"

# Current directory (where the .pt files are located)
CURRENT_DIR=$(pwd)

echo "Current directory: $CURRENT_DIR"
echo "S3 destination: $S3_PATH"
echo ""

# Check if we're in the right directory
if [ ! -f "pi_enroll_fixedphi_age_offset_0_sex_0_10000_try2_withpcs_newrun.pt" ]; then
    echo "WARNING: Expected .pt files not found in current directory."
    echo "Looking for files matching pattern: *_enroll_fixedphi_age_offset_*.pt"
    echo ""
fi

# Count files to upload
PT_COUNT=$(ls -1 *.pt 2>/dev/null | wc -l)
if [ "$PT_COUNT" -eq 0 ]; then
    echo "ERROR: No .pt files found in current directory!"
    exit 1
fi

echo "Found $PT_COUNT .pt files to upload"
echo ""

# Upload all .pt files
echo "Uploading files..."
aws s3 sync . "$S3_PATH/" \
    --exclude "*" \
    --include "*.pt" \
    --no-progress

echo ""
echo "=========================================="
echo "Upload complete!"
echo "=========================================="
echo "Files uploaded to: $S3_PATH"
echo ""

# Verify upload
echo "Verifying upload..."
aws s3 ls "$S3_PATH/" | grep "\.pt$" | wc -l | xargs -I {} echo "Uploaded {} files to S3"

echo ""
echo "=========================================="
echo "Ready to shut down EC2 instance"
echo "=========================================="
echo ""
echo "To shut down the instance, choose one of the following:"
echo ""
echo "Option 1: Stop instance (can restart later)"
echo "  aws ec2 stop-instances --instance-ids i-xxxxxxxxxxxxx"
echo ""
echo "Option 2: Terminate instance (permanent deletion)"
echo "  aws ec2 terminate-instances --instance-ids i-xxxxxxxxxxxxx"
echo ""
echo "Option 3: Shutdown from EC2 (stops instance)"
echo "  sudo shutdown -h now"
echo ""
echo "Option 4: Use AWS Console"
echo "  Go to EC2 Console → Select instance → Instance state → Stop/Terminate"
echo ""
echo "To download files later from your local machine:"
echo "  aws s3 sync $S3_PATH/ ./local_enrollment_models/"
echo ""







