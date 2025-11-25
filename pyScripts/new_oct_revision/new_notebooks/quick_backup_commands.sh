#!/bin/bash
# Quick commands to run on EC2 instance
# Run these commands directly on EC2

# Set your S3 bucket name here
S3_BUCKET="sarah-research-aladynoulli"

# Create archive
cd /home/ubuntu/aladyn_project/output
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_NAME="retrospective_pooled_results_${TIMESTAMP}.tar.gz"

echo "Creating archive: ${ARCHIVE_NAME}"
tar -czf ${ARCHIVE_NAME} retrospective_pooled/

# Upload to S3
echo "Uploading to S3..."
aws s3 cp ${ARCHIVE_NAME} s3://${S3_BUCKET}/results/${ARCHIVE_NAME}

# Verify upload
echo "Verifying upload..."
aws s3 ls s3://${S3_BUCKET}/results/ | grep ${ARCHIVE_NAME}

echo ""
echo "=========================================="
echo "BACKUP COMPLETE"
echo "=========================================="
echo "Archive: ${ARCHIVE_NAME}"
echo "S3 location: s3://${S3_BUCKET}/results/${ARCHIVE_NAME}"
echo ""
echo "To download locally, run:"
echo "  aws s3 cp s3://${S3_BUCKET}/results/${ARCHIVE_NAME} ~/Downloads/"
echo ""
echo "To shut down EC2:"
echo "  sudo shutdown -h now"

