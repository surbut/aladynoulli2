#!/bin/bash
# Script to backup results to S3 and shut down EC2 instance
# Run this on EC2 instance

# Set variables
RESULTS_DIR="/home/ubuntu/aladyn_project/output/retrospective_pooled"
S3_BUCKET="sarah-research-aladynoulli"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_NAME="retrospective_pooled_results_${TIMESTAMP}.tar.gz"
S3_KEY="results/${ARCHIVE_NAME}"

echo "=========================================="
echo "BACKING UP RESULTS TO S3"
echo "=========================================="
echo "Results directory: ${RESULTS_DIR}"
echo "S3 bucket: s3://${S3_BUCKET}/${S3_KEY}"
echo "Archive name: ${ARCHIVE_NAME}"
echo ""

# Step 1: Create tar archive
echo "Step 1: Creating tar archive..."
cd $(dirname ${RESULTS_DIR})
tar -czf ${ARCHIVE_NAME} $(basename ${RESULTS_DIR})
if [ $? -eq 0 ]; then
    echo "✓ Archive created: ${ARCHIVE_NAME}"
    echo "  Size: $(du -h ${ARCHIVE_NAME} | cut -f1)"
else
    echo "✗ Failed to create archive"
    exit 1
fi

# Step 2: Upload to S3
echo ""
echo "Step 2: Uploading to S3..."
aws s3 cp ${ARCHIVE_NAME} s3://${S3_BUCKET}/${S3_KEY}
if [ $? -eq 0 ]; then
    echo "✓ Upload successful: s3://${S3_BUCKET}/${S3_KEY}"
else
    echo "✗ Upload failed"
    exit 1
fi

# Step 3: Verify upload
echo ""
echo "Step 3: Verifying upload..."
aws s3 ls s3://${S3_BUCKET}/${S3_KEY}
if [ $? -eq 0 ]; then
    echo "✓ Verification successful"
else
    echo "✗ Verification failed"
    exit 1
fi

# Step 4: Clean up local archive (optional)
echo ""
read -p "Delete local archive ${ARCHIVE_NAME}? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm ${ARCHIVE_NAME}
    echo "✓ Local archive deleted"
fi

echo ""
echo "=========================================="
echo "BACKUP COMPLETE"
echo "=========================================="
echo "Archive location: s3://${S3_BUCKET}/${S3_KEY}"
echo ""
echo "To download locally, run:"
echo "  aws s3 cp s3://${S3_BUCKET}/${S3_KEY} ~/Downloads/${ARCHIVE_NAME}"
echo ""
echo "To shut down EC2 instance, run:"
echo "  sudo shutdown -h now"
echo "  OR"
echo "  aws ec2 stop-instances --instance-ids <your-instance-id>"

