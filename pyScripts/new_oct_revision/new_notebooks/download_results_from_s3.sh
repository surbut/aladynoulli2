#!/bin/bash
# Script to download results from S3 to local machine
# Run this on your LOCAL machine

# Set variables
S3_BUCKET="sarah-research-aladynoulli"
S3_KEY="results/aladynoulli_results_YYYYMMDD_HHMMSS.tar.gz"  # Update with actual key
DOWNLOAD_DIR="${HOME}/Downloads"  # Or wherever you want to download

echo "=========================================="
echo "DOWNLOADING RESULTS FROM S3"
echo "=========================================="
echo "S3 location: s3://${S3_BUCKET}/${S3_KEY}"
echo "Download directory: ${DOWNLOAD_DIR}"
echo ""

# List available files in S3
echo "Available files in s3://${S3_BUCKET}/results/:"
aws s3 ls s3://${S3_BUCKET}/results/ --human-readable
echo ""

# Download the file
ARCHIVE_NAME=$(basename ${S3_KEY})
echo "Downloading ${ARCHIVE_NAME}..."
aws s3 cp s3://${S3_BUCKET}/${S3_KEY} ${DOWNLOAD_DIR}/${ARCHIVE_NAME}
if [ $? -eq 0 ]; then
    echo "✓ Download successful: ${DOWNLOAD_DIR}/${ARCHIVE_NAME}"
    echo "  Size: $(du -h ${DOWNLOAD_DIR}/${ARCHIVE_NAME} | cut -f1)"
    echo ""
    echo "To extract, run:"
    echo "  cd ${DOWNLOAD_DIR}"
    echo "  tar -xzf ${ARCHIVE_NAME}"
else
    echo "✗ Download failed"
    exit 1
fi

