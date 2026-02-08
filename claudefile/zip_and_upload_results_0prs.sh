#!/bin/bash
# Zip results_0prs with multithreaded gzip (pigz), upload to S3, with download instructions.
#
# Run on EC2 after batches complete:
#   ./zip_and_upload_results_0prs.sh
#
# Then on local machine to download:
#   aws s3 cp s3://sarah-research-aladynoulli/results_0prs/enrollment_joint_nolr_0prs.tar.gz . --no-sign-request
#   tar -xzf enrollment_joint_nolr_0prs.tar.gz

set -e

RESULTS_DIR="${1:-/home/ubuntu/aladyn_project/results_0prs}"
BUCKET="${2:-s3://sarah-research-aladynoulli}"
S3_KEY="results_0prs/enrollment_joint_nolr_0prs.tar.gz"
ARCHIVE_NAME="enrollment_joint_nolr_0prs.tar.gz"

# Create archive in parent of results dir so extract gives results_0prs/
PARENT=$(dirname "$RESULTS_DIR")
DIRNAME=$(basename "$RESULTS_DIR")

cd "$PARENT" || exit 1

echo "=========================================="
echo "Zipping and uploading results_0prs"
echo "=========================================="
echo "Source: $RESULTS_DIR"
echo "Bucket: $BUCKET/$S3_KEY"
echo ""

# Use pigz (multithreaded gzip) if available, else gzip
if command -v pigz &> /dev/null; then
    echo "Using pigz (multithreaded)..."
    tar -I pigz -cvf "$ARCHIVE_NAME" "$DIRNAME"
else
    echo "pigz not found, using gzip (single-thread). Install with: sudo apt install pigz"
    tar -czvf "$ARCHIVE_NAME" "$DIRNAME"
fi

echo ""
echo "Archive created: $PARENT/$ARCHIVE_NAME"
ls -lh "$ARCHIVE_NAME"

# Verify all 40 batch files are in the archive
COUNT=$(tar -tzf "$ARCHIVE_NAME" 2>/dev/null | grep -c '\.pt$' || echo "0")
if [ "$COUNT" = "40" ]; then
    echo "✓ Verified: 40 .pt files in archive"
else
    echo "✗ ERROR: Expected 40 .pt files, found $COUNT"
    echo "Contents:"
    tar -tzf "$ARCHIVE_NAME" | grep '\.pt$' || true
    exit 1
fi

echo ""
echo "Uploading to S3..."
aws s3 cp "$ARCHIVE_NAME" "$BUCKET/$S3_KEY"

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
echo ""
echo "To download on your local machine:"
echo "  aws s3 cp $BUCKET/$S3_KEY ."
echo "  tar -xzf $ARCHIVE_NAME"
echo ""
echo "Or with explicit path:"
echo "  aws s3 cp $BUCKET/$S3_KEY ~/Downloads/"
echo "  cd ~/Downloads && tar -xzf $ARCHIVE_NAME"
echo ""
