#!/bin/bash
# Download all pi batches and models from AWS EC2 instance
# Uses /dev/shm on AWS for intermediate storage to avoid disk space issues

set -e

# AWS connection details
AWS_HOST="ubuntu@ip-172-31-43-126"
AWS_REMOTE_DIR="~/aladyn_project/output/retrospective_pooled"
LOCAL_DOWNLOAD_DIR="$HOME/Downloads/aws_retrospective_pooled"

# Create local download directory
mkdir -p "$LOCAL_DOWNLOAD_DIR"

echo "================================================================================"
echo "DOWNLOADING AWS RETROSPECTIVE POOLED RESULTS"
echo "================================================================================"
echo "Remote: $AWS_HOST:$AWS_REMOTE_DIR"
echo "Local:  $LOCAL_DOWNLOAD_DIR"
echo "================================================================================"

# First, check what's on AWS
echo ""
echo "Checking remote directory contents..."
ssh $AWS_HOST "cd $AWS_REMOTE_DIR && echo 'Files:' && ls -lh && echo '' && echo 'pibatch/ contents:' && ls -lh pibatch/ | head -20 && echo '...' && echo 'models/ contents:' && ls -lh models/ | head -20"

# Check disk space on AWS
echo ""
echo "Checking AWS disk space..."
ssh $AWS_HOST "df -h /dev/shm && df -h /dev/root"

# Create tar archives on AWS using /dev/shm (RAM disk) to avoid disk space issues
echo ""
echo "================================================================================"
echo "CREATING TAR ARCHIVES ON AWS (using /dev/shm for scratch space)"
echo "================================================================================"

# Archive pi batches
echo "Creating pi_batches.tar.gz..."
ssh $AWS_HOST "cd $AWS_REMOTE_DIR && tar -czf /dev/shm/pi_batches.tar.gz -C pibatch . && ls -lh /dev/shm/pi_batches.tar.gz"

# Archive models
echo "Creating models.tar.gz..."
ssh $AWS_HOST "cd $AWS_REMOTE_DIR && tar -czf /dev/shm/models.tar.gz -C models . && ls -lh /dev/shm/models.tar.gz"

# Download the archives
echo ""
echo "================================================================================"
echo "DOWNLOADING ARCHIVES FROM AWS"
echo "================================================================================"

echo "Downloading pi_batches.tar.gz..."
scp $AWS_HOST:/dev/shm/pi_batches.tar.gz "$LOCAL_DOWNLOAD_DIR/"

echo "Downloading models.tar.gz..."
scp $AWS_HOST:/dev/shm/models.tar.gz "$LOCAL_DOWNLOAD_DIR/"

echo "Downloading batch_info.pt..."
scp $AWS_HOST:$AWS_REMOTE_DIR/batch_info.pt "$LOCAL_DOWNLOAD_DIR/"

echo "Downloading aws_first_10_batches_models.tar.gz..."
scp $AWS_HOST:$AWS_REMOTE_DIR/aws_first_10_batches_models.tar.gz "$LOCAL_DOWNLOAD_DIR/"

# Extract archives locally
echo ""
echo "================================================================================"
echo "EXTRACTING ARCHIVES LOCALLY"
echo "================================================================================"

cd "$LOCAL_DOWNLOAD_DIR"

echo "Extracting pi_batches.tar.gz..."
mkdir -p pibatch
tar -xzf pi_batches.tar.gz -C pibatch
echo "✓ Extracted pi batches"

echo "Extracting models.tar.gz..."
mkdir -p models
tar -xzf models.tar.gz -C models
echo "✓ Extracted models"

# Clean up remote /dev/shm files
echo ""
echo "Cleaning up remote /dev/shm files..."
ssh $AWS_HOST "rm -f /dev/shm/pi_batches.tar.gz /dev/shm/models.tar.gz"

# Summary
echo ""
echo "================================================================================"
echo "DOWNLOAD COMPLETE"
echo "================================================================================"
echo "Local directory: $LOCAL_DOWNLOAD_DIR"
echo ""
echo "Contents:"
ls -lh "$LOCAL_DOWNLOAD_DIR"
echo ""
echo "Pi batches: $(ls -1 $LOCAL_DOWNLOAD_DIR/pibatch/ | wc -l) files"
echo "Models: $(ls -1 $LOCAL_DOWNLOAD_DIR/models/ | wc -l) files"
echo "================================================================================"

