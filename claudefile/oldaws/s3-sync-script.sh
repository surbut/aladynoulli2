#!/bin/bash
# Script to sync data with S3 for persistent storage
# This helps reduce costs by using S3 instead of expensive EBS volumes

set -e

# Configuration - UPDATE THESE VALUES
S3_BUCKET="s3://your-bucket-name"
S3_DATA_PREFIX="aladynoulli-data"
S3_RESULTS_PREFIX="aladynoulli-results"

LOCAL_DATA_DIR="/home/ubuntu/aladynoulli2/data"
LOCAL_RESULTS_DIR="/home/ubuntu/aladynoulli2/results"

# Function to download data from S3
download_data() {
    echo "Downloading data from S3..."
    echo "Source: $S3_BUCKET/$S3_DATA_PREFIX"
    echo "Destination: $LOCAL_DATA_DIR"

    mkdir -p "$LOCAL_DATA_DIR"

    aws s3 sync \
        "$S3_BUCKET/$S3_DATA_PREFIX/" \
        "$LOCAL_DATA_DIR/" \
        --exclude "*" \
        --include "*.pt" \
        --include "*.csv" \
        --include "*.rds"

    echo "Data download complete!"
}

# Function to upload results to S3
upload_results() {
    echo "Uploading results to S3..."
    echo "Source: $LOCAL_RESULTS_DIR"
    echo "Destination: $S3_BUCKET/$S3_RESULTS_PREFIX"

    if [ ! -d "$LOCAL_RESULTS_DIR" ]; then
        echo "No results directory found. Skipping upload."
        return
    fi

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)

    aws s3 sync \
        "$LOCAL_RESULTS_DIR/" \
        "$S3_BUCKET/$S3_RESULTS_PREFIX/$TIMESTAMP/" \
        --exclude "*.log"

    # Also keep a "latest" copy
    aws s3 sync \
        "$LOCAL_RESULTS_DIR/" \
        "$S3_BUCKET/$S3_RESULTS_PREFIX/latest/" \
        --exclude "*.log"

    echo "Results upload complete!"
    echo "Results available at: $S3_BUCKET/$S3_RESULTS_PREFIX/$TIMESTAMP/"
}

# Function to backup specific result files
backup_file() {
    local FILE=$1
    echo "Backing up $FILE to S3..."

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BASENAME=$(basename "$FILE")

    aws s3 cp \
        "$FILE" \
        "$S3_BUCKET/$S3_RESULTS_PREFIX/backups/${TIMESTAMP}_${BASENAME}"

    echo "Backup complete!"
}

# Function to list available data in S3
list_s3_data() {
    echo "Available data in S3:"
    aws s3 ls "$S3_BUCKET/$S3_DATA_PREFIX/" --recursive --human-readable
}

# Function to list results in S3
list_s3_results() {
    echo "Available results in S3:"
    aws s3 ls "$S3_BUCKET/$S3_RESULTS_PREFIX/" --recursive --human-readable
}

# Main script logic
case "${1:-}" in
    download)
        download_data
        ;;
    upload)
        upload_results
        ;;
    backup)
        if [ -z "$2" ]; then
            echo "Usage: $0 backup <file_path>"
            exit 1
        fi
        backup_file "$2"
        ;;
    list-data)
        list_s3_data
        ;;
    list-results)
        list_s3_results
        ;;
    sync-all)
        echo "Syncing all data and results..."
        download_data
        upload_results
        echo "Sync complete!"
        ;;
    *)
        echo "Usage: $0 {download|upload|backup|list-data|list-results|sync-all}"
        echo ""
        echo "Commands:"
        echo "  download      - Download data from S3 to local"
        echo "  upload        - Upload results from local to S3"
        echo "  backup <file> - Backup a specific file to S3"
        echo "  list-data     - List available data in S3"
        echo "  list-results  - List available results in S3"
        echo "  sync-all      - Download data and upload results"
        echo ""
        echo "Before using, update S3_BUCKET in this script!"
        exit 1
        ;;
esac
