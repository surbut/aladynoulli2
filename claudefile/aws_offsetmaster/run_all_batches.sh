#!/bin/bash
# Run all 10 batches sequentially
# Usage: ./run_all_batches.sh [s3_bucket] [max_age_offset]

set -e

S3_BUCKET="${1:-s3://sarah-research-aladynoulli}"
MAX_AGE_OFFSET="${2:-10}"
BATCH_SIZE=10000
NUM_BATCHES=10
TOTAL_START_TIME=$(date +%s)

echo "=========================================="
echo "Running All $NUM_BATCHES Batches"
echo "=========================================="
echo "S3 Bucket: $S3_BUCKET"
echo "Batch Size: $BATCH_SIZE"
echo "Max Age Offset: $MAX_AGE_OFFSET"
echo "Start Time: $(date)"
echo ""

for i in {0..9}; do
    BATCH_START=$((i * BATCH_SIZE))
    BATCH_END=$(((i + 1) * BATCH_SIZE))
    BATCH_NUM=$((i + 1))
    
    BATCH_START_TIME=$(date +%s)
    
    echo "=========================================="
    echo "Batch $BATCH_NUM/$NUM_BATCHES: Indices $BATCH_START-$BATCH_END"
    echo "Started: $(date)"
    echo "=========================================="
    
    # Run the batch
    ./run_aws.sh $BATCH_START $BATCH_END $S3_BUCKET $MAX_AGE_OFFSET
    
    # Check exit code
    EXIT_CODE=$?
    BATCH_END_TIME=$(date +%s)
    BATCH_DURATION=$((BATCH_END_TIME - BATCH_START_TIME))
    BATCH_MINUTES=$((BATCH_DURATION / 60))
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Batch $BATCH_NUM completed successfully!"
        echo "  Duration: ${BATCH_MINUTES} minutes"
        echo "  Completed: $(date)"
        echo ""
    else
        echo ""
        echo "✗ Batch $BATCH_NUM FAILED with exit code $EXIT_CODE"
        echo "  Check the log file for details"
        echo ""
        read -p "Continue with next batch? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Stopping batch execution."
            exit 1
        fi
    fi
    
    # Small delay between batches (optional)
    if [ $BATCH_NUM -lt $NUM_BATCHES ]; then
        echo "Waiting 10 seconds before next batch..."
        sleep 10
        echo ""
    fi
done

TOTAL_END_TIME=$(date +%s)
TOTAL_DURATION=$((TOTAL_END_TIME - TOTAL_START_TIME))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))

echo "=========================================="
echo "All Batches Complete!"
echo "=========================================="
echo "Total batches: $NUM_BATCHES"
echo "Total duration: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m"
echo "End time: $(date)"
echo ""
echo "Results uploaded to: $S3_BUCKET/results/"
echo ""
echo "⚠️  IMPORTANT: Terminate your EC2 instance now!"
echo "   Instance type: c7i.24xlarge"
echo "   Cost: ~\$6/hour"
echo "   Total runtime: ~${TOTAL_HOURS}h ${TOTAL_MINUTES}m"
echo "   Estimated cost: ~\$$((TOTAL_HOURS * 6))-$$(((TOTAL_HOURS + 1) * 6))"
echo ""
echo "To terminate: Go to EC2 Console → Select instance → Instance state → Terminate"




