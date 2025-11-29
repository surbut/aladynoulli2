#!/bin/bash
# Run all 40 batches (0-39), 2 at a time in parallel
# Usage: bash run_all_batches_parallel.sh

set -e

echo "=========================================="
echo "Running All 40 Batches (2 at a time)"
echo "=========================================="
echo ""

# Create logs directory
mkdir -p ~/batch_logs

# Function to run a single batch
run_batch() {
    local batch_num=$1
    local start_idx=$((batch_num * 10000))
    local end_idx=$((start_idx + 10000))
    
    echo "Starting batch $batch_num ($start_idx to $end_idx)..."
    
    bash setup_and_run_ec2.sh $start_idx $end_idx > ~/batch_logs/batch_${batch_num}.log 2>&1
    
    echo "Completed batch $batch_num"
}

# Run batches in pairs (0-1, 2-3, 4-5, ..., 38-39)
for pair_start in {0..38..2}; do
    batch1=$pair_start
    batch2=$((pair_start + 1))
    
    echo ""
    echo "=========================================="
    echo "Running batches $batch1 and $batch2 in parallel"
    echo "=========================================="
    
    # Run both batches in background with nohup
    nohup bash -c "run_batch $batch1" > ~/batch_logs/nohup_batch_${batch1}.log 2>&1 &
    PID1=$!
    
    nohup bash -c "run_batch $batch2" > ~/batch_logs/nohup_batch_${batch2}.log 2>&1 &
    PID2=$!
    
    echo "Batch $batch1 PID: $PID1"
    echo "Batch $batch2 PID: $PID2"
    
    # Wait for both to complete
    echo "Waiting for batches $batch1 and $batch2 to complete..."
    wait $PID1
    STATUS1=$?
    wait $PID2
    STATUS2=$?
    
    if [ $STATUS1 -eq 0 ] && [ $STATUS2 -eq 0 ]; then
        echo "✓ Both batches $batch1 and $batch2 completed successfully"
    else
        echo "⚠️  Batch $batch1 exit status: $STATUS1"
        echo "⚠️  Batch $batch2 exit status: $STATUS2"
    fi
    
    echo ""
done

echo "=========================================="
echo "All 40 batches complete!"
echo "=========================================="
echo ""
echo "Check logs in ~/batch_logs/"
echo "Check results in /results/"

