#!/bin/bash
# Run all 40 batches, 2 at a time in parallel with nohup
# You can disconnect and batches will continue running
# Usage: bash run_all_batches_nohup.sh

set -e

echo "=========================================="
echo "Starting All 40 Batches (2 at a time, nohup)"
echo "=========================================="
echo ""

# Create logs directory
mkdir -p ~/batch_logs

# Function to run a single batch
run_batch() {
    local batch_num=$1
    local start_idx=$((batch_num * 10000))
    local end_idx=$((start_idx + 10000))
    
    echo "[$(date)] Starting batch $batch_num ($start_idx to $end_idx)" | tee -a ~/batch_logs/master.log
    
    bash setup_and_run_ec2.sh $start_idx $end_idx >> ~/batch_logs/batch_${batch_num}.log 2>&1
    
    echo "[$(date)] Completed batch $batch_num" | tee -a ~/batch_logs/master.log
}

# Run batches in pairs (0-1, 2-3, 4-5, ..., 38-39)
for pair_start in {0..38..2}; do
    batch1=$pair_start
    batch2=$((pair_start + 1))
    
    echo "Starting batches $batch1 and $batch2..."
    
    # Run both batches in background with nohup
    nohup bash -c "run_batch $batch1" >> ~/batch_logs/nohup_batch_${batch1}.log 2>&1 &
    PID1=$!
    
    nohup bash -c "run_batch $batch2" >> ~/batch_logs/nohup_batch_${batch2}.log 2>&1 &
    PID2=$!
    
    echo "  Batch $batch1: PID $PID1"
    echo "  Batch $batch2: PID $PID2"
    echo "  Logs: ~/batch_logs/batch_${batch1}.log and ~/batch_logs/batch_${batch2}.log"
    echo ""
done

echo "=========================================="
echo "All batches started in background!"
echo "=========================================="
echo ""
echo "Monitor progress:"
echo "  tail -f ~/batch_logs/master.log"
echo ""
echo "Check specific batch:"
echo "  tail -f ~/batch_logs/batch_0.log"
echo ""
echo "Check running processes:"
echo "  ps aux | grep setup_and_run"
echo ""
echo "Results will be saved to: /results/"


