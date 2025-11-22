#!/bin/bash
# Resume script for batches 16-40
# After restarting computer, run this to continue from batch 16

cd /Users/sarahurbut/aladynoulli2/claudefile

echo "Resuming from batch 16 to batch 40..."
echo "This will process 25 batches (batches 16-40 inclusive)"
echo ""

nohup python run_aladyn_predict_with_master.py \
    --trained_model_path /Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/master_for_fitting_pooled_enrollment_data.pt \
    --output_dir /Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_ENROLLMENT_pooled/ \
    --start_batch 16 \
    --max_batches 25 \
    > resume_batch_16.log 2>&1 &

PID=$!
echo "Process started with PID: $PID"
echo "Log file: resume_batch_16.log"
echo ""
echo "To monitor progress:"
echo "  tail -f resume_batch_16.log"
echo ""
echo "To check if still running:"
echo "  ps aux | grep run_aladyn_predict_with_master"

