#!/bin/bash
# Simple nohup approach (if SLURM not available)

# Navigate to script directory
cd /path/to/your/script/directory

# Run R script in background with nohup
# Output will go to nohup.out and mediation_analysis.log
nohup Rscript mediation_analysis.R > mediation_analysis.log 2>&1 &

# Get the job PID
echo "Job started with PID: $!"
echo "Monitor progress with: tail -f mediation_analysis.log"
echo "Check if still running: ps -p $!"

