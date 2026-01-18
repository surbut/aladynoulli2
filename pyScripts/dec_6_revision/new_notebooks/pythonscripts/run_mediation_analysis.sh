#!/bin/bash
#SBATCH --job-name=mediation_analysis
#SBATCH --output=mediation_analysis_%j.out
#SBATCH --error=mediation_analysis_%j.err
#SBATCH --time=48:00:00        # 48 hours (adjust as needed)
#SBATCH --mem=64G              # Memory (adjust based on your data size)
#SBATCH --cpus-per-task=1      # Single core (R is single-threaded unless parallelized)
#SBATCH --partition=normal     # Adjust partition name if different on your system

# Load R module if needed (adjust module name for your system)
# module load R/4.2.0

# Navigate to script directory
cd /path/to/your/script/directory

# Run R script
Rscript mediation_analysis.R

echo "Job completed at $(date)"

