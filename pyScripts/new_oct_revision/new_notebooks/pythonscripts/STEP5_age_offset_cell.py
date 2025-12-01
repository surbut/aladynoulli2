# ============================================================================
# STEP 5: GENERATE AGE OFFSET PREDICTIONS (RUN ONCE, THEN MARK AS "NOT EVALUATED")
# ============================================================================
"""
IMPORTANT: This cell generates rolling 1-year predictions using models trained at different time offsets.
- Evaluates 1-year predictions using models trained at enrollment + 0, 1, 2, ..., 9 years
- Uses pre-computed pi batches from AWS run (downloaded to ~/Downloads/age_offset_files/)
- Currently runs on batch 0-10000 (first 10K patients)
- Run once, then mark as "not evaluated"

This analysis shows how model performance changes when predictions are made at different time points
after enrollment, using models that have been retrained with additional follow-up data.
"""

import subprocess
import sys
from pathlib import Path

# Set script directory
script_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks')

print("="*80)
print("GENERATING AGE OFFSET PREDICTIONS")
print("="*80)
print("\nThis will generate rolling 1-year predictions using models trained at")
print("enrollment + 0, 1, 2, ..., 9 years (offsets 0-9).")
print("\nUses pre-computed pi batches from AWS run.")
print("Currently evaluates on batch 0-10000 (first 10K patients).")
print("\nNOTE: Run once, then mark this cell as 'not evaluated'.")
print("="*80)

# Generate age offset predictions for pooled retrospective (main approach)
print("\n1. Generating pooled_retrospective age offset predictions...")
result1 = subprocess.run([
    sys.executable,
    str(script_dir / 'generate_age_offset_predictions.py'),
    '--approach', 'pooled_retrospective',
    '--max_offset', '9',
    '--start_idx', '0',
    '--end_idx', '10000'
], capture_output=True, text=True)
print(result1.stdout)
if result1.stderr:
    print("STDERR:", result1.stderr)
if result1.returncode != 0:
    print(f"\n⚠️  WARNING: Script exited with return code {result1.returncode}")
else:
    print("✓ pooled_retrospective completed successfully")

# Generate age offset predictions for pooled enrollment (for comparison)
print("\n2. Generating pooled_enrollment age offset predictions...")
result2 = subprocess.run([
    sys.executable,
    str(script_dir / 'generate_age_offset_predictions.py'),
    '--approach', 'pooled_enrollment',
    '--max_offset', '9',
    '--start_idx', '0',
    '--end_idx', '10000'
], capture_output=True, text=True)
print(result2.stdout)
if result2.stderr:
    print("STDERR:", result2.stderr)
if result2.returncode != 0:
    print(f"\n⚠️  WARNING: Script exited with return code {result2.returncode}")
else:
    print("✓ pooled_enrollment completed successfully")

print("\n" + "="*80)
print("AGE OFFSET PREDICTIONS COMPLETE")
print("="*80)
print("\nResults saved to: results/age_offset/{approach}/")
print("  - age_offset_aucs_batch_0_10000.csv (AUCs by disease and offset)")
print("  - age_offset_aucs_pivot_batch_0_10000.csv (pivot table for easy viewing)")
print("\nROC curves are plotted for ASCVD by default (saved as PDF).")

