# ============================================================================
# STEP 2B: GENERATE WASHOUT TIME HORIZON PREDICTIONS (RUN ONCE, THEN MARK AS "NOT EVALUATED")
# ============================================================================
"""
IMPORTANT: This cell generates 10-year and 30-year predictions with washout for pooled_retrospective.
- Generates dynamic 10-year and 30-year predictions with 1-year washout
- Processes ALL 400K patients at once using pre-computed pi tensors
- Uses evaluate_major_diseases_wsex_with_bootstrap_dynamic_withwashout()
- Only runs for pooled_retrospective approach (main clinically implementable approach)
- Run once, then mark as "not evaluated"
- Script: generate_washout_time_horizons.py
- Results saved to: results/washout_time_horizons/pooled_retrospective/
"""

import subprocess
import sys
from pathlib import Path

# Set script directory
script_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks')

print("="*80)
print("GENERATING WASHOUT TIME HORIZON PREDICTIONS")
print("="*80)
print("\nThis will generate 10-year and 30-year predictions with 1-year washout")
print("Approach: pooled_retrospective only")
print("\nNOTE: Run once, then mark this cell as 'not evaluated'.")
print("="*80)

# Generate washout time horizon predictions for pooled retrospective
print("\nGenerating pooled_retrospective washout time horizon predictions...")
result = subprocess.run([
    sys.executable,
    str(script_dir / 'generate_washout_time_horizons.py'),
    '--n_bootstraps', '100',
    '--washout_years', '1'
], capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
if result.returncode != 0:
    print(f"\n⚠️  WARNING: Script exited with return code {result.returncode}")
else:
    print("✓ pooled_retrospective washout time horizon predictions completed successfully")

print("\n" + "="*80)
print("WASHOUT TIME HORIZON PREDICTIONS COMPLETE")
print("="*80)
print("\nResults saved to: results/washout_time_horizons/pooled_retrospective/")
print("  - washout_1yr_10yr_results.csv")
print("  - washout_1yr_30yr_results.csv")
print("  - washout_1yr_comparison_all_horizons.csv")


