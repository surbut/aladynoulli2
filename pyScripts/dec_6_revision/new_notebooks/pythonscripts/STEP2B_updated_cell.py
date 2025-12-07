# ============================================================================
# STEP 2B: GENERATE RETROSPECTIVE WASHOUT ALL HORIZONS (RUN ONCE, THEN MARK AS "NOT EVALUATED")
# ============================================================================
"""
IMPORTANT: This cell generates 10-year, 30-year, and static 10-year predictions with washout for pooled_retrospective.
- Generates dynamic 10-year and 30-year predictions with 1-year washout
- Generates static 10-year predictions (1-year score) with 1-year washout
- Processes ALL 400K patients at once using pre-computed pi tensors
- Uses _from_pi versions: evaluate_major_diseases_wsex_with_bootstrap_dynamic_withwashout_from_pi() and 
  evaluate_major_diseases_wsex_with_bootstrap_withwashout_from_pi()
- Only runs for pooled_retrospective approach (main clinically implementable approach)
- Run once, then mark as "not evaluated"
- Script: generate_retrospective_washout_all_horizons.py
- Results saved to: results/washout_time_horizons/pooled_retrospective/
"""

import subprocess
import sys
from pathlib import Path

# Set script directory
script_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks')

print("="*80)
print("GENERATING RETROSPECTIVE WASHOUT ALL HORIZONS")
print("="*80)
print("\nThis will generate:")
print("  - 10-year dynamic predictions with 1-year washout")
print("  - 30-year dynamic predictions with 1-year washout")
print("  - Static 10-year predictions (1-year score) with 1-year washout")
print("Approach: pooled_retrospective only")
print("Processing: ALL 400K patients at once")
print("\nNOTE: Run once, then mark this cell as 'not evaluated'.")
print("="*80)

# Generate washout predictions for all horizons
print("\nGenerating pooled_retrospective washout predictions for all horizons...")
result = subprocess.run([
    sys.executable,
    str(script_dir / 'generate_retrospective_washout_all_horizons.py'),
    '--n_bootstraps', '100',
    '--washout_years', '1'
], capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
if result.returncode != 0:
    print(f"\n⚠️  WARNING: Script exited with return code {result.returncode}")
else:
    print("✓ pooled_retrospective washout predictions completed successfully")

print("\n" + "="*80)
print("RETROSPECTIVE WASHOUT PREDICTIONS COMPLETE")
print("="*80)
print("\nResults saved to: results/washout_time_horizons/pooled_retrospective/")
print("  - washout_1yr_10yr_dynamic_results.csv")
print("  - washout_1yr_30yr_dynamic_results.csv")
print("  - washout_1yr_10yr_static_results.csv")
print("  - washout_1yr_comparison_all_horizons.csv")


