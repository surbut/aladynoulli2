#!/usr/bin/env python3
"""
Compare age offset CSV results (batch 0-10K, offset 0) with washout evaluation (using pi_full).

The CSV contains results from age_offset runs using batch-specific pi tensors.
The comparison script used pi_full (from washout) but evaluated on batch 0-10K.

This verifies that the pi tensors are similar even though they came from different runs.
"""

import pandas as pd
import numpy as np

# Load CSV results (age offset, batch 0-10K, offset 0)
csv_path = '/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks/results/age_offset/pooled_retrospective/age_offset_aucs_pivot_batch_0_10000.csv'
csv_df = pd.read_csv(csv_path, index_col=0)

# Results from comparison script (washout pi_full, evaluated on batch 0-10K, offset 0)
washout_results = {
    'ASCVD': 0.8670,
    'Diabetes': 0.7659,
    'Atrial_Fib': 0.7926,
    'CKD': 0.8929,
    'All_Cancers': 0.7640,
    'Stroke': 0.7203,
    'Heart_Failure': 0.8290,
    'Pneumonia': 0.4930,
    'COPD': 0.7298,
    'Osteoporosis': 0.6508,
    'Anemia': 0.6004,
    'Colorectal_Cancer': 0.9198,
    'Breast_Cancer': 0.7999,
    'Prostate_Cancer': 0.7945,
    'Lung_Cancer': 0.7648,
    'Bladder_Cancer': 0.9907,
    'Secondary_Cancer': 0.6581,
    'Depression': 0.9058,
    'Anxiety': 0.8713,
}

print("="*80)
print("COMPARING AGE OFFSET CSV (batch 0-10K) vs WASHOUT EVALUATION (pi_full, batch 0-10K)")
print("="*80)
print("\nNote: CSV uses batch-specific pi tensors from age_offset runs")
print("      Washout uses pi_full (from washout run) evaluated on batch 0-10K")
print("\n" + "="*80)
print(f"{'Disease':<30} {'CSV (offset 0)':<18} {'Washout (0yr)':<18} {'Difference':<15} {'Match':<10}")
print("-"*80)

matches = []
close_matches = []
mismatches = []

for disease, washout_auc in washout_results.items():
    if disease in csv_df.index:
        csv_auc = csv_df.loc[disease, '0']  # Offset 0 column
        
        if pd.isna(csv_auc):
            print(f"{disease:<30} {'N/A':<18} {washout_auc:>17.4f} {'N/A':<15} {'N/A':<10}")
            continue
        
        diff = abs(washout_auc - csv_auc)
        
        # Very close match (< 0.001 difference)
        if diff < 0.001:
            match_status = "✓ Exact"
            matches.append(disease)
        # Close match (< 0.01 difference)
        elif diff < 0.01:
            match_status = "✓ Close"
            close_matches.append(disease)
        else:
            match_status = "⚠ Diff"
            mismatches.append((disease, csv_auc, washout_auc, diff))
        
        print(f"{disease:<30} {csv_auc:>17.4f} {washout_auc:>17.4f} {diff:>14.4f} {match_status:<10}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"✓ Exact matches (<0.001): {len(matches)}")
print(f"✓ Close matches (<0.01): {len(close_matches)}")
print(f"⚠ Differences (>=0.01): {len(mismatches)}")

if matches:
    print(f"\nExact matches: {', '.join(matches)}")

if close_matches:
    print(f"\nClose matches: {', '.join(close_matches)}")

if mismatches:
    print(f"\nDifferences:")
    for disease, csv_auc, washout_auc, diff in mismatches:
        print(f"  {disease}: CSV={csv_auc:.4f}, Washout={washout_auc:.4f}, Diff={diff:.4f}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
total_compared = len(matches) + len(close_matches) + len(mismatches)
if len(mismatches) == 0:
    print("✅ PERFECT MATCH! All AUCs match within 0.01")
    print("   This confirms that pi tensors from different runs are very similar.")
elif len(mismatches) <= 2:
    print("✅ VERY CLOSE! Most AUCs match within 0.01")
    print("   Minor differences likely due to:")
    print("   - Different random seeds/initialization")
    print("   - Different batch processing (batch 0-10K vs full 400K)")
    print("   - Minor numerical differences")
else:
    print("⚠ SOME DIFFERENCES FOUND")
    print("   Check mismatches above to understand discrepancies")

