#!/usr/bin/env python3
"""
Calculate corrected medians for age-offset analysis by replacing offset 0 
with 0-year washout predictions.
"""

import pandas as pd
import numpy as np

# Load age-offset pivot data
age_offset = pd.read_csv('age_offset/pooled_retrospective/age_offset_aucs_pivot_batch_0_10000.csv', index_col=0)

# Load washout 0-year data
washout_0yr = pd.read_csv('washout/pooled_retrospective/washout_0yr_results.csv', index_col=0)

print("="*80)
print("CORRECTED MEDIANS (replacing offset 0 with washout_0yr)")
print("="*80)

# For each disease, replace offset 0 with washout_0yr value and recalculate
corrected_stats = []

for disease in age_offset.index:
    if disease not in washout_0yr.index:
        continue
    
    # Get original offsets (excluding NaN)
    offsets = age_offset.loc[disease].dropna().values
    washout_val = washout_0yr.loc[disease, 'AUC']
    
    # Replace offset 0 (first value) with washout value
    if len(offsets) > 0:
        offsets_corrected = offsets.copy()
        offsets_corrected[0] = washout_val
        
        # Calculate statistics
        median_corrected = np.median(offsets_corrected)
        mean_corrected = np.mean(offsets_corrected)
        min_corrected = np.min(offsets_corrected)
        max_corrected = np.max(offsets_corrected)
        
        # Original stats (for comparison)
        median_original = np.median(offsets)
        mean_original = np.mean(offsets)
        
        corrected_stats.append({
            'Disease': disease,
            'Washout_0yr': washout_val,
            'Original_Offset0': offsets[0],
            'Original_Median': median_original,
            'Original_Mean': mean_original,
            'Corrected_Median': median_corrected,
            'Corrected_Mean': mean_corrected,
            'Corrected_Min': min_corrected,
            'Corrected_Max': max_corrected,
            'N_Offsets': len(offsets_corrected)
        })

df_corrected = pd.DataFrame(corrected_stats)

# Save results
df_corrected.to_csv('corrected_age_offset_medians.csv', index=False)

print("\nKey Diseases:")
print(df_corrected[df_corrected['Disease'].isin(['ASCVD', 'All_Cancers', 'Breast_Cancer', 'Diabetes', 'Heart_Failure'])].to_string(index=False))

print(f"\n✓ Saved to corrected_age_offset_medians.csv")
print(f"  Total diseases: {len(df_corrected)}")














