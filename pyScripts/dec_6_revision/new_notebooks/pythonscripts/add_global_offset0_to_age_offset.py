#!/usr/bin/env python3
"""
Add a column to age_offset pivot CSV that pulls the global offset 0 from washout_0yr_results.csv.

This ensures consistency: offset 0 uses the full dataset (washout_0yr), while offsets 1-9 use the batch-specific results.
"""

import pandas as pd
from pathlib import Path

# Paths
base_dir = Path('pyScripts/dec_6_revision/new_notebooks/results')
age_offset_pivot = base_dir / 'age_offset/pooled_retrospective/age_offset_aucs_pivot_batch_0_10000.csv'
washout_0yr = base_dir / 'washout/pooled_retrospective/washout_0yr_results.csv'
output_file = base_dir / 'age_offset/pooled_retrospective/age_offset_aucs_pivot_batch_0_10000_with_global0.csv'

print("="*80)
print("ADDING GLOBAL OFFSET 0 COLUMN TO AGE OFFSET PIVOT")
print("="*80)

# Load files
print(f"\n1. Loading age_offset pivot: {age_offset_pivot}")
age_offset_df = pd.read_csv(age_offset_pivot)
print(f"   Shape: {age_offset_df.shape}")
print(f"   Columns: {list(age_offset_df.columns)}")

print(f"\n2. Loading washout_0yr (global offset 0): {washout_0yr}")
washout_df = pd.read_csv(washout_0yr)
print(f"   Shape: {washout_df.shape}")
print(f"   Columns: {list(washout_df.columns)}")

# Create a mapping from Disease to AUC for washout_0yr
washout_dict = dict(zip(washout_df['Disease'], washout_df['AUC']))
print(f"\n3. Created mapping for {len(washout_dict)} diseases from washout_0yr")

# Add new column '0_global' that uses washout_0yr values
print("\n4. Adding '0_global' column (offset 0 from full dataset)...")
age_offset_df['0_global'] = age_offset_df['Disease'].map(washout_dict)

# Check for diseases that don't have a match
missing = age_offset_df[age_offset_df['0_global'].isna()]
if len(missing) > 0:
    print(f"\n⚠️  Warning: {len(missing)} diseases in age_offset not found in washout_0yr:")
    print(missing[['Disease']].to_string(index=False))
    print("\n   These will have NaN for 0_global column")
else:
    print("   ✓ All diseases matched!")

# Reorder columns: Disease, 0_global, 0, 1, 2, ..., 9
# Keep original '0' column for comparison
cols = ['Disease', '0_global'] + [col for col in age_offset_df.columns if col not in ['Disease', '0_global']]
age_offset_df = age_offset_df[cols]

# Show comparison for a few diseases
print("\n5. Sample comparison (original offset 0 vs global offset 0):")
print("="*80)
print(f"{'Disease':<25} {'0 (batch)':<15} {'0_global (full)':<18} {'Difference':<15}")
print("-"*80)
sample_diseases = ['Diabetes', 'ASCVD', 'Breast_Cancer', 'Heart_Failure', 'Atrial_Fib']
for disease in sample_diseases:
    if disease in age_offset_df['Disease'].values:
        row = age_offset_df[age_offset_df['Disease'] == disease].iloc[0]
        batch_0 = row['0'] if pd.notna(row['0']) else 'N/A'
        global_0 = row['0_global'] if pd.notna(row['0_global']) else 'N/A'
        if pd.notna(row['0']) and pd.notna(row['0_global']):
            diff = row['0_global'] - row['0']
            print(f"{disease:<25} {batch_0:<15.4f} {global_0:<18.4f} {diff:<15.4f}")
        else:
            print(f"{disease:<25} {batch_0:<15} {global_0:<18} {'N/A':<15}")

# Save updated file
print(f"\n6. Saving updated file: {output_file}")
age_offset_df.to_csv(output_file, index=False)
print(f"   ✓ Saved! Shape: {age_offset_df.shape}")

# Also update the summary file to recalculate median using 0_global
print("\n7. Updating summary file with median using 0_global...")
summary_file = base_dir / 'age_offset/pooled_retrospective/age_offset_aucs_summary_batch_0_10000.csv'
summary_df = pd.read_csv(summary_file)

# Recalculate median for each disease using 0_global + offsets 1-9
print("   Recalculating medians...")
new_medians = []
for _, row in summary_df.iterrows():
    disease = row['Disease']
    disease_row = age_offset_df[age_offset_df['Disease'] == disease]
    
    if len(disease_row) > 0:
        # Get values: 0_global, 1, 2, ..., 9
        values = []
        if pd.notna(disease_row.iloc[0]['0_global']):
            values.append(disease_row.iloc[0]['0_global'])
        for offset in range(1, 10):
            val = disease_row.iloc[0][str(offset)]
            if pd.notna(val):
                values.append(val)
        
        if len(values) > 0:
            median_val = sorted(values)[len(values) // 2] if len(values) % 2 == 1 else \
                        (sorted(values)[len(values) // 2 - 1] + sorted(values)[len(values) // 2]) / 2
            new_medians.append(median_val)
        else:
            new_medians.append(row['median'])  # Keep original if no values
    else:
        new_medians.append(row['median'])  # Keep original if disease not found

summary_df['median_with_global0'] = new_medians

# Show comparison for sample diseases
print("\n   Sample median comparison:")
print("="*80)
print(f"{'Disease':<25} {'Original Median':<18} {'Median (w/ global0)':<20} {'Difference':<15}")
print("-"*80)
for disease in sample_diseases:
    if disease in summary_df['Disease'].values:
        row = summary_df[summary_df['Disease'] == disease].iloc[0]
        orig = row['median']
        new = row['median_with_global0']
        diff = new - orig
        print(f"{disease:<25} {orig:<18.4f} {new:<20.4f} {diff:<15.4f}")

# Save updated summary
summary_output = base_dir / 'age_offset/pooled_retrospective/age_offset_aucs_summary_batch_0_10000_with_global0.csv'
summary_df.to_csv(summary_output, index=False)
print(f"\n   ✓ Saved updated summary: {summary_output}")

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
print(f"Updated files:")
print(f"  1. {output_file}")
print(f"  2. {summary_output}")

