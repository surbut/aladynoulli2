"""
Compare two washout comparison CSV files to verify they're similar.
Comparing:
1. pooled_retrospective/washout_comparison_all_offsets.csv
2. pooled_retrospective_withlocal/washout_comparison_all_offsets.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("COMPARING WASHOUT COMPARISON RESULTS")
print("="*80)

# Load both files
file1_path = Path("pyScripts/new_oct_revision/new_notebooks/results/washout/pooled_retrospective/washout_comparison_all_offsets.csv")
file2_path = Path("pyScripts/new_oct_revision/new_notebooks/results/washout/pooled_retrospective_withlocal/washout_comparison_all_offsets.csv")

print(f"\n1. Loading files...")
df1 = pd.read_csv(file1_path, index_col=0)
df2 = pd.read_csv(file2_path, index_col=0)

print(f"   File 1 (pooled_retrospective): {df1.shape}")
print(f"   File 2 (pooled_retrospective_withlocal): {df2.shape}")

# Check if they have the same diseases
print(f"\n2. Disease comparison:")
diseases1 = set(df1.index)
diseases2 = set(df2.index)
print(f"   File 1 diseases: {len(diseases1)}")
print(f"   File 2 diseases: {len(diseases2)}")
print(f"   Common diseases: {len(diseases1 & diseases2)}")
if diseases1 != diseases2:
    print(f"   ⚠️  Different diseases!")
    print(f"   Only in file 1: {diseases1 - diseases2}")
    print(f"   Only in file 2: {diseases2 - diseases1}")

# Compare common diseases
common_diseases = sorted(list(diseases1 & diseases2))
print(f"\n3. Comparing AUC values for {len(common_diseases)} common diseases...")

# Extract AUC columns (0yr_AUC, 1yr_AUC, 2yr_AUC)
auc_cols = [col for col in df1.columns if '_AUC' in col]
print(f"   AUC columns: {auc_cols}")

differences = []
for disease in common_diseases:
    for col in auc_cols:
        val1 = df1.loc[disease, col]
        val2 = df2.loc[disease, col]
        diff = abs(val1 - val2)
        differences.append({
            'Disease': disease,
            'Column': col,
            'File1': val1,
            'File2': val2,
            'Difference': diff,
            'Relative_Diff_Pct': (diff / val1 * 100) if val1 > 0 else 0
        })

diff_df = pd.DataFrame(differences)

print(f"\n4. Difference statistics:")
print(f"   Mean absolute difference: {diff_df['Difference'].mean():.6f}")
print(f"   Max absolute difference: {diff_df['Difference'].max():.6f}")
print(f"   Median absolute difference: {diff_df['Difference'].median():.6f}")
print(f"   Std of differences: {diff_df['Difference'].std():.6f}")
print(f"   Mean relative difference: {diff_df['Relative_Diff_Pct'].mean():.4f}%")
print(f"   Max relative difference: {diff_df['Relative_Diff_Pct'].max():.4f}%")

# Show top differences
print(f"\n5. Top 10 largest differences:")
top_diff = diff_df.nlargest(10, 'Difference')[['Disease', 'Column', 'File1', 'File2', 'Difference', 'Relative_Diff_Pct']]
for idx, row in top_diff.iterrows():
    print(f"   {row['Disease']:25s} {row['Column']:10s} File1: {row['File1']:.6f}  File2: {row['File2']:.6f}  Diff: {row['Difference']:.6f} ({row['Relative_Diff_Pct']:.3f}%)")

# Check if they're very similar (within tolerance)
tolerance = 0.01  # 1% difference
very_close = (diff_df['Difference'] < tolerance).sum()
total = len(diff_df)
pct_close = (very_close / total) * 100

print(f"\n6. Similarity check:")
print(f"   Values within {tolerance} (1%): {very_close}/{total} ({pct_close:.2f}%)")

if diff_df['Difference'].max() < 0.01:
    print(f"   ✓ Files are very similar (max diff < 0.01)")
elif diff_df['Difference'].max() < 0.05:
    print(f"   ✓ Files are similar (max diff < 0.05)")
else:
    print(f"   ⚠️  Files have some differences (max diff >= 0.05)")

# Per-disease summary
print(f"\n7. Per-disease summary (mean difference across all AUC columns):")
per_disease_diff = diff_df.groupby('Disease')['Difference'].mean().sort_values(ascending=False)
print(f"   Top 5 diseases with largest mean differences:")
for disease, mean_diff in per_disease_diff.head(5).items():
    print(f"   {disease:25s} Mean diff: {mean_diff:.6f}")

# Per-timepoint summary
print(f"\n8. Per-timepoint summary (mean difference across all diseases):")
per_timepoint_diff = diff_df.groupby('Column')['Difference'].mean()
for col, mean_diff in per_timepoint_diff.items():
    print(f"   {col:15s} Mean diff: {mean_diff:.6f}")

print("\n" + "="*80)
print("COMPARISON COMPLETE")
print("="*80)

