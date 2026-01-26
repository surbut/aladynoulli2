#!/usr/bin/env python
"""
Reshape AUC results from long form to wide form (disease x method table)
"""

import pandas as pd
import numpy as np

# Read the results files (try 5 configs first, fall back to 4)
static_file_5 = '/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/all_5_configs_5batches_static_10yr_auc_results.csv'
static_file_4 = '/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/all_4_configs_5batches_static_10yr_auc_results.csv'
dynamic_file_5 = '/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/all_5_configs_5batches_dynamic_1yr_auc_results.csv'
dynamic_file_4 = '/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/all_4_configs_5batches_dynamic_1yr_auc_results.csv'

import os
static_file = static_file_5 if os.path.exists(static_file_5) else static_file_4
dynamic_file = dynamic_file_5 if os.path.exists(dynamic_file_5) else dynamic_file_4

print("="*80)
print("STATIC 10-YEAR AUC RESULTS")
print("="*80)

# Load static results
df_static = pd.read_csv(static_file)

# Pivot to wide format
static_wide = df_static.pivot_table(
    index='disease',
    columns='config',
    values='auc',
    aggfunc='first'
)

# Reorder columns to match logical progression
config_order = ['original', 'fixedk_freeg', 'fixedg_freek', 'fixedgk', 'fixedgk_nolr']
# Only include configs that exist in the data
available_configs = [c for c in config_order if c in static_wide.columns]
static_wide = static_wide[available_configs]

# Rename columns for clarity
col_names = {
    'original': 'Original\n(free γ, free κ)',
    'fixedk_freeg': 'Fixed κ\n(free γ)',
    'fixedg_freek': 'Fixed γ\n(free κ)',
    'fixedgk': 'Fixed γκ\n(regularized)',
    'fixedgk_nolr': 'Fixed γκ\n(unregularized)'
}
static_wide.columns = [col_names.get(col, col) for col in static_wide.columns]

# Sort by disease name
static_wide = static_wide.sort_index()

# Format AUC values to 3 decimal places
static_wide = static_wide.round(3)

print("\nAUC by Disease and Method:")
print(static_wide.to_string())

# Calculate differences from original
print("\n" + "="*80)
print("DIFFERENCES FROM ORIGINAL (Original - Fixed)")
print("="*80)
diff_static = static_wide.copy()
for col in diff_static.columns[1:]:
    diff_static[col] = static_wide.iloc[:, 0] - static_wide[col]

print(diff_static.iloc[:, 1:].to_string())

# Save to CSV
output_static = static_file.replace('.csv', '_wide.csv')
static_wide.to_csv(output_static)
print(f"\n✓ Saved wide format to: {output_static}")

print("\n" + "="*80)
print("DYNAMIC 1-YEAR AUC RESULTS")
print("="*80)

# Load dynamic results
df_dynamic = pd.read_csv(dynamic_file)

# Pivot to wide format
dynamic_wide = df_dynamic.pivot_table(
    index='disease',
    columns='config',
    values='auc',
    aggfunc='first'
)

# Reorder columns
available_configs_dyn = [c for c in config_order if c in dynamic_wide.columns]
dynamic_wide = dynamic_wide[available_configs_dyn]

# Rename columns
dynamic_wide.columns = [col_names.get(col, col) for col in dynamic_wide.columns]

# Sort by disease name
dynamic_wide = dynamic_wide.sort_index()

# Format AUC values
dynamic_wide = dynamic_wide.round(3)

print("\nAUC by Disease and Method:")
print(dynamic_wide.to_string())

# Calculate differences from original
print("\n" + "="*80)
print("DIFFERENCES FROM ORIGINAL (Original - Fixed)")
print("="*80)
diff_dynamic = dynamic_wide.copy()
for col in diff_dynamic.columns[1:]:
    diff_dynamic[col] = dynamic_wide.iloc[:, 0] - dynamic_wide[col]

print(diff_dynamic.iloc[:, 1:].to_string())

# Save to CSV
output_dynamic = dynamic_file.replace('.csv', '_wide.csv')
dynamic_wide.to_csv(output_dynamic)
print(f"\n✓ Saved wide format to: {output_dynamic}")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print("\nStatic 10-Year - Mean AUC:")
print(static_wide.mean().round(3))

print("\nStatic 10-Year - Mean Absolute Difference from Original:")
print(diff_static.iloc[:, 1:].abs().mean().round(3))

print("\nDynamic 1-Year - Mean AUC:")
print(dynamic_wide.mean().round(3))

print("\nDynamic 1-Year - Mean Absolute Difference from Original:")
print(diff_dynamic.iloc[:, 1:].abs().mean().round(3))
