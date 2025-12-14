#!/usr/bin/env python3
"""
Compare effect sizes across different versions to identify what causes larger effects in March version.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# File paths
files = {
    'March (loop, no sex, no PCs, wrong E)': '/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/paper_figs/prs_signatures_oldpaper/gamma_associations.csv',
    'October (loop, sex+PCs, wrong E)': '/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/paper_figs/prs_signatures_october_no_E_withPCSSEX/gamma_associations.csv',
    'Current (vector, sex+PCs, correct E)': '/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/paper_figs/prs_signatures_newestE_PCs_Sex/gamma_associations.csv',
    'No PCs (vector, sex, correct E)': '/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/paper_figs/prs_signatures_nopPCS_E/gamma_associations.csv',
    'No Sex (vector, PCs, correct E)': '/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/paper_figs/prs_signatures_noSEX_withE_andPCS/gamma_associations.csv',
}

# Load all dataframes
dfs = {}
for name, path in files.items():
    try:
        df = pd.read_csv(path)
        dfs[name] = df
        print(f"✓ Loaded {name}: {len(df)} associations")
    except Exception as e:
        print(f"✗ Failed to load {name}: {e}")

# Create comparison dataframe
comparison_data = []
for name, df in dfs.items():
    # Filter out zero effects (Sig 20)
    df_filtered = df[df['effect'] != 0].copy()
    
    comparison_data.append({
        'version': name,
        'mean_abs_effect': df_filtered['effect'].abs().mean(),
        'median_abs_effect': df_filtered['effect'].abs().median(),
        'max_abs_effect': df_filtered['effect'].abs().max(),
        'std_abs_effect': df_filtered['effect'].abs().std(),
        'mean_sem': df_filtered['sem'].mean() if 'sem' in df_filtered.columns else np.nan,
        'median_sem': df_filtered['sem'].median() if 'sem' in df_filtered.columns else np.nan,
        'n_associations': len(df_filtered),
    })

comparison_df = pd.DataFrame(comparison_data)
print("\n" + "="*80)
print("SUMMARY STATISTICS BY VERSION")
print("="*80)
print(comparison_df.to_string(index=False))

# Merge all dataframes for direct comparison
print("\n" + "="*80)
print("DIRECT COMPARISON OF EFFECT SIZES")
print("="*80)

# Create merged dataframe with effects and z-scores from each version
march_df = dfs['March (loop, no sex, no PCs, wrong E)'].copy()
march_df = march_df.rename(columns={'effect': 'effect_march', 'z_score': 'z_score_march'})
march_df = march_df[['prs', 'signature', 'effect_march', 'z_score_march']]

october_df = dfs['October (loop, sex+PCs, wrong E)'].copy()
october_df = october_df.rename(columns={'effect': 'effect_october', 'z_score': 'z_score_october'})
october_df = october_df[['prs', 'signature', 'effect_october', 'z_score_october']]

current_df = dfs['Current (vector, sex+PCs, correct E)'].copy()
current_df = current_df.rename(columns={'effect': 'effect_current', 'z_score': 'z_score_current'})
current_df = current_df[['prs', 'signature', 'effect_current', 'z_score_current']]

no_pcs_df = dfs['No PCs (vector, sex, correct E)'].copy()
no_pcs_df = no_pcs_df.rename(columns={'effect': 'effect_no_pcs', 'z_score': 'z_score_no_pcs'})
no_pcs_df = no_pcs_df[['prs', 'signature', 'effect_no_pcs', 'z_score_no_pcs']]

no_sex_df = dfs['No Sex (vector, PCs, correct E)'].copy()
no_sex_df = no_sex_df.rename(columns={'effect': 'effect_no_sex', 'z_score': 'z_score_no_sex'})
no_sex_df = no_sex_df[['prs', 'signature', 'effect_no_sex', 'z_score_no_sex']]

# Merge
merged = march_df.merge(october_df, on=['prs', 'signature'], how='inner')
merged = merged.merge(current_df, on=['prs', 'signature'], how='inner')
merged = merged.merge(no_pcs_df, on=['prs', 'signature'], how='inner')
merged = merged.merge(no_sex_df, on=['prs', 'signature'], how='inner')

# Filter out zero effects
merged = merged[(merged['effect_march'] != 0) & (merged['effect_current'] != 0)]

# Calculate ratios
merged['ratio_march_to_current'] = merged['effect_march'].abs() / merged['effect_current'].abs()
merged['ratio_march_to_october'] = merged['effect_march'].abs() / merged['effect_october'].abs()
merged['ratio_october_to_current'] = merged['effect_october'].abs() / merged['effect_current'].abs()

print(f"\nTotal associations compared: {len(merged)}")
print(f"\nEffect Size Ratios:")
print(f"  March / Current: median={merged['ratio_march_to_current'].median():.2f}, mean={merged['ratio_march_to_current'].mean():.2f}")
print(f"  March / October: median={merged['ratio_march_to_october'].median():.2f}, mean={merged['ratio_march_to_october'].mean():.2f}")
print(f"  October / Current: median={merged['ratio_october_to_current'].median():.2f}, mean={merged['ratio_october_to_current'].mean():.2f}")

# Find largest differences
print("\n" + "="*80)
print("TOP 20 ASSOCIATIONS WITH LARGEST MARCH EFFECTS")
print("="*80)
top_march = merged.nlargest(20, 'effect_march', keep='all')
for idx, row in top_march.iterrows():
    print(f"{row['prs']} - {row['signature']}:")
    print(f"  March: {row['effect_march']:.6f}")
    print(f"  October: {row['effect_october']:.6f} (ratio: {row['ratio_march_to_october']:.2f}x)")
    print(f"  Current: {row['effect_current']:.6f} (ratio: {row['ratio_march_to_current']:.2f}x)")
    print(f"  No PCs: {row['effect_no_pcs']:.6f}")
    print(f"  No Sex: {row['effect_no_sex']:.6f}")
    print()

# Compare versions with/without sex and PCs
print("\n" + "="*80)
print("COMPARING EFFECTS: SEX vs NO SEX (both with PCs, correct E)")
print("="*80)
current_vs_no_sex = merged.copy()
current_vs_no_sex['ratio_current_to_no_sex'] = current_vs_no_sex['effect_current'].abs() / current_vs_no_sex['effect_no_sex'].abs()
print(f"  Current (with sex) / No Sex: median={current_vs_no_sex['ratio_current_to_no_sex'].median():.2f}, mean={current_vs_no_sex['ratio_current_to_no_sex'].mean():.2f}")

print("\n" + "="*80)
print("COMPARING EFFECTS: PCs vs NO PCs (both with sex, correct E)")
print("="*80)
current_vs_no_pcs = merged.copy()
current_vs_no_pcs['ratio_current_to_no_pcs'] = current_vs_no_pcs['effect_current'].abs() / current_vs_no_pcs['effect_no_pcs'].abs()
print(f"  Current (with PCs) / No PCs: median={current_vs_no_pcs['ratio_current_to_no_pcs'].median():.2f}, mean={current_vs_no_pcs['ratio_current_to_no_pcs'].mean():.2f}")

# Check if March effects are systematically larger
print("\n" + "="*80)
print("SYSTEMATIC DIFFERENCES")
print("="*80)
print(f"Associations where March effect > 10x October: {(merged['ratio_march_to_october'] > 10).sum()} / {len(merged)} ({(merged['ratio_march_to_october'] > 10).sum()/len(merged)*100:.1f}%)")
print(f"Associations where March effect > 10x Current: {(merged['ratio_march_to_current'] > 10).sum()} / {len(merged)} ({(merged['ratio_march_to_current'] > 10).sum()/len(merged)*100:.1f}%)")
print(f"Associations where March effect > 5x October: {(merged['ratio_march_to_october'] > 5).sum()} / {len(merged)} ({(merged['ratio_march_to_october'] > 5).sum()/len(merged)*100:.1f}%)")
print(f"Associations where March effect > 5x Current: {(merged['ratio_march_to_current'] > 5).sum()} / {len(merged)} ({(merged['ratio_march_to_current'] > 5).sum()/len(merged)*100:.1f}%)")

# Compare z-scores (effect / SEM) - this is what matters for significance
print("\n" + "="*80)
print("Z-SCORE COMPARISON (Effect / SEM) - This is what matters for significance!")
print("="*80)

# Filter to associations with valid z-scores
valid_z = merged[
    merged['z_score_march'].notna() & 
    merged['z_score_current'].notna() & 
    merged['z_score_october'].notna()
].copy()

if len(valid_z) > 0:
    print(f"\nAssociations with valid z-scores: {len(valid_z)}")
    
    # Calculate correlations between z-scores
    z_corr_march_current = valid_z['z_score_march'].corr(valid_z['z_score_current'])
    z_corr_march_october = valid_z['z_score_march'].corr(valid_z['z_score_october'])
    z_corr_october_current = valid_z['z_score_october'].corr(valid_z['z_score_current'])
    
    print(f"\nZ-score Correlations:")
    print(f"  March vs Current: r = {z_corr_march_current:.4f}")
    print(f"  March vs October: r = {z_corr_march_october:.4f}")
    print(f"  October vs Current: r = {z_corr_october_current:.4f}")
    
    # Calculate mean absolute differences
    z_diff_march_current = (valid_z['z_score_march'] - valid_z['z_score_current']).abs()
    z_diff_march_october = (valid_z['z_score_march'] - valid_z['z_score_october']).abs()
    z_diff_october_current = (valid_z['z_score_october'] - valid_z['z_score_current']).abs()
    
    print(f"\nMean Absolute Z-score Differences:")
    print(f"  |March - Current|: mean={z_diff_march_current.mean():.4f}, median={z_diff_march_current.median():.4f}")
    print(f"  |March - October|: mean={z_diff_march_october.mean():.4f}, median={z_diff_march_october.median():.4f}")
    print(f"  |October - Current|: mean={z_diff_october_current.mean():.4f}, median={z_diff_october_current.median():.4f}")
    
    # Summary statistics for z-scores
    print(f"\nZ-score Summary Statistics:")
    print(f"  March: mean={valid_z['z_score_march'].abs().mean():.2f}, median={valid_z['z_score_march'].abs().median():.2f}")
    print(f"  October: mean={valid_z['z_score_october'].abs().mean():.2f}, median={valid_z['z_score_october'].abs().median():.2f}")
    print(f"  Current: mean={valid_z['z_score_current'].abs().mean():.2f}, median={valid_z['z_score_current'].abs().median():.2f}")
    
    print(f"\n✓ Z-scores are {'VERY SIMILAR' if z_corr_march_current > 0.95 and z_diff_march_current.median() < 0.5 else 'DIFFERENT'} across versions")
    print(f"  This confirms that statistical significance patterns are consistent!")
else:
    print("  Warning: No valid z-scores found for comparison")

# Save comparison
output_path = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/paper_figs/gamma_version_comparison.csv')
merged.to_csv(output_path, index=False)
print(f"\n✓ Saved detailed comparison to: {output_path}")
