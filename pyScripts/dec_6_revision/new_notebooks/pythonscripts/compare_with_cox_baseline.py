#!/usr/bin/env python3
"""
Compare Aladynoulli predictions with Cox baseline model (age + sex only, no Noulli).

The Cox baseline was trained on patients 20001-30000 and tested on patients 0-10000
using a 10-year follow-up period.

Usage in notebook:
    %run compare_with_cox_baseline.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# 1. LOAD COX BASELINE RESULTS
# =============================================================================

print("="*100)
print("ALADYNOULLI vs COX BASELINE (AGE + SEX ONLY, NO NOULLI)")
print("="*100)
print("Cox model: 10-year follow-up (trained on 20001-30000, tested on 0-10000)")
print("Aladynoulli: Static 10-year (1-year score for 10-year outcome)")
print("="*100)

print("\nLoading Cox baseline results...")

# Try multiple possible paths
cox_file = None
possible_paths = [
    '/Users/sarahurbut/Library/CloudStorage/Dropbox/auc_results_cox_20000_30000train_0_10000test_1121.csv',
    '/Users/sarahurbut/aladynoulli2/pyScripts/auc_results_cox_20000_30000train_0_10000test.csv'
]

for path in possible_paths:
    if Path(path).exists():
        cox_file = path
        break

if cox_file is None:
    print("⚠️  ERROR: Could not find Cox baseline file at any of these paths:")
    for path in possible_paths:
        print(f"   - {path}")
    exit(1)

try:
    cox_df = pd.read_csv(cox_file)
    print(f"✓ Loaded Cox baseline results: {len(cox_df)} diseases")
    print(f"  Columns: {list(cox_df.columns)}")
    
    # Standardize column names
    if 'disease_group' in cox_df.columns:
        cox_df = cox_df.rename(columns={'disease_group': 'Disease'})
    if 'auc' in cox_df.columns:
        cox_df = cox_df.rename(columns={'auc': 'Cox_AUC'})
    
    print(f"\nFirst few rows:")
    print(cox_df.head())
    
except FileNotFoundError:
    print(f"⚠️  ERROR: Cox baseline file not found at: {cox_file}")
    print("   Please check the file path.")
    cox_df = None
except Exception as e:
    print(f"⚠️  ERROR loading Cox baseline: {e}")
    cox_df = None

if cox_df is None:
    print("\nCannot proceed without Cox baseline results.")
    exit(1)

# =============================================================================
# 2. LOAD ALADYNOULLI RESULTS (STATIC 10-YEAR, MATCHING COX HORIZON)
# =============================================================================

print("\nLoading Aladynoulli static 10-year results...")
print("Note: Cox model uses 10-year follow-up, comparing with static 10-year (1-year score for 10-year outcome)")

# Load static 10-year results (matching Cox's 10-year follow-up)
aladynoulli_static10yr_file = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/time_horizons/pooled_retrospective/static_10yr_results.csv')

try:
    aladynoulli_10yr = pd.read_csv(aladynoulli_static10yr_file)
    print(f"✓ Loaded Aladynoulli static 10-year results: {len(aladynoulli_10yr)} diseases")
    
    # Standardize column names
    if 'AUC' in aladynoulli_10yr.columns:
        aladynoulli_10yr = aladynoulli_10yr.rename(columns={'AUC': 'Aladynoulli_AUC'})
    
except FileNotFoundError:
    print(f"⚠️  ERROR: Aladynoulli results file not found at: {aladynoulli_10yr_file}")
    aladynoulli_10yr = None
except Exception as e:
    print(f"⚠️  ERROR loading Aladynoulli results: {e}")
    aladynoulli_10yr = None

if aladynoulli_10yr is None:
    print("\nCannot proceed without Aladynoulli results.")
    exit(1)

# =============================================================================
# 3. MERGE AND COMPARE
# =============================================================================

print("\nMerging results...")

# Merge on Disease name
comparison = cox_df[['Disease', 'Cox_AUC']].merge(
    aladynoulli_10yr[['Disease', 'Aladynoulli_AUC', 'CI_lower', 'CI_upper', 'N_Events', 'Event_Rate']],
    on='Disease',
    how='outer'
)

# Calculate improvement
comparison['Improvement'] = comparison['Aladynoulli_AUC'] - comparison['Cox_AUC']
comparison['Percent_Improvement'] = (comparison['Improvement'] / comparison['Cox_AUC'] * 100)

# Sort by improvement
comparison = comparison.sort_values('Improvement', ascending=False, na_position='last')

print(f"✓ Merged results: {len(comparison)} diseases")
print(f"  Diseases in both: {comparison[comparison[['Cox_AUC', 'Aladynoulli_AUC']].notna().all(axis=1)].shape[0]}")
print(f"  Only in Cox: {comparison[comparison['Cox_AUC'].notna() & comparison['Aladynoulli_AUC'].isna()].shape[0]}")
print(f"  Only in Aladynoulli: {comparison[comparison['Cox_AUC'].isna() & comparison['Aladynoulli_AUC'].notna()].shape[0]}")

# =============================================================================
# 4. DISPLAY RESULTS
# =============================================================================

print("\n" + "=" * 100)
print("ALADYNOULLI vs COX BASELINE: STATIC 10-YEAR PREDICTIONS")
print("=" * 100)

# Filter to diseases present in both
valid_comparison = comparison[comparison[['Cox_AUC', 'Aladynoulli_AUC']].notna().all(axis=1)].copy()

if len(valid_comparison) > 0:
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    print(f"\n{'Disease':<25} {'Cox_AUC':>10} {'Aladynoulli_AUC':>15} {'Improvement':>12} {'Percent':>10} {'N_Events':>10}")
    print("-" * 100)
    
    for idx, row in valid_comparison.iterrows():
        disease = row['Disease']
        cox_auc = row['Cox_AUC']
        ala_auc = row['Aladynoulli_AUC']
        improvement = row['Improvement']
        pct = row['Percent_Improvement']
        n_events = row.get('N_Events', np.nan)
        
        print(f"{disease:<25} {cox_auc:>10.4f} {ala_auc:>15.4f} {improvement:>12.4f} {pct:>9.1f}% {n_events:>10.0f}")

# =============================================================================
# 5. SUMMARY STATISTICS
# =============================================================================

print("\n" + "=" * 100)
print("SUMMARY STATISTICS")
print("=" * 100)

if len(valid_comparison) > 0:
    print(f"\nTotal diseases compared: {len(valid_comparison)}")
    print(f"\nAUC Statistics:")
    print(f"  Cox baseline mean:     {valid_comparison['Cox_AUC'].mean():.4f}")
    print(f"  Cox baseline median:   {valid_comparison['Cox_AUC'].median():.4f}")
    print(f"  Aladynoulli mean:      {valid_comparison['Aladynoulli_AUC'].mean():.4f}")
    print(f"  Aladynoulli median:    {valid_comparison['Aladynoulli_AUC'].median():.4f}")
    
    print(f"\nImprovement Statistics:")
    print(f"  Mean improvement:       {valid_comparison['Improvement'].mean():.4f}")
    print(f"  Median improvement:    {valid_comparison['Improvement'].median():.4f}")
    print(f"  Mean % improvement:    {valid_comparison['Percent_Improvement'].mean():.1f}%")
    
    wins = valid_comparison[valid_comparison['Improvement'] > 0]
    losses = valid_comparison[valid_comparison['Improvement'] < 0]
    ties = valid_comparison[valid_comparison['Improvement'] == 0]
    
    print(f"\nWin/Loss Summary:")
    print(f"  Aladynoulli wins:       {len(wins)}/{len(valid_comparison)} ({len(wins)/len(valid_comparison)*100:.1f}%)")
    print(f"  Aladynoulli losses:     {len(losses)}/{len(valid_comparison)} ({len(losses)/len(valid_comparison)*100:.1f}%)")
    print(f"  Ties:                   {len(ties)}/{len(valid_comparison)} ({len(ties)/len(valid_comparison)*100:.1f}%)")
    
    if len(wins) > 0:
        print(f"\n  Average win margin:     {wins['Improvement'].mean():.4f}")
        print(f"  Largest win:            {wins['Improvement'].max():.4f} ({wins.loc[wins['Improvement'].idxmax(), 'Disease']})")
    
    if len(losses) > 0:
        print(f"\n  Average loss margin:    {losses['Improvement'].mean():.4f}")
        print(f"  Largest loss:           {losses['Improvement'].min():.4f} ({losses.loc[losses['Improvement'].idxmin(), 'Disease']})")
    
    # Significant improvements (>0.05 AUC)
    sig_improvements = valid_comparison[valid_comparison['Improvement'] > 0.05]
    print(f"\nSignificant Improvements (>0.05 AUC):")
    print(f"  Count: {len(sig_improvements)}/{len(valid_comparison)} ({len(sig_improvements)/len(valid_comparison)*100:.1f}%)")
    if len(sig_improvements) > 0:
        print(f"\n  Top improvements:")
        for idx, row in sig_improvements.head(10).iterrows():
            print(f"    {row['Disease']:<25} +{row['Improvement']:.4f} ({row['Percent_Improvement']:.1f}%)")

# =============================================================================
# 6. TOP IMPROVEMENTS AND LOSSES
# =============================================================================

if len(valid_comparison) > 0:
    print("\n" + "=" * 100)
    print("TOP 10 BIGGEST IMPROVEMENTS")
    print("=" * 100)
    
    top10 = valid_comparison.nlargest(10, 'Improvement')
    for i, (idx, row) in enumerate(top10.iterrows(), 1):
        print(f"\n{i}. {row['Disease']}")
        print(f"   Cox baseline:    {row['Cox_AUC']:.4f}")
        print(f"   Aladynoulli:     {row['Aladynoulli_AUC']:.4f}")
        print(f"   Improvement:     +{row['Improvement']:.4f} ({row['Percent_Improvement']:.1f}% better)")
        if not pd.isna(row.get('N_Events')):
            print(f"   Events:          {row['N_Events']:.0f}")
    
    losses_only = valid_comparison[valid_comparison['Improvement'] < 0]
    if len(losses_only) > 0:
        print("\n" + "=" * 100)
        print("DISEASES WHERE COX BASELINE IS BETTER")
        print("=" * 100)
        
        top_losses = losses_only.nsmallest(min(10, len(losses_only)), 'Improvement')
        for i, (idx, row) in enumerate(top_losses.iterrows(), 1):
            print(f"\n{i}. {row['Disease']}")
            print(f"   Cox baseline:    {row['Cox_AUC']:.4f}")
            print(f"   Aladynoulli:     {row['Aladynoulli_AUC']:.4f}")
            print(f"   Difference:     {row['Improvement']:.4f} ({row['Percent_Improvement']:.1f}% worse)")

# =============================================================================
# 7. SAVE RESULTS
# =============================================================================

print("\n" + "=" * 100)
print("SAVING RESULTS")
print("=" * 100)

output_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/comparisons/pooled_retrospective')
output_dir.mkdir(parents=True, exist_ok=True)

# Save full comparison
comparison.to_csv(output_dir / 'cox_baseline_comparison_static10yr_full.csv', index=False)
print(f"Full comparison saved to: {output_dir / 'cox_baseline_comparison_static10yr_full.csv'}")

# Save wins only
if len(valid_comparison) > 0:
    wins = valid_comparison[valid_comparison['Improvement'] > 0]
    if len(wins) > 0:
        wins.to_csv(output_dir / 'cox_baseline_comparison_static10yr_wins.csv', index=False)
        print(f"Wins saved to: {output_dir / 'cox_baseline_comparison_static10yr_wins.csv'}")

print("\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)

# Make variables available
print("\nAvailable variables:")
print("  - comparison: Full comparison DataFrame")
if len(valid_comparison) > 0:
    print("  - valid_comparison: Diseases present in both Cox and Aladynoulli")
    wins = valid_comparison[valid_comparison['Improvement'] > 0]
    if len(wins) > 0:
        print("  - wins: Diseases where Aladynoulli wins")

