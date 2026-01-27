#!/usr/bin/env python3
"""
Compare fixed gamma/kappa AUC results with original results for full 400k dataset.

This script:
1. Loads fixed gamma/kappa results (static 10yr and dynamic 1yr)
2. Loads original results (static 10yr and washout 0yr)
3. Compares them side-by-side
4. Saves comparison CSV

Usage:
    python compare_fixedgk_vs_original_full_400k.py
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Compare fixed gamma/kappa vs original AUC results')
    parser.add_argument('--fixedgk_static', type=str,
                       default='/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/fixedgk_static_10yr_auc_results.csv',
                       help='Path to fixed gamma/kappa static 10yr results')
    parser.add_argument('--fixedgk_dynamic', type=str,
                       default='/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/fixedgk_dynamic_1yr_auc_results.csv',
                       help='Path to fixed gamma/kappa dynamic 1yr results')
    parser.add_argument('--original_static', type=str,
                       default='/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/time_horizons/pooled_retrospective/static_10yr_results.csv',
                       help='Path to original static 10yr results')
    parser.add_argument('--original_dynamic', type=str,
                       default='/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/washout/pooled_retrospective/washout_0yr_results.csv',
                       help='Path to original dynamic 1yr results (washout 0)')
    parser.add_argument('--output_dir', type=str,
                       default='/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/',
                       help='Output directory for comparison results')
    
    args = parser.parse_args()
    
    print("="*80)
    print("COMPARING FIXED GAMMA/KAPPA VS ORIGINAL RESULTS - FULL 400K")
    print("="*80)
    
    # Load results
    print("\nLoading results...")
    fixedgk_static = pd.read_csv(args.fixedgk_static)
    fixedgk_dynamic = pd.read_csv(args.fixedgk_dynamic)
    original_static = pd.read_csv(args.original_static)
    original_dynamic = pd.read_csv(args.original_dynamic)
    
    print(f"Fixed GK static: {len(fixedgk_static)} diseases")
    print(f"Fixed GK dynamic: {len(fixedgk_dynamic)} diseases")
    print(f"Original static: {len(original_static)} diseases")
    print(f"Original dynamic: {len(original_dynamic)} diseases")
    
    # ============================================================================
    # STATIC 10-YEAR COMPARISON
    # ============================================================================
    print("\n" + "="*80)
    print("STATIC 10-YEAR AUC COMPARISON")
    print("="*80)
    
    # Prepare dataframes with consistent column names
    original_static_clean = original_static[['Disease', 'AUC', 'CI_lower', 'CI_upper', 'N_Events']].copy()
    original_static_clean.columns = ['disease', 'auc_original', 'ci_lower_original', 'ci_upper_original', 'n_events_original']
    
    fixedgk_static_clean = fixedgk_static[['disease', 'auc', 'ci_lower', 'ci_upper', 'n_events']].copy()
    fixedgk_static_clean.columns = ['disease', 'auc_fixedgk', 'ci_lower_fixedgk', 'ci_upper_fixedgk', 'n_events_fixedgk']
    
    # Merge on disease name
    static_merged = pd.merge(
        original_static_clean,
        fixedgk_static_clean,
        on='disease',
        how='outer'
    )
    
    static_merged['auc_diff'] = static_merged['auc_fixedgk'] - static_merged['auc_original']
    static_merged['auc_diff_abs'] = static_merged['auc_diff'].abs()
    
    # Sort by absolute difference
    static_merged = static_merged.sort_values('auc_diff_abs', ascending=False)
    
    print("\nTop 10 diseases by absolute AUC difference:")
    print("-"*80)
    print(f"{'Disease':<30} {'Original AUC':<15} {'Fixed GK AUC':<15} {'Difference':<15}")
    print("-"*80)
    for _, row in static_merged.head(10).iterrows():
        orig_auc = row['auc_original']
        new_auc = row['auc_fixedgk']
        diff = row['auc_diff']
        disease = row['disease']
        if pd.notna(orig_auc) and pd.notna(new_auc):
            print(f"{disease:<30} {orig_auc:>8.4f}        {new_auc:>8.4f}        {diff:>+8.4f}")
    
    # Summary statistics
    valid_comparisons = static_merged.dropna(subset=['auc_original', 'auc_fixedgk'])
    print(f"\nSummary Statistics:")
    print(f"  Mean absolute difference: {valid_comparisons['auc_diff_abs'].mean():.6f}")
    print(f"  Std absolute difference: {valid_comparisons['auc_diff_abs'].std():.6f}")
    print(f"  Min difference: {valid_comparisons['auc_diff'].min():.6f}")
    print(f"  Max difference: {valid_comparisons['auc_diff'].max():.6f}")
    print(f"  Diseases with |diff| > 0.02: {(valid_comparisons['auc_diff_abs'] > 0.02).sum()}")
    print(f"  Diseases with |diff| > 0.03: {(valid_comparisons['auc_diff_abs'] > 0.03).sum()}")
    print(f"  Diseases with |diff| > 0.04: {(valid_comparisons['auc_diff_abs'] > 0.04).sum()}")
    
    # Save comparison
    static_output = os.path.join(args.output_dir, 'fixedgk_vs_original_static_10yr_comparison.csv')
    static_merged.to_csv(static_output, index=False)
    print(f"\n✓ Saved static 10-year comparison to: {static_output}")
    
    # ============================================================================
    # DYNAMIC 1-YEAR COMPARISON
    # ============================================================================
    print("\n" + "="*80)
    print("DYNAMIC 1-YEAR AUC COMPARISON")
    print("="*80)
    
    # Prepare dataframes with consistent column names
    original_dynamic_clean = original_dynamic[['Disease', 'AUC', 'CI_lower', 'CI_upper', 'N_Events']].copy()
    original_dynamic_clean.columns = ['disease', 'auc_original', 'ci_lower_original', 'ci_upper_original', 'n_events_original']
    
    fixedgk_dynamic_clean = fixedgk_dynamic[['disease', 'auc', 'ci_lower', 'ci_upper', 'n_events']].copy()
    fixedgk_dynamic_clean.columns = ['disease', 'auc_fixedgk', 'ci_lower_fixedgk', 'ci_upper_fixedgk', 'n_events_fixedgk']
    
    # Merge on disease name
    dynamic_merged = pd.merge(
        original_dynamic_clean,
        fixedgk_dynamic_clean,
        on='disease',
        how='outer'
    )
    
    dynamic_merged['auc_diff'] = dynamic_merged['auc_fixedgk'] - dynamic_merged['auc_original']
    dynamic_merged['auc_diff_abs'] = dynamic_merged['auc_diff'].abs()
    
    # Sort by absolute difference
    dynamic_merged = dynamic_merged.sort_values('auc_diff_abs', ascending=False)
    
    print("\nTop 10 diseases by absolute AUC difference:")
    print("-"*80)
    print(f"{'Disease':<30} {'Original AUC':<15} {'Fixed GK AUC':<15} {'Difference':<15}")
    print("-"*80)
    for _, row in dynamic_merged.head(10).iterrows():
        orig_auc = row['auc_original']
        new_auc = row['auc_fixedgk']
        diff = row['auc_diff']
        disease = row['disease']
        if pd.notna(orig_auc) and pd.notna(new_auc):
            print(f"{disease:<30} {orig_auc:>8.4f}        {new_auc:>8.4f}        {diff:>+8.4f}")
    
    # Summary statistics
    valid_comparisons_dyn = dynamic_merged.dropna(subset=['auc_original', 'auc_fixedgk'])
    print(f"\nSummary Statistics:")
    print(f"  Mean absolute difference: {valid_comparisons_dyn['auc_diff_abs'].mean():.6f}")
    print(f"  Std absolute difference: {valid_comparisons_dyn['auc_diff_abs'].std():.6f}")
    print(f"  Min difference: {valid_comparisons_dyn['auc_diff'].min():.6f}")
    print(f"  Max difference: {valid_comparisons_dyn['auc_diff'].max():.6f}")
    print(f"  Diseases with |diff| > 0.02: {(valid_comparisons_dyn['auc_diff_abs'] > 0.02).sum()}")
    print(f"  Diseases with |diff| > 0.03: {(valid_comparisons_dyn['auc_diff_abs'] > 0.03).sum()}")
    print(f"  Diseases with |diff| > 0.04: {(valid_comparisons_dyn['auc_diff_abs'] > 0.04).sum()}")
    
    # Save comparison
    dynamic_output = os.path.join(args.output_dir, 'fixedgk_vs_original_dynamic_1yr_comparison.csv')
    dynamic_merged.to_csv(dynamic_output, index=False)
    print(f"\n✓ Saved dynamic 1-year comparison to: {dynamic_output}")
    
    # ============================================================================
    # WIDE FORMAT: DISEASE BY METHOD (28 x 4)
    # ============================================================================
    print("\n" + "="*80)
    print("CREATING WIDE FORMAT TABLE")
    print("="*80)
    
    # Prepare dataframes with just AUC columns for wide format
    original_static_auc = original_static[['Disease', 'AUC']].copy()
    original_static_auc.columns = ['disease', 'original_static_10yr']
    
    fixedgk_static_auc = fixedgk_static[['disease', 'auc']].copy()
    fixedgk_static_auc.columns = ['disease', 'fixedgk_static_10yr']
    
    original_dynamic_auc = original_dynamic[['Disease', 'AUC']].copy()
    original_dynamic_auc.columns = ['disease', 'original_dynamic_1yr']
    
    fixedgk_dynamic_auc = fixedgk_dynamic[['disease', 'auc']].copy()
    fixedgk_dynamic_auc.columns = ['disease', 'fixedgk_dynamic_1yr']
    
    # Merge all four together
    wide_format = original_static_auc.merge(
        fixedgk_static_auc, on='disease', how='outer'
    ).merge(
        original_dynamic_auc, on='disease', how='outer'
    ).merge(
        fixedgk_dynamic_auc, on='disease', how='outer'
    )
    
    # Sort by disease name for consistent ordering
    wide_format = wide_format.sort_values('disease').reset_index(drop=True)
    
    # Reorder columns: disease, then static 10yr (original, fixedgk), then dynamic 1yr (original, fixedgk)
    wide_format = wide_format[['disease', 'original_static_10yr', 'fixedgk_static_10yr', 
                               'original_dynamic_1yr', 'fixedgk_dynamic_1yr']]
    
    # Save wide format
    wide_output = os.path.join(args.output_dir, 'fixedgk_vs_original_wide_format.csv')
    wide_format.to_csv(wide_output, index=False)
    print(f"\n✓ Saved wide format table to: {wide_output}")
    print(f"   Shape: {wide_format.shape[0]} diseases x {wide_format.shape[1]} columns")
    print(f"   Columns: disease | original_static_10yr | fixedgk_static_10yr | original_dynamic_1yr | fixedgk_dynamic_1yr")
    
    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Static 10-year: {len(valid_comparisons)} diseases compared")
    print(f"Dynamic 1-year: {len(valid_comparisons_dyn)} diseases compared")
    print(f"\nResults saved to:")
    print(f"  - {static_output}")
    print(f"  - {dynamic_output}")
    print(f"  - {wide_output} (wide format: disease x method)")
    print("="*80)
    print("COMPLETED")
    print("="*80)


if __name__ == '__main__':
    main()
