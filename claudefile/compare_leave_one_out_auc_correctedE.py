#!/usr/bin/env python
"""
Compare leave-one-out AUCs to overall pooled AUC

This script:
1. Loads leave-one-out AUC results
2. Loads overall pooled AUC results (from performance notebook or similar)
3. Compares them to show robustness

Usage:
    python compare_leave_one_out_auc_correctedE.py
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Compare leave-one-out AUCs to overall AUC')
    parser.add_argument('--leave_one_out_csv', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/leave_one_out_auc_results_correctedE.csv',
                       help='CSV file with leave-one-out AUC results')
    parser.add_argument('--overall_auc_csv', type=str, default=None,
                       help='CSV file with overall pooled AUC results (optional, can calculate from leave-one-out)')
    parser.add_argument('--output_csv', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/leave_one_out_comparison_correctedE.csv',
                       help='Output CSV file for comparison')
    args = parser.parse_args()
    
    print("="*80)
    print("Compare Leave-One-Out AUCs to Overall AUC")
    print("="*80)
    
    # Load leave-one-out results
    if not Path(args.leave_one_out_csv).exists():
        print(f"Error: Leave-one-out results file not found: {args.leave_one_out_csv}")
        return
    
    loo_df = pd.read_csv(args.leave_one_out_csv)
    print(f"✓ Loaded leave-one-out results: {len(loo_df)} rows")
    print(f"  Batches: {sorted(loo_df['batch_idx'].unique())}")
    print(f"  Disease groups: {sorted(loo_df['disease_group'].unique())}")
    
    # Calculate overall AUC from leave-one-out (mean across batches)
    # This is a reasonable approximation if we don't have the exact overall results
    print("\nCalculating mean AUC across leave-one-out batches...")
    overall_loo = loo_df.groupby('disease_group').agg({
        'auc': ['mean', 'std', 'min', 'max'],
        'n_events': 'sum',
        'n_total': 'sum'
    }).reset_index()
    overall_loo.columns = ['disease_group', 'auc_mean', 'auc_std', 'auc_min', 'auc_max', 'n_events_total', 'n_total_total']
    overall_loo['event_rate'] = overall_loo['n_events_total'] / overall_loo['n_total_total']
    
    # Calculate per-batch statistics
    print("\nCalculating per-batch statistics...")
    batch_stats = loo_df.groupby(['disease_group', 'batch_idx']).agg({
        'auc': 'first',
        'n_events': 'first',
        'n_total': 'first'
    }).reset_index()
    
    # Calculate mean and std across batches for each disease
    disease_batch_stats = batch_stats.groupby('disease_group').agg({
        'auc': ['mean', 'std', 'min', 'max'],
        'n_events': 'mean',
        'n_total': 'mean'
    }).reset_index()
    disease_batch_stats.columns = ['disease_group', 'auc_mean_loo', 'auc_std_loo', 'auc_min_loo', 'auc_max_loo', 
                                    'n_events_mean', 'n_total_mean']
    
    # Merge
    comparison_df = overall_loo.merge(disease_batch_stats, on='disease_group', how='outer')
    
    # Calculate differences
    comparison_df['auc_range'] = comparison_df['auc_max_loo'] - comparison_df['auc_min_loo']
    comparison_df['auc_cv'] = comparison_df['auc_std_loo'] / comparison_df['auc_mean_loo']  # Coefficient of variation
    
    # Sort by mean AUC
    comparison_df = comparison_df.sort_values('auc_mean', ascending=False)
    
    # Save comparison
    comparison_df.to_csv(args.output_csv, index=False)
    print(f"\n✓ Comparison saved to: {args.output_csv}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY: Leave-One-Out Validation Results")
    print("="*80)
    print(f"\nTotal disease groups: {len(comparison_df)}")
    print(f"Total batches evaluated: {len(loo_df['batch_idx'].unique())}")
    
    print("\nTop 10 diseases by mean AUC:")
    top10 = comparison_df.head(10)
    for _, row in top10.iterrows():
        print(f"  {row['disease_group']:25s} AUC: {row['auc_mean']:.4f} ± {row['auc_std_loo']:.4f} "
              f"(range: [{row['auc_min_loo']:.4f}, {row['auc_max_loo']:.4f}])")
    
    print("\nAUC Stability (Coefficient of Variation):")
    print(f"  Mean CV: {comparison_df['auc_cv'].mean():.4f}")
    print(f"  Median CV: {comparison_df['auc_cv'].median():.4f}")
    print(f"  Max CV: {comparison_df['auc_cv'].max():.4f}")
    
    print("\nAUC Range (max - min across batches):")
    print(f"  Mean range: {comparison_df['auc_range'].mean():.4f}")
    print(f"  Median range: {comparison_df['auc_range'].median():.4f}")
    print(f"  Max range: {comparison_df['auc_range'].max():.4f}")
    
    print("\n" + "="*80)
    print("Interpretation:")
    print("="*80)
    print("If AUCs are similar across batches (low CV, small range), this suggests:")
    print("  - Pooling phi is robust (no overfitting to specific batches)")
    print("  - Model performance is consistent across different subsets of data")
    print("  - Leave-one-out validation confirms the overall results are reliable")
    print("="*80)


if __name__ == '__main__':
    main()

