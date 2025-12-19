#!/usr/bin/env python3
"""
Compare AUC between with-G (genetics + sex + PCs) and 0G (no genetic/covariate effects) versions.

This script:
- Loads pi predictions from first 5 batches from both directories:
  - enrollment_predictions_fixedphi_correctedE_vectorized_withLR (with G)
  - enrollment_predictions_fixedphi_correctedE_vectorized_0G (0G, no G effects)
- Evaluates:
  - 1-year predictions with 0 washout (start_offset=0, follow_up_duration_years=1)
  - 10-year predictions (follow_up_duration_years=10)
- Compares AUC between the two versions

Usage:
    python compare_withG_vs_0G_auc.py --n_batches 5 --n_bootstraps 100
"""

import argparse
import sys
import os
import glob
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# Add path for imports
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/')
from evaluatetdccode import evaluate_major_diseases_wsex_with_bootstrap_dynamic_1year_different_start_end_numeric_sex
from fig5utils import evaluate_major_diseases_wsex_with_bootstrap_from_pi

# Load essentials (disease names, etc.)
def load_essentials():
    """Load model essentials including disease names"""
    essentials_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/model_essentials.pt'
    essentials = torch.load(essentials_path, weights_only=False)
    return essentials

def get_start_index(filename):
    """Extract start index from filename for sorting"""
    basename = os.path.basename(filename)
    parts = basename.replace('.pt', '').split('_')
    numeric_parts = [p for p in parts if p.isdigit()]
    if len(numeric_parts) >= 2:
        return int(numeric_parts[0])  # First number is start index
    raise ValueError(f"Could not extract start index from {filename}")

def load_pi_batches(directory, n_batches=5):
    """
    Load pi predictions from first n_batches in a directory.
    
    Args:
        directory: Path to directory containing pi batch files
        n_batches: Number of batches to load
        
    Returns:
        Concatenated pi tensor
    """
    print(f"\nLoading pi batches from: {directory}")
    
    # Find all pi batch files
    pattern = os.path.join(directory, "pi_enroll_fixedphi_sex_*_*.pt")
    pi_files = glob.glob(pattern)
    
    if len(pi_files) == 0:
        raise ValueError(f"No pi batch files found in {directory}")
    
    # Sort by start index numerically
    pi_files = sorted(pi_files, key=get_start_index)
    
    # Take first n_batches
    pi_files = pi_files[:n_batches]
    print(f"Found {len(pi_files)} batch files (loading first {n_batches} in order)")
    
    # Load and concatenate
    pi_batches = []
    for i, pi_file in enumerate(pi_files):
        print(f"  Loading batch {i+1}/{len(pi_files)}: {os.path.basename(pi_file)}")
        pi_batch = torch.load(pi_file, weights_only=False)
        pi_batches.append(pi_batch)
    
    pi_full = torch.cat(pi_batches, dim=0)
    print(f"  Concatenated shape: {pi_full.shape}")
    
    return pi_full

def main():
    parser = argparse.ArgumentParser(description='Compare AUC between with-G and 0G predictions')
    parser.add_argument('--n_batches', type=int, default=5,
                       help='Number of batches to compare (default: 5)')
    parser.add_argument('--n_bootstraps', type=int, default=100,
                       help='Number of bootstrap iterations for CI (default: 100)')
    parser.add_argument('--withG_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_correctedE_vectorized/',
                       help='Directory with with-G predictions')
    parser.add_argument('--zeroG_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_correctedE_vectorized_0G/',
                       help='Directory with 0G predictions')
    parser.add_argument('--output_dir', type=str,
                       default='/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/withG_vs_0G_comparison/',
                       help='Output directory for results')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("COMPARING WITH-G vs 0G PREDICTIONS")
    print("="*80)
    print(f"With-G directory: {args.withG_dir}")
    print(f"0G directory: {args.zeroG_dir}")
    print(f"Number of batches: {args.n_batches}")
    print(f"Bootstrap iterations: {args.n_bootstraps}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # Load pi predictions from both directories
    print("\n" + "="*80)
    print("LOADING PREDICTIONS")
    print("="*80)
    
    pi_withG = load_pi_batches(args.withG_dir, n_batches=args.n_batches)
    pi_0G = load_pi_batches(args.zeroG_dir, n_batches=args.n_batches)
    
    # Verify shapes match
    if pi_withG.shape != pi_0G.shape:
        raise ValueError(f"Shape mismatch! With-G: {pi_withG.shape}, 0G: {pi_0G.shape}")
    
    print(f"\n✓ Both pi tensors have shape: {pi_withG.shape}")
    
    # Load other required data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    Y_full = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/Y_tensor.pt', weights_only=False)
    E_full = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/E_enrollment_full.pt', weights_only=False)
    pce_df_full = pd.read_csv('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pce_prevent_full.csv')
    essentials = load_essentials()
    disease_names = essentials['disease_names']
    
    # Convert Sex column to numeric if needed
    if 'Sex' in pce_df_full.columns and pce_df_full['Sex'].dtype == 'object':
        pce_df_full['sex'] = pce_df_full['Sex'].map({'Female': 0, 'Male': 1}).astype(int)
    elif 'sex' not in pce_df_full.columns:
        raise ValueError("Need 'Sex' or 'sex' column in pce_df")
    
    # Subset to match pi size
    N = pi_withG.shape[0]
    print(f"\nSubsetting data to first {N} patients...")
    Y_subset = Y_full[:N]
    E_subset = E_full[:N]
    pce_df_subset = pce_df_full.iloc[:N].reset_index(drop=True)
    
    print(f"Y_subset: {Y_subset.shape}, E_subset: {E_subset.shape}, pce_df_subset: {len(pce_df_subset)}")
    
    # Verify sizes match
    if not (N == Y_subset.shape[0] == E_subset.shape[0] == len(pce_df_subset)):
        raise ValueError(f"Size mismatch! pi: {N}, Y: {Y_subset.shape[0]}, E: {E_subset.shape[0]}, pce_df: {len(pce_df_subset)}")
    
    # Evaluate 1-year predictions (0 washout)
    print("\n" + "="*80)
    print("EVALUATING 1-YEAR PREDICTIONS (0 WASHOUT)")
    print("="*80)
    
    print("\nEvaluating with-G version...")
    results_1yr_withG = evaluate_major_diseases_wsex_with_bootstrap_dynamic_1year_different_start_end_numeric_sex(
        pi=pi_withG,
        Y_100k=Y_subset,
        E_100k=E_subset,
        disease_names=disease_names,
        pce_df=pce_df_subset,
        n_bootstraps=args.n_bootstraps,
        follow_up_duration_years=1,
        start_offset=0
    )
    
    print("\nEvaluating 0G version...")
    results_1yr_0G = evaluate_major_diseases_wsex_with_bootstrap_dynamic_1year_different_start_end_numeric_sex(
        pi=pi_0G,
        Y_100k=Y_subset,
        E_100k=E_subset,
        disease_names=disease_names,
        pce_df=pce_df_subset,
        n_bootstraps=args.n_bootstraps,
        follow_up_duration_years=1,
        start_offset=0
    )
    
    # Evaluate 10-year predictions
    print("\n" + "="*80)
    print("EVALUATING 10-YEAR PREDICTIONS")
    print("="*80)
    
    print("\nEvaluating with-G version...")
    results_10yr_withG = evaluate_major_diseases_wsex_with_bootstrap_from_pi(
        pi=pi_withG,
        Y_100k=Y_subset,
        E_100k=E_subset,
        disease_names=disease_names,
        pce_df=pce_df_subset,
        n_bootstraps=args.n_bootstraps,
        follow_up_duration_years=10
    )
    
    print("\nEvaluating 0G version...")
    results_10yr_0G = evaluate_major_diseases_wsex_with_bootstrap_from_pi(
        pi=pi_0G,
        Y_100k=Y_subset,
        E_100k=E_subset,
        disease_names=disease_names,
        pce_df=pce_df_subset,
        n_bootstraps=args.n_bootstraps,
        follow_up_duration_years=10
    )
    
    # Create comparison DataFrames
    print("\n" + "="*80)
    print("CREATING COMPARISON TABLES")
    print("="*80)
    
    # Get all diseases from both results
    all_diseases_1yr = set(results_1yr_withG.keys()) | set(results_1yr_0G.keys())
    all_diseases_10yr = set(results_10yr_withG.keys()) | set(results_10yr_0G.keys())
    
    # 1-year comparison
    comparison_1yr = pd.DataFrame(index=sorted(all_diseases_1yr))
    comparison_1yr['AUC_withG'] = [results_1yr_withG.get(d, {}).get('auc', np.nan) for d in comparison_1yr.index]
    comparison_1yr['CI_lower_withG'] = [results_1yr_withG.get(d, {}).get('ci_lower', np.nan) for d in comparison_1yr.index]
    comparison_1yr['CI_upper_withG'] = [results_1yr_withG.get(d, {}).get('ci_upper', np.nan) for d in comparison_1yr.index]
    comparison_1yr['AUC_0G'] = [results_1yr_0G.get(d, {}).get('auc', np.nan) for d in comparison_1yr.index]
    comparison_1yr['CI_lower_0G'] = [results_1yr_0G.get(d, {}).get('ci_lower', np.nan) for d in comparison_1yr.index]
    comparison_1yr['CI_upper_0G'] = [results_1yr_0G.get(d, {}).get('ci_upper', np.nan) for d in comparison_1yr.index]
    comparison_1yr['AUC_diff'] = comparison_1yr['AUC_withG'] - comparison_1yr['AUC_0G']
    comparison_1yr['N_events'] = [results_1yr_withG.get(d, {}).get('n_events', np.nan) for d in comparison_1yr.index]
    
    # 10-year comparison
    comparison_10yr = pd.DataFrame(index=sorted(all_diseases_10yr))
    comparison_10yr['AUC_withG'] = [results_10yr_withG.get(d, {}).get('auc', np.nan) for d in comparison_10yr.index]
    comparison_10yr['CI_lower_withG'] = [results_10yr_withG.get(d, {}).get('ci_lower', np.nan) for d in comparison_10yr.index]
    comparison_10yr['CI_upper_withG'] = [results_10yr_withG.get(d, {}).get('ci_upper', np.nan) for d in comparison_10yr.index]
    comparison_10yr['AUC_0G'] = [results_10yr_0G.get(d, {}).get('auc', np.nan) for d in comparison_10yr.index]
    comparison_10yr['CI_lower_0G'] = [results_10yr_0G.get(d, {}).get('ci_lower', np.nan) for d in comparison_10yr.index]
    comparison_10yr['CI_upper_0G'] = [results_10yr_0G.get(d, {}).get('ci_upper', np.nan) for d in comparison_10yr.index]
    comparison_10yr['AUC_diff'] = comparison_10yr['AUC_withG'] - comparison_10yr['AUC_0G']
    comparison_10yr['N_events'] = [results_10yr_withG.get(d, {}).get('n_events', np.nan) for d in comparison_10yr.index]
    
    # Sort by absolute AUC difference
    comparison_1yr = comparison_1yr.sort_values('AUC_diff', key=abs, ascending=False)
    comparison_10yr = comparison_10yr.sort_values('AUC_diff', key=abs, ascending=False)
    
    # Save results
    print("\nSaving results...")
    comparison_1yr.to_csv(output_dir / 'comparison_1yr_0washout.csv')
    comparison_10yr.to_csv(output_dir / 'comparison_10yr.csv')
    
    print(f"✓ Saved 1-year comparison to: {output_dir / 'comparison_1yr_0washout.csv'}")
    print(f"✓ Saved 10-year comparison to: {output_dir / 'comparison_10yr.csv'}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\n1-YEAR PREDICTIONS (0 WASHOUT):")
    print("-" * 80)
    print(f"{'Disease':<30} {'With-G AUC':<20} {'0G AUC':<20} {'Difference':<15}")
    print("-" * 80)
    for disease in comparison_1yr.index[:10]:  # Top 10 by absolute difference
        auc_withG = comparison_1yr.loc[disease, 'AUC_withG']
        auc_0G = comparison_1yr.loc[disease, 'AUC_0G']
        diff = comparison_1yr.loc[disease, 'AUC_diff']
        if not (pd.isna(auc_withG) or pd.isna(auc_0G)):
            print(f"{disease:<30} {auc_withG:<20.4f} {auc_0G:<20.4f} {diff:<15.4f}")
    
    print("\n10-YEAR PREDICTIONS:")
    print("-" * 80)
    print(f"{'Disease':<30} {'With-G AUC':<20} {'0G AUC':<20} {'Difference':<15}")
    print("-" * 80)
    for disease in comparison_10yr.index[:10]:  # Top 10 by absolute difference
        auc_withG = comparison_10yr.loc[disease, 'AUC_withG']
        auc_0G = comparison_10yr.loc[disease, 'AUC_0G']
        diff = comparison_10yr.loc[disease, 'AUC_diff']
        if not (pd.isna(auc_withG) or pd.isna(auc_0G)):
            print(f"{disease:<30} {auc_withG:<20.4f} {auc_0G:<20.4f} {diff:<15.4f}")
    
    # Calculate overall statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    
    print(f"\n1-Year Predictions:")
    print(f"  Mean AUC difference: {comparison_1yr['AUC_diff'].mean():.6f}")
    print(f"  Median AUC difference: {comparison_1yr['AUC_diff'].median():.6f}")
    print(f"  Max absolute difference: {comparison_1yr['AUC_diff'].abs().max():.6f}")
    print(f"  Diseases with difference > 0.01: {(comparison_1yr['AUC_diff'].abs() > 0.01).sum()}")
    print(f"  Diseases with difference > 0.005: {(comparison_1yr['AUC_diff'].abs() > 0.005).sum()}")
    
    print(f"\n10-Year Predictions:")
    print(f"  Mean AUC difference: {comparison_10yr['AUC_diff'].mean():.6f}")
    print(f"  Median AUC difference: {comparison_10yr['AUC_diff'].median():.6f}")
    print(f"  Max absolute difference: {comparison_10yr['AUC_diff'].abs().max():.6f}")
    print(f"  Diseases with difference > 0.01: {(comparison_10yr['AUC_diff'].abs() > 0.01).sum()}")
    print(f"  Diseases with difference > 0.005: {(comparison_10yr['AUC_diff'].abs() > 0.005).sum()}")
    
    print("\n" + "="*80)
    print("COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Results saved to: {output_dir}")

if __name__ == '__main__':
    main()

