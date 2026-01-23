#!/usr/bin/env python3
"""
Pool leave-one-out pi predictions and evaluate AUCs (static 10yr and dynamic 10yr).

This script:
1. Loads all 40 leave-one-out batch pi files
2. Concatenates them into a single pooled tensor
3. Saves the pooled pi as FULL file
4. Runs static 10-year and dynamic 10-year AUC evaluations
5. Compares with master pooled results

Usage:
    python pool_loo_pis_and_evaluate.py --n_bootstraps 100
"""

import argparse
import sys
import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# Add path for imports
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/')
from fig5utils import (
    evaluate_major_diseases_wsex_with_bootstrap_dynamic_from_pi,
    evaluate_major_diseases_wsex_with_bootstrap_from_pi
)

def load_essentials():
    """Load model essentials including disease names"""
    essentials_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/model_essentials.pt'
    essentials = torch.load(essentials_path, weights_only=False)
    return essentials

def pool_loo_predictions(loo_base_dir, batch_size=10000, total_batches=40):
    """Load and pool all leave-one-out pi predictions."""
    print("="*80)
    print("POOLING LEAVE-ONE-OUT PREDICTIONS")
    print("="*80)
    
    loo_base_dir = Path(loo_base_dir)
    pi_batches = []
    valid_batches = []
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size
        
        pi_file = loo_base_dir / f'batch_{batch_idx}' / f'pi_enroll_fixedphi_sex_{start_idx}_{end_idx}.pt'
        
        if not pi_file.exists():
            print(f"⚠️  Skipping batch {batch_idx} (file not found: {pi_file})")
            continue
        
        print(f"Loading batch {batch_idx}: {pi_file.name}")
        pi_batch = torch.load(pi_file, weights_only=False)
        print(f"  Shape: {pi_batch.shape}")
        pi_batches.append(pi_batch)
        valid_batches.append(batch_idx)
    
    if not pi_batches:
        raise ValueError("No prediction files found!")
    
    # Concatenate all batches
    print(f"\nConcatenating {len(pi_batches)} batches...")
    pi_pooled = torch.cat(pi_batches, dim=0)
    print(f"✓ Pooled pi shape: {pi_pooled.shape}")
    
    # Save pooled pi
    output_file = loo_base_dir / 'pi_enroll_fixedphi_sex_FULL.pt'
    torch.save(pi_pooled, output_file)
    print(f"✓ Saved pooled pi to: {output_file}")
    
    return pi_pooled, valid_batches

def main():
    parser = argparse.ArgumentParser(description='Pool leave-one-out pis and evaluate AUCs')
    parser.add_argument('--loo_pi_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/leave_one_out_correctedE/',
                       help='Base directory containing leave-one-out pi predictions')
    parser.add_argument('--master_pi_file', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_correctedE_vectorized/pi_enroll_fixedphi_sex_FULL.pt',
                       help='Path to master pooled pi file')
    parser.add_argument('--batch_size', type=int, default=10000,
                       help='Batch size (samples per batch)')
    parser.add_argument('--total_batches', type=int, default=40,
                       help='Total number of batches')
    parser.add_argument('--n_bootstraps', type=int, default=100,
                       help='Number of bootstrap iterations for AUC CI')
    parser.add_argument('--max_patients', type=int, default=400000,
                       help='Maximum number of patients to use')
    parser.add_argument('--skip_pooling', action='store_true',
                       help='Skip pooling step (use existing FULL file)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("POOL LEAVE-ONE-OUT PIS AND EVALUATE AUCs")
    print("="*80)
    print(f"Leave-one-out directory: {args.loo_pi_dir}")
    print(f"Master pi file: {args.master_pi_file}")
    print(f"Bootstrap iterations: {args.n_bootstraps}")
    print("="*80)
    
    # Load essentials
    essentials = load_essentials()
    disease_names = essentials['disease_names']
    
    # Load full data
    print("\nLoading full data files...")
    Y_full = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/Y_tensor.pt', weights_only=False)
    E_full = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/E_enrollment_full.pt', weights_only=False)
    pce_df_full = pd.read_csv('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pce_prevent_full.csv')
    
    print(f"Loaded Y: {Y_full.shape}, E: {E_full.shape}, pce_df: {len(pce_df_full)}")
    
    # Subset to max_patients
    print(f"\nSubsetting to first {args.max_patients} patients...")
    Y_full = Y_full[:args.max_patients]
    E_full = E_full[:args.max_patients]
    pce_df_full = pce_df_full.iloc[:args.max_patients].reset_index(drop=True)
    
    print(f"After subsetting: Y: {Y_full.shape}, E: {E_full.shape}, pce_df: {len(pce_df_full)}")
    
    # Pool leave-one-out predictions
    loo_full_file = Path(args.loo_pi_dir) / 'pi_enroll_fixedphi_sex_FULL.pt'
    
    if args.skip_pooling and loo_full_file.exists():
        print(f"\nSkipping pooling, loading existing FULL file: {loo_full_file}")
        pi_loo_pooled = torch.load(loo_full_file, weights_only=False)
        print(f"Loaded pi shape: {pi_loo_pooled.shape}")
    else:
        pi_loo_pooled, valid_batches = pool_loo_predictions(
            args.loo_pi_dir, 
            batch_size=args.batch_size,
            total_batches=args.total_batches
        )
    
    # Load master pooled pi
    print(f"\nLoading master pooled pi: {args.master_pi_file}")
    pi_master_pooled = torch.load(args.master_pi_file, weights_only=False)
    print(f"Master pi shape: {pi_master_pooled.shape}")
    
    # Verify shapes match
    if pi_loo_pooled.shape != pi_master_pooled.shape:
        print(f"⚠️  Shape mismatch! LOO: {pi_loo_pooled.shape}, Master: {pi_master_pooled.shape}")
        # Use minimum size
        min_size = min(pi_loo_pooled.shape[0], pi_master_pooled.shape[0], len(pce_df_full))
        print(f"Using first {min_size} patients for comparison")
        pi_loo_pooled = pi_loo_pooled[:min_size]
        pi_master_pooled = pi_master_pooled[:min_size]
        Y_full = Y_full[:min_size]
        E_full = E_full[:min_size]
        pce_df_full = pce_df_full.iloc[:min_size].reset_index(drop=True)
    
    # Calculate correlation (sample subset to avoid memory issues)
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS (sampled)")
    print("="*80)
    
    from scipy.stats import pearsonr
    # Sample 10,000 random elements for correlation (much faster)
    n_samples = 10000
    total_elements = pi_loo_pooled.numel()
    if total_elements > n_samples:
        # Random sampling
        np.random.seed(42)  # For reproducibility
        sample_indices = np.random.choice(total_elements, size=n_samples, replace=False)
        pi_loo_sample = pi_loo_pooled.flatten().cpu().numpy()[sample_indices]
        pi_master_sample = pi_master_pooled.flatten().cpu().numpy()[sample_indices]
        print(f"Sampling {n_samples:,} elements from {total_elements:,} total for correlation")
    else:
        pi_loo_sample = pi_loo_pooled.flatten().cpu().numpy()
        pi_master_sample = pi_master_pooled.flatten().cpu().numpy()
        print(f"Using all {total_elements:,} elements for correlation")
    
    corr, pval = pearsonr(pi_loo_sample, pi_master_sample)
    print(f"Overall correlation: r = {corr:.6f}, p = {pval:.2e}")
    
    # Run AUC evaluations
    print("\n" + "="*80)
    print("AUC EVALUATION: STATIC 10-YEAR")
    print("="*80)
    
    print("\nEvaluating master predictions...")
    results_static_master = evaluate_major_diseases_wsex_with_bootstrap_from_pi(
        pi=pi_master_pooled,
        Y_100k=Y_full,
        E_100k=E_full,
        disease_names=disease_names,
        pce_df=pce_df_full,
        n_bootstraps=args.n_bootstraps,
        follow_up_duration_years=10
    )
    
    print("\nEvaluating leave-one-out predictions...")
    results_static_loo = evaluate_major_diseases_wsex_with_bootstrap_from_pi(
        pi=pi_loo_pooled,
        Y_100k=Y_full,
        E_100k=E_full,
        disease_names=disease_names,
        pce_df=pce_df_full,
        n_bootstraps=args.n_bootstraps,
        follow_up_duration_years=10
    )
    
    print("\n" + "="*80)
    print("AUC EVALUATION: DYNAMIC 10-YEAR")
    print("="*80)
    
    print("\nEvaluating master predictions...")
    results_dynamic_master = evaluate_major_diseases_wsex_with_bootstrap_dynamic_from_pi(
        pi=pi_master_pooled,
        Y_100k=Y_full,
        E_100k=E_full,
        disease_names=disease_names,
        pce_df=pce_df_full,
        n_bootstraps=args.n_bootstraps,
        follow_up_duration_years=10
    )
    
    print("\nEvaluating leave-one-out predictions...")
    results_dynamic_loo = evaluate_major_diseases_wsex_with_bootstrap_dynamic_from_pi(
        pi=pi_loo_pooled,
        Y_100k=Y_full,
        E_100k=E_full,
        disease_names=disease_names,
        pce_df=pce_df_full,
        n_bootstraps=args.n_bootstraps,
        follow_up_duration_years=10
    )
    
    # Compare results
    print("\n" + "="*80)
    print("POOLED COMPARISON SUMMARY")
    print("="*80)
    
    # Static 10-year comparison
    print("\nSTATIC 10-YEAR AUC COMPARISON:")
    print(f"{'Disease':<30} {'Master AUC':<15} {'LOO AUC':<15} {'Difference':<15}")
    print("-"*80)
    
    all_diseases = set(results_static_master.keys()) | set(results_static_loo.keys())
    static_comparison = []
    for disease in sorted(all_diseases):
        auc_master = results_static_master.get(disease, {}).get('auc', np.nan)
        auc_loo = results_static_loo.get(disease, {}).get('auc', np.nan)
        diff = auc_loo - auc_master
        static_comparison.append({
            'disease': disease,
            'master_auc': auc_master,
            'loo_auc': auc_loo,
            'difference': diff,
            'horizon': 'static_10yr'
        })
        if not np.isnan(auc_master) and not np.isnan(auc_loo):
            print(f"{disease:<30} {auc_master:>8.4f}        {auc_loo:>8.4f}        {diff:>+8.4f}")
    
    # Dynamic 10-year comparison
    print("\nDYNAMIC 10-YEAR AUC COMPARISON:")
    print(f"{'Disease':<30} {'Master AUC':<15} {'LOO AUC':<15} {'Difference':<15}")
    print("-"*80)
    
    dynamic_comparison = []
    for disease in sorted(all_diseases):
        auc_master = results_dynamic_master.get(disease, {}).get('auc', np.nan)
        auc_loo = results_dynamic_loo.get(disease, {}).get('auc', np.nan)
        diff = auc_loo - auc_master
        dynamic_comparison.append({
            'disease': disease,
            'master_auc': auc_master,
            'loo_auc': auc_loo,
            'difference': diff,
            'horizon': 'dynamic_10yr'
        })
        if not np.isnan(auc_master) and not np.isnan(auc_loo):
            print(f"{disease:<30} {auc_master:>8.4f}        {auc_loo:>8.4f}        {diff:>+8.4f}")
    
    # Save results
    all_results = static_comparison + dynamic_comparison
    results_df = pd.DataFrame(all_results)
    output_file = '/Users/sarahurbut/aladynoulli2/claudefile/loo_vs_master_pooled_comparison.csv'
    results_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"✓ Saved comparison results to: {output_file}")
    print(f"  Total comparisons: {len(all_results)}")
    print(f"  Correlation: r = {corr:.6f}")
    
    # Summary statistics
    diffs = [r['difference'] for r in all_results if not np.isnan(r['difference'])]
    if diffs:
        abs_diffs = [abs(d) for d in diffs]
        print(f"\nAUC Difference Statistics:")
        print(f"  Mean: {np.mean(diffs):.6f}")
        print(f"  Median: {np.median(diffs):.6f}")
        print(f"  Mean |diff|: {np.mean(abs_diffs):.6f}")
        print(f"  Max |diff|: {max(abs_diffs):.6f}")
        print(f"  |diff| < 0.01: {sum(1 for d in abs_diffs if d < 0.01)} ({100*sum(1 for d in abs_diffs if d < 0.01)/len(abs_diffs):.1f}%)")
    
    print(f"\n{'='*80}")
    print("COMPLETED")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()

