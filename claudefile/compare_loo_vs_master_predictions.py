#!/usr/bin/env python3
"""
Compare leave-one-out predictions with master predictions for a few batches.

This script:
1. Loads pi predictions from both master and leave-one-out checkpoints
2. Calculates correlation between predictions
3. Runs AUC evaluation for static 10-year and 10-year dynamic
4. Compares results

Usage:
    python compare_loo_vs_master_predictions.py --batches 0,1,2
"""

import argparse
import sys
import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

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

def main():
    parser = argparse.ArgumentParser(description='Compare leave-one-out vs master predictions')
    parser.add_argument('--batches', type=str, default='0,1,2',
                       help='Comma-separated list of batch indices to check (e.g., 0,1,2)')
    parser.add_argument('--batch_size', type=int, default=10000,
                       help='Batch size (samples per batch)')
    parser.add_argument('--n_bootstraps', type=int, default=100,
                       help='Number of bootstrap iterations for AUC CI')
    parser.add_argument('--master_pi_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_correctedE_vectorized/',
                       help='Directory containing master pi predictions')
    parser.add_argument('--loo_pi_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/leave_one_out_correctedE/',
                       help='Base directory containing leave-one-out pi predictions')
    
    args = parser.parse_args()
    
    # Parse batch indices
    batch_indices = [int(b.strip()) for b in args.batches.split(',')]
    
    print("="*80)
    print("COMPARING LEAVE-ONE-OUT VS MASTER PREDICTIONS")
    print("="*80)
    print(f"Batches to check: {batch_indices}")
    print(f"Master pi directory: {args.master_pi_dir}")
    print(f"Leave-one-out pi directory: {args.loo_pi_dir}")
    print("="*80)
    
    # Load essentials
    essentials = load_essentials()
    disease_names = essentials['disease_names']
    
    # Load full data (we'll subset by batch)
    print("\nLoading full data files...")
    Y_full = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/Y_tensor.pt', weights_only=False)
    E_full = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/E_enrollment_full.pt', weights_only=False)
    pce_df_full = pd.read_csv('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pce_prevent_full.csv')
    
    print(f"Loaded Y: {Y_full.shape}, E: {E_full.shape}, pce_df: {len(pce_df_full)}")
    
    # Process each batch
    all_results = []
    
    for batch_idx in batch_indices:
        start_idx = batch_idx * args.batch_size
        end_idx = (batch_idx + 1) * args.batch_size
        
        print(f"\n{'='*80}")
        print(f"BATCH {batch_idx}: samples {start_idx}-{end_idx}")
        print(f"{'='*80}")
        
        # File paths
        master_pi_file = os.path.join(args.master_pi_dir, f'pi_enroll_fixedphi_sex_{start_idx}_{end_idx}.pt')
        loo_pi_file = os.path.join(args.loo_pi_dir, f'batch_{batch_idx}/pi_enroll_fixedphi_sex_{start_idx}_{end_idx}.pt')
        
        # Check files exist
        if not os.path.exists(master_pi_file):
            print(f"⚠️  Master pi file not found: {master_pi_file}")
            continue
        if not os.path.exists(loo_pi_file):
            print(f"⚠️  Leave-one-out pi file not found: {loo_pi_file}")
            continue
        
        # Load pi predictions
        print(f"Loading master pi: {Path(master_pi_file).name}")
        pi_master = torch.load(master_pi_file, weights_only=False)
        print(f"Loading leave-one-out pi: {Path(loo_pi_file).name}")
        pi_loo = torch.load(loo_pi_file, weights_only=False)
        
        print(f"Master pi shape: {pi_master.shape}")
        print(f"Leave-one-out pi shape: {pi_loo.shape}")
        
        # Check shapes match
        if pi_master.shape != pi_loo.shape:
            print(f"⚠️  Shape mismatch! Master: {pi_master.shape}, LOO: {pi_loo.shape}")
            continue
        
        # Calculate correlation
        print("\n" + "-"*80)
        print("CORRELATION ANALYSIS")
        print("-"*80)
        
        # Flatten for correlation
        pi_master_flat = pi_master.flatten().cpu().numpy()
        pi_loo_flat = pi_loo.flatten().cpu().numpy()
        
        corr, pval = pearsonr(pi_master_flat, pi_loo_flat)
        print(f"Overall correlation: r = {corr:.6f}, p = {pval:.2e}")
        
        # Per-disease correlation
        print("\nPer-disease correlations (top 10):")
        disease_corrs = []
        for d_idx, disease in enumerate(disease_names):
            pi_master_d = pi_master[:, d_idx, :].flatten().cpu().numpy()
            pi_loo_d = pi_loo[:, d_idx, :].flatten().cpu().numpy()
            corr_d, _ = pearsonr(pi_master_d, pi_loo_d)
            disease_corrs.append((disease, corr_d))
        
        disease_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
        for disease, corr_d in disease_corrs[:10]:
            print(f"  {disease:30s}: r = {corr_d:.6f}")
        
        # Subset data for this batch
        print(f"\nSubsetting data for batch {batch_idx}...")
        Y_batch = Y_full[start_idx:end_idx]
        E_batch = E_full[start_idx:end_idx]
        pce_df_batch = pce_df_full.iloc[start_idx:end_idx].reset_index(drop=True)
        
        print(f"Batch data shapes: Y={Y_batch.shape}, E={E_batch.shape}, pce_df={len(pce_df_batch)}")
        
        # Run AUC evaluations
        print("\n" + "-"*80)
        print("AUC EVALUATION: STATIC 10-YEAR")
        print("-"*80)
        
        results_static_master = evaluate_major_diseases_wsex_with_bootstrap_from_pi(
            pi=pi_master,
            Y_100k=Y_batch,
            E_100k=E_batch,
            disease_names=disease_names,
            pce_df=pce_df_batch,
            n_bootstraps=args.n_bootstraps,
            follow_up_duration_years=10
        )
        
        results_static_loo = evaluate_major_diseases_wsex_with_bootstrap_from_pi(
            pi=pi_loo,
            Y_100k=Y_batch,
            E_100k=E_batch,
            disease_names=disease_names,
            pce_df=pce_df_batch,
            n_bootstraps=args.n_bootstraps,
            follow_up_duration_years=10
        )
        
        print("\n" + "-"*80)
        print("AUC EVALUATION: DYNAMIC 10-YEAR")
        print("-"*80)
        
        results_dynamic_master = evaluate_major_diseases_wsex_with_bootstrap_dynamic_from_pi(
            pi=pi_master,
            Y_100k=Y_batch,
            E_100k=E_batch,
            disease_names=disease_names,
            pce_df=pce_df_batch,
            n_bootstraps=args.n_bootstraps,
            follow_up_duration_years=10
        )
        
        results_dynamic_loo = evaluate_major_diseases_wsex_with_bootstrap_dynamic_from_pi(
            pi=pi_loo,
            Y_100k=Y_batch,
            E_100k=E_batch,
            disease_names=disease_names,
            pce_df=pce_df_batch,
            n_bootstraps=args.n_bootstraps,
            follow_up_duration_years=10
        )
        
        # Compare results
        print("\n" + "="*80)
        print(f"BATCH {batch_idx} COMPARISON SUMMARY")
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
                'batch': batch_idx,
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
                'batch': batch_idx,
                'disease': disease,
                'master_auc': auc_master,
                'loo_auc': auc_loo,
                'difference': diff,
                'horizon': 'dynamic_10yr'
            })
            if not np.isnan(auc_master) and not np.isnan(auc_loo):
                print(f"{disease:<30} {auc_master:>8.4f}        {auc_loo:>8.4f}        {diff:>+8.4f}")
        
        # Store results
        all_results.extend(static_comparison)
        all_results.extend(dynamic_comparison)
        
        print(f"\n✓ Batch {batch_idx} complete!")
    
    # Save summary
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_file = '/Users/sarahurbut/aladynoulli2/claudefile/loo_vs_master_comparison.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"✓ Saved comparison results to: {output_file}")
        print(f"  Total comparisons: {len(all_results)}")
        print(f"  Batches analyzed: {len(batch_indices)}")
    
    print(f"\n{'='*80}")
    print("COMPLETED")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()

