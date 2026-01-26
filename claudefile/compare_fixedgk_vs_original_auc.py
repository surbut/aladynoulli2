#!/usr/bin/env python3
"""
Compare AUC for fixed gamma/kappa predictions vs original predictions.
Compares first few batches for:
- 10-year static AUC
- 1-year dynamic AUC

Usage:
    python compare_fixedgk_vs_original_auc.py --n_batches 5
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
from evaluatetdccode import (
    evaluate_major_diseases_wsex_with_bootstrap_dynamic_1year_different_start_end_numeric_sex
)

def load_essentials():
    """Load model essentials including disease names"""
    essentials_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/model_essentials.pt'
    essentials = torch.load(essentials_path, weights_only=False)
    return essentials

def main():
    parser = argparse.ArgumentParser(description='Compare fixed gamma/kappa AUC vs original')
    parser.add_argument('--n_batches', type=int, default=5,
                       help='Number of batches to compare')
    parser.add_argument('--batch_size', type=int, default=10000,
                       help='Batch size (samples per batch)')
    parser.add_argument('--n_bootstraps', type=int, default=100,
                       help='Number of bootstrap iterations for AUC CI')
    parser.add_argument('--old_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_correctedE_vectorized/',
                       help='Directory with original predictions')
    parser.add_argument('--new_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_fixedgk_vectorized/',
                       help='Directory with new fixed gamma/kappa predictions')
    
    args = parser.parse_args()
    
    print("="*80)
    print("COMPARING FIXED GAMMA/KAPPA VS ORIGINAL PREDICTIONS - AUC")
    print("="*80)
    print(f"Comparing first {args.n_batches} batches")
    print(f"Old directory: {args.old_dir}")
    print(f"New directory: {args.new_dir}")
    print("="*80)
    
    # Load essentials
    essentials = load_essentials()
    disease_names = essentials['disease_names']
    
    # Load full data (we'll subset by batch)
    print("\nLoading full data files...")
    Y_full = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/Y_tensor.pt', weights_only=False)
    E_full = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/E_enrollment_full.pt', weights_only=False)
    pce_df_full = pd.read_csv('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pce_prevent_full.csv')
    
    # Ensure both 'Sex' (for 10-year) and 'sex' (for 1-year) columns exist
    # The 10-year function expects 'Sex' (uppercase), 1-year expects 'sex' (lowercase, numeric)
    if 'Sex' not in pce_df_full.columns:
        if 'sex' in pce_df_full.columns:
            # Create 'Sex' from 'sex' if needed (convert numeric to string if necessary)
            if pce_df_full['sex'].dtype in [int, float]:
                # Assume 0=female, 1=male (or 1=female, 2=male for UKB)
                pce_df_full['Sex'] = pce_df_full['sex'].map({0: 'Female', 1: 'Male', 1: 'Female', 2: 'Male'}).fillna('Unknown')
            else:
                pce_df_full['Sex'] = pce_df_full['sex']
        else:
            raise ValueError("Neither 'Sex' nor 'sex' column found in pce_df")
    
    # Ensure 'sex' column exists for 1-year function (numeric: 0=female, 1=male)
    if 'sex' not in pce_df_full.columns:
        if 'Sex' in pce_df_full.columns:
            # Convert 'Sex' to numeric 'sex'
            pce_df_full['sex'] = pce_df_full['Sex'].map({'Female': 0, 'Male': 1, 'F': 0, 'M': 1}).fillna(-1)
            if (pce_df_full['sex'] == -1).any():
                print("⚠️  Warning: Some sex values couldn't be converted to numeric")
        else:
            raise ValueError("Neither 'Sex' nor 'sex' column found in pce_df")
    
    # Ensure 'age' column exists (lowercase, both functions expect it)
    if 'age' not in pce_df_full.columns:
        if 'Age' in pce_df_full.columns:
            pce_df_full['age'] = pce_df_full['Age']
        else:
            raise ValueError("Neither 'age' nor 'Age' column found in pce_df")
    
    print(f"Loaded Y: {Y_full.shape}, E: {E_full.shape}, pce_df: {len(pce_df_full)}")
    print(f"pce_df columns: {list(pce_df_full.columns)[:10]}...")  # Print first 10 columns
    
    # Process each batch
    all_results = []
    
    for batch_num in range(1, args.n_batches + 1):
        start_idx = (batch_num - 1) * args.batch_size
        end_idx = batch_num * args.batch_size
        
        print(f"\n{'='*80}")
        print(f"BATCH {batch_num}: samples {start_idx}-{end_idx}")
        print(f"{'='*80}")
        
        # File paths
        old_pi_file = os.path.join(args.old_dir, f'pi_enroll_fixedphi_sex_{start_idx}_{end_idx}.pt')
        new_pi_file = os.path.join(args.new_dir, f'pi_enroll_fixedphi_sex_{start_idx}_{end_idx}.pt')
        
        # Check files exist
        if not os.path.exists(old_pi_file):
            print(f"⚠️  Old pi file not found: {old_pi_file}")
            continue
        if not os.path.exists(new_pi_file):
            print(f"⚠️  New pi file not found: {new_pi_file}")
            continue
        
        # Load pi predictions
        print(f"Loading old pi: {Path(old_pi_file).name}")
        pi_old = torch.load(old_pi_file, weights_only=False)
        print(f"Loading new pi: {Path(new_pi_file).name}")
        pi_new = torch.load(new_pi_file, weights_only=False)
        
        print(f"Old pi shape: {pi_old.shape}")
        print(f"New pi shape: {pi_new.shape}")
        
        # Check shapes match
        if pi_old.shape != pi_new.shape:
            print(f"⚠️  Shape mismatch! Old: {pi_old.shape}, New: {pi_new.shape}")
            continue
        
        # Subset data for this batch
        print(f"\nSubsetting data for batch {batch_num}...")
        Y_batch = Y_full[start_idx:end_idx]
        E_batch = E_full[start_idx:end_idx]
        pce_df_batch = pce_df_full.iloc[start_idx:end_idx].reset_index(drop=True)
        
        print(f"Batch data shapes: Y={Y_batch.shape}, E={E_batch.shape}, pce_df={len(pce_df_batch)}")
        
        # Run AUC evaluations: STATIC 10-YEAR
        print("\n" + "-"*80)
        print("AUC EVALUATION: STATIC 10-YEAR")
        print("-"*80)
        
        print("Evaluating old predictions...")
        results_static_old = evaluate_major_diseases_wsex_with_bootstrap_from_pi(
            pi=pi_old,
            Y_100k=Y_batch,
            E_100k=E_batch,
            disease_names=disease_names,
            pce_df=pce_df_batch,
            n_bootstraps=args.n_bootstraps,
            follow_up_duration_years=10
        )
        
        print("Evaluating new predictions...")
        results_static_new = evaluate_major_diseases_wsex_with_bootstrap_from_pi(
            pi=pi_new,
            Y_100k=Y_batch,
            E_100k=E_batch,
            disease_names=disease_names,
            pce_df=pce_df_batch,
            n_bootstraps=args.n_bootstraps,
            follow_up_duration_years=10
        )
        
        # Run AUC evaluations: DYNAMIC 1-YEAR
        print("\n" + "-"*80)
        print("AUC EVALUATION: DYNAMIC 1-YEAR")
        print("-"*80)
        
        print("Evaluating old predictions...")
        results_1yr_old = evaluate_major_diseases_wsex_with_bootstrap_dynamic_1year_different_start_end_numeric_sex(
            pi=pi_old,
            Y_100k=Y_batch,
            E_100k=E_batch,
            disease_names=disease_names,
            pce_df=pce_df_batch,
            n_bootstraps=args.n_bootstraps
        )
        
        print("Evaluating new predictions...")
        results_1yr_new = evaluate_major_diseases_wsex_with_bootstrap_dynamic_1year_different_start_end_numeric_sex(
            pi=pi_new,
            Y_100k=Y_batch,
            E_100k=E_batch,
            disease_names=disease_names,
            pce_df=pce_df_batch,
            n_bootstraps=args.n_bootstraps
        )
        
        # Compare results
        print("\n" + "="*80)
        print(f"BATCH {batch_num} COMPARISON SUMMARY")
        print("="*80)
        
        # Static 10-year comparison
        print("\nSTATIC 10-YEAR AUC COMPARISON:")
        print(f"{'Disease':<30} {'Old AUC':<15} {'New AUC':<15} {'Difference':<15}")
        print("-"*80)
        
        all_diseases = set(results_static_old.keys()) | set(results_static_new.keys())
        static_comparison = []
        for disease in sorted(all_diseases):
            auc_old = results_static_old.get(disease, {}).get('auc', np.nan)
            auc_new = results_static_new.get(disease, {}).get('auc', np.nan)
            diff = auc_new - auc_old
            static_comparison.append({
                'batch': batch_num,
                'disease': disease,
                'old_auc': auc_old,
                'new_auc': auc_new,
                'difference': diff,
                'horizon': 'static_10yr'
            })
            if not np.isnan(auc_old) and not np.isnan(auc_new):
                print(f"{disease:<30} {auc_old:>8.4f}        {auc_new:>8.4f}        {diff:>+8.4f}")
        
        # Dynamic 1-year comparison
        print("\nDYNAMIC 1-YEAR AUC COMPARISON:")
        print(f"{'Disease':<30} {'Old AUC':<15} {'New AUC':<15} {'Difference':<15}")
        print("-"*80)
        
        dynamic_comparison = []
        for disease in sorted(all_diseases):
            auc_old = results_1yr_old.get(disease, {}).get('auc', np.nan)
            auc_new = results_1yr_new.get(disease, {}).get('auc', np.nan)
            diff = auc_new - auc_old
            dynamic_comparison.append({
                'batch': batch_num,
                'disease': disease,
                'old_auc': auc_old,
                'new_auc': auc_new,
                'difference': diff,
                'horizon': 'dynamic_1yr'
            })
            if not np.isnan(auc_old) and not np.isnan(auc_new):
                print(f"{disease:<30} {auc_old:>8.4f}        {auc_new:>8.4f}        {diff:>+8.4f}")
        
        # Store results
        all_results.extend(static_comparison)
        all_results.extend(dynamic_comparison)
        
        print(f"\n✓ Batch {batch_num} complete!")
    
    # Save summary
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_file = '/Users/sarahurbut/aladynoulli2/claudefile/fixedgk_vs_original_auc_comparison.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"✓ Saved comparison results to: {output_file}")
        print(f"  Total comparisons: {len(all_results)}")
        print(f"  Batches analyzed: {args.n_batches}")
        
        # Print summary statistics
        print("\nSUMMARY STATISTICS:")
        print("-"*80)
        for horizon in ['static_10yr', 'dynamic_1yr']:
            horizon_results = results_df[results_df['horizon'] == horizon]
            if len(horizon_results) > 0:
                diffs = horizon_results['difference'].dropna()
                print(f"\n{horizon.upper()}:")
                print(f"  Mean difference: {diffs.mean():.6f}")
                print(f"  Std difference: {diffs.std():.6f}")
                print(f"  Min difference: {diffs.min():.6f}")
                print(f"  Max difference: {diffs.max():.6f}")
                print(f"  Mean absolute difference: {np.abs(diffs).mean():.6f}")
    
    print(f"\n{'='*80}")
    print("COMPLETED")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
