#!/usr/bin/env python3
"""
Evaluate 1-year and 10-year predictions for all washout versions.

This script:
- Loads pi tensors from each washout version (no_washout, 1month, 3month, 6month)
- Calculates 1-year predictions (offset=0) using washout evaluation function
- Calculates 10-year predictions using time horizon evaluation function
- Saves results for each washout version

Usage:
    python evaluate_washout_predictions.py
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
from evaluatetdccode import evaluate_major_diseases_wsex_with_bootstrap_dynamic_1year_different_start_end_numeric_sex
from fig5utils import evaluate_major_diseases_wsex_with_bootstrap_from_pi

# Load essentials (disease names, etc.)
def load_essentials():
    """Load model essentials including disease names"""
    essentials_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/model_essentials.pt'
    essentials = torch.load(essentials_path, weights_only=False)
    return essentials

def main():
    parser = argparse.ArgumentParser(description='Evaluate washout predictions (1yr and 10yr)')
    parser.add_argument('--washout_base_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox/washout_comparison_10k/',
                       help='Base directory containing washout pi tensors')
    parser.add_argument('--n_bootstraps', type=int, default=100,
                       help='Number of bootstrap iterations')
    parser.add_argument('--output_dir', type=str,
                       default='/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/washout_evaluation/',
                       help='Output directory for results')
    parser.add_argument('--subset_size', type=int, default=10000,
                       help='Number of patients in washout pi tensors (default: 10000)')
    
    args = parser.parse_args()
    
    # Define washout versions
    washout_versions = ['no_washout', '1month', '3month', '6month']
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("EVALUATING WASHOUT PREDICTIONS: 1-YEAR AND 10-YEAR")
    print("="*80)
    print(f"Washout base directory: {args.washout_base_dir}")
    print(f"Washout versions: {washout_versions}")
    print(f"Output directory: {output_dir}")
    print(f"Subset size: {args.subset_size}")
    print("="*80)
    
    # Load common data (subset to first 10K for washout analysis)
    print("\nLoading common data...")
    Y_full = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/Y_tensor.pt', weights_only=False)
    E_full = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/E_enrollment_full.pt', weights_only=False)
    pce_df_full = pd.read_csv('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pce_prevent_full.csv')
    essentials = load_essentials()
    disease_names = essentials['disease_names']
    
    # Subset to first 10K patients (matching washout pi tensors)
    print(f"\nSubsetting to first {args.subset_size} patients...")
    Y_subset = Y_full[:args.subset_size]
    E_subset = E_full[:args.subset_size]
    pce_df_subset = pce_df_full.iloc[:args.subset_size].reset_index(drop=True)
    
    print(f"After subsetting: Y: {Y_subset.shape[0]}, E: {E_subset.shape[0]}, pce_df: {len(pce_df_subset)}")
    
    # Convert Sex column to numeric if needed
    if 'Sex' in pce_df_subset.columns and pce_df_subset['Sex'].dtype == 'object':
        pce_df_subset['sex'] = pce_df_subset['Sex'].map({'Female': 0, 'Male': 1}).astype(int)
    elif 'sex' not in pce_df_subset.columns:
        raise ValueError("Need 'Sex' or 'sex' column in pce_df")
    
    # Process each washout version
    all_results = {}
    
    for washout_version in washout_versions:
        print(f"\n{'='*80}")
        print(f"PROCESSING WASHOUT VERSION: {washout_version.upper()}")
        print(f"{'='*80}")
        
        # Load pi tensor for this washout version
        pi_path = Path(args.washout_base_dir) / washout_version / f'pi_washout_{washout_version}_0_{args.subset_size}.pt'
        
        if not pi_path.exists():
            print(f"⚠️  WARNING: Pi tensor not found: {pi_path}")
            print(f"   Skipping {washout_version}")
            continue
        
        print(f"Loading pi tensor from: {pi_path}")
        pi_washout = torch.load(pi_path, weights_only=False)
        print(f"Pi tensor shape: {pi_washout.shape}")
        
        # Verify sizes match
        N = pi_washout.shape[0]
        if not (N == Y_subset.shape[0] == E_subset.shape[0] == len(pce_df_subset)):
            print(f"⚠️  WARNING: Size mismatch for {washout_version}!")
            print(f"   pi: {N}, Y: {Y_subset.shape[0]}, E: {E_subset.shape[0]}, pce_df: {len(pce_df_subset)}")
            continue
        
        washout_results = {}
        
        # 1. Calculate 1-year predictions (offset=0)
        print(f"\n{'='*60}")
        print(f"1-YEAR PREDICTIONS (offset=0)")
        print(f"{'='*60}")
        
        results_1yr = evaluate_major_diseases_wsex_with_bootstrap_dynamic_1year_different_start_end_numeric_sex(
            pi=pi_washout,
            Y_100k=Y_subset,
            E_100k=E_subset,
            disease_names=disease_names,
            pce_df=pce_df_subset,
            n_bootstraps=args.n_bootstraps,
            follow_up_duration_years=1,
            start_offset=0
        )
        
        washout_results['1yr'] = results_1yr
        
        # Save 1-year results
        results_1yr_df = pd.DataFrame({
            'Disease': list(results_1yr.keys()),
            'AUC': [r['auc'] for r in results_1yr.values()],
            'CI_lower': [r['ci_lower'] for r in results_1yr.values()],
            'CI_upper': [r['ci_upper'] for r in results_1yr.values()],
            'N_Events': [r['n_events'] for r in results_1yr.values()],
            'Event_Rate': [r['event_rate'] for r in results_1yr.values()]
        })
        results_1yr_df = results_1yr_df.set_index('Disease').sort_values('AUC', ascending=False)
        
        output_1yr = output_dir / f'{washout_version}_1yr_results.csv'
        results_1yr_df.to_csv(output_1yr)
        print(f"✓ Saved 1-year results to {output_1yr}")
        
        # 2. Calculate 10-year predictions (static)
        print(f"\n{'='*60}")
        print(f"10-YEAR PREDICTIONS (static)")
        print(f"{'='*60}")
        
        results_10yr = evaluate_major_diseases_wsex_with_bootstrap_from_pi(
            pi=pi_washout,
            Y_100k=Y_subset,
            E_100k=E_subset,
            disease_names=disease_names,
            pce_df=pce_df_subset,
            n_bootstraps=args.n_bootstraps,
            follow_up_duration_years=10
        )
        
        washout_results['10yr'] = results_10yr
        
        # Save 10-year results
        results_10yr_df = pd.DataFrame({
            'Disease': list(results_10yr.keys()),
            'AUC': [r['auc'] for r in results_10yr.values()],
            'CI_lower': [r['ci_lower'] for r in results_10yr.values()],
            'CI_upper': [r['ci_upper'] for r in results_10yr.values()],
            'N_Events': [r['n_events'] for r in results_10yr.values()],
            'Event_Rate': [r['event_rate'] for r in results_10yr.values()]
        })
        results_10yr_df = results_10yr_df.set_index('Disease').sort_values('AUC', ascending=False)
        
        output_10yr = output_dir / f'{washout_version}_10yr_results.csv'
        results_10yr_df.to_csv(output_10yr)
        print(f"✓ Saved 10-year results to {output_10yr}")
        
        all_results[washout_version] = washout_results
    
    # Create combined comparison file
    print(f"\n{'='*80}")
    print("CREATING COMBINED COMPARISON FILE")
    print(f"{'='*80}")
    
    # Collect all diseases
    all_diseases = set()
    for washout_results in all_results.values():
        for horizon_results in washout_results.values():
            all_diseases.update(horizon_results.keys())
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(index=sorted(all_diseases))
    
    for washout_version, washout_results in all_results.items():
        for horizon, results in washout_results.items():
            comparison_df[f'{washout_version}_{horizon}_AUC'] = [
                results.get(d, {}).get('auc', np.nan) for d in comparison_df.index
            ]
            comparison_df[f'{washout_version}_{horizon}_CI_lower'] = [
                results.get(d, {}).get('ci_lower', np.nan) for d in comparison_df.index
            ]
            comparison_df[f'{washout_version}_{horizon}_CI_upper'] = [
                results.get(d, {}).get('ci_upper', np.nan) for d in comparison_df.index
            ]
            comparison_df[f'{washout_version}_{horizon}_N_Events'] = [
                results.get(d, {}).get('n_events', np.nan) for d in comparison_df.index
            ]
    
    comparison_file = output_dir / 'washout_comparison_1yr_10yr.csv'
    comparison_df.to_csv(comparison_file)
    print(f"✓ Saved combined comparison to {comparison_file}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"\nIndividual results:")
    for washout_version in all_results.keys():
        print(f"  {washout_version}:")
        print(f"    - {washout_version}_1yr_results.csv")
        print(f"    - {washout_version}_10yr_results.csv")
    print(f"\nCombined comparison:")
    print(f"  - washout_comparison_1yr_10yr.csv")
    
    # Print sample summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    for washout_version, washout_results in all_results.items():
        print(f"\n{washout_version.upper()}:")
        for horizon, results in washout_results.items():
            aucs = [r['auc'] for r in results.values() if 'auc' in r]
            if aucs:
                print(f"  {horizon}: Mean AUC = {np.mean(aucs):.3f}, Median AUC = {np.median(aucs):.3f}, N diseases = {len(aucs)}")

if __name__ == '__main__':
    main()

