#!/usr/bin/env python3
"""
Generate time horizon predictions (5yr, 10yr, 30yr, static 10yr) using pre-computed pi tensors.

This script processes all patients at once (not batch-by-batch) for statistically better results.
Uses pre-computed pi tensors to avoid model forward passes.

Usage:
    python generate_time_horizon_predictions.py --approach pooled_retrospective --horizons 5,10,30,static10
    python generate_time_horizon_predictions.py --approach pooled_enrollment --horizons 5,10,30,static10
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

# Load essentials (disease names, etc.)
def load_essentials():
    """Load model essentials including disease names"""
    essentials_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/model_essentials.pt'
    essentials = torch.load(essentials_path, weights_only=False)
    return essentials

def main():
    parser = argparse.ArgumentParser(description='Generate time horizon predictions from pre-computed pi')
    parser.add_argument('--approach', type=str, required=True, 
                       choices=['pooled_enrollment', 'pooled_retrospective'],
                       help='Which approach to use')
    parser.add_argument('--horizons', type=str, default='5,10,30,static10',
                       help='Comma-separated list of horizons: 5,10,30,static10')
    parser.add_argument('--n_bootstraps', type=int, default=100,
                       help='Number of bootstrap iterations')
    parser.add_argument('--output_dir', type=str, 
                       default='/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks/results/time_horizons/',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Parse horizons
    horizons = [h.strip() for h in args.horizons.split(',')]
    
    # Set up paths based on approach
    if args.approach == 'pooled_enrollment':
        pi_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_ENROLLMENT_pooled/pi_enroll_fixedphi_sex_FULL.pt'
        approach_name = 'pooled_enrollment'
    elif args.approach == 'pooled_retrospective':
        pi_path = '/Users/sarahurbut/Downloads/pi_full_400k.pt'
        approach_name = 'pooled_retrospective'
    
    # Create output directory
    output_dir = Path(args.output_dir) / approach_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f"GENERATING TIME HORIZON PREDICTIONS: {approach_name.upper()}")
    print("="*80)
    print(f"Pi tensor: {pi_path}")
    print(f"Horizons: {horizons}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    pi_full = torch.load(pi_path, weights_only=False)
    Y_full = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/Y_tensor.pt', weights_only=False)
    E_full = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/E_enrollment_full.pt', weights_only=False)
    pce_df_full = pd.read_csv('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pce_prevent_full.csv')
    essentials = load_essentials()
    disease_names = essentials['disease_names']
    
    print(f"Loaded pi tensor: {pi_full.shape}")
    print(f"Loaded Y tensor: {Y_full.shape}")
    print(f"Loaded E tensor: {E_full.shape}")
    print(f"Loaded pce_df: {len(pce_df_full)} patients")
    
    # Cap at 400K patients (explicit subsetting)
    MAX_PATIENTS = 400000
    print(f"\nSubsetting to first {MAX_PATIENTS} patients...")
    pi_full = pi_full[:MAX_PATIENTS]
    Y_full = Y_full[:MAX_PATIENTS]
    E_full = E_full[:MAX_PATIENTS]
    pce_df_full = pce_df_full.iloc[:MAX_PATIENTS].reset_index(drop=True)
    
    print(f"After subsetting: pi: {pi_full.shape[0]}, Y: {Y_full.shape[0]}, E: {E_full.shape[0]}, pce_df: {len(pce_df_full)}")
    
    # Verify sizes match after subsetting
    N = pi_full.shape[0]
    if not (N == Y_full.shape[0] == E_full.shape[0] == len(pce_df_full)):
        raise ValueError(f"Size mismatch after subsetting! pi: {N}, Y: {Y_full.shape[0]}, E: {E_full.shape[0]}, pce_df: {len(pce_df_full)}")
    
    # Check if results already exist
    expected_files = []
    for horizon in horizons:
        if horizon == 'static10':
            horizon_name = 'static_10yr'
        else:
            horizon_years = int(horizon.replace('yr', ''))
            horizon_name = f'{horizon_years}yr'
        expected_files.append(output_dir / f'{horizon_name}_results.csv')
    comparison_file = output_dir / 'comparison_all_horizons.csv'
    all_exist = all(f.exists() for f in expected_files) and comparison_file.exists()
    
    if all_exist:
        print("\n" + "="*80)
        print("RESULTS ALREADY EXIST - SKIPPING REGENERATION")
        print("="*80)
        print(f"Found existing results in: {output_dir}")
        for f in expected_files:
            print(f"  ✓ {f.name}")
        print(f"  ✓ {comparison_file.name}")
        print("\nTo regenerate, delete the existing result files first.")
        return
    
    # Process each horizon
    all_results = {}
    
    for horizon in horizons:
        if horizon == 'static10':
            horizon_name = 'static_10yr'
        else:
            horizon_years = int(horizon.replace('yr', ''))
            horizon_name = f'{horizon_years}yr'
        
        output_file = output_dir / f'{horizon_name}_results.csv'
        
        # Skip if this specific result already exists
        if output_file.exists():
            print(f"\n{'='*80}")
            print(f"SKIPPING HORIZON: {horizon_name} (already exists)")
            print(f"{'='*80}")
            print(f"  File exists: {output_file}")
            continue
        
        print(f"\n{'='*80}")
        print(f"PROCESSING HORIZON: {horizon}")
        print(f"{'='*80}")
        
        if horizon == 'static10':
            # Static 10-year (1-year score for 10-year outcome)
            print("Evaluating static 10-year predictions (1-year score for 10-year outcome)...")
            results = evaluate_major_diseases_wsex_with_bootstrap_from_pi(
                pi=pi_full,
                Y_100k=Y_full,
                E_100k=E_full,
                disease_names=disease_names,
                pce_df=pce_df_full,
                n_bootstraps=args.n_bootstraps,
                follow_up_duration_years=10
            )
        else:
            # Dynamic predictions (5yr, 10yr, 30yr)
            horizon_years = int(horizon.replace('yr', ''))
            print(f"Evaluating dynamic {horizon_years}-year predictions...")
            results = evaluate_major_diseases_wsex_with_bootstrap_dynamic_from_pi(
                pi=pi_full,
                Y_100k=Y_full,
                E_100k=E_full,
                disease_names=disease_names,
                pce_df=pce_df_full,
                n_bootstraps=args.n_bootstraps,
                follow_up_duration_years=horizon_years
            )
        
        all_results[horizon_name] = results
        
        # Save individual results
        results_df = pd.DataFrame({
            'Disease': list(results.keys()),
            'AUC': [r['auc'] for r in results.values()],
            'CI_lower': [r['ci_lower'] for r in results.values()],
            'CI_upper': [r['ci_upper'] for r in results.values()],
            'N_Events': [r['n_events'] for r in results.values()],
            'Event_Rate': [r['event_rate'] for r in results.values()]
        })
        results_df = results_df.set_index('Disease').sort_values('AUC', ascending=False)
        
        results_df.to_csv(output_file)
        print(f"\n✓ Saved results to {output_file}")
    
    # Create combined comparison file
    print(f"\n{'='*80}")
    print("CREATING COMBINED COMPARISON FILE")
    print(f"{'='*80}")
    
    all_diseases = set()
    for results in all_results.values():
        all_diseases.update(results.keys())
    
    comparison_df = pd.DataFrame(index=sorted(all_diseases))
    for horizon_name, results in all_results.items():
        comparison_df[f'{horizon_name}_AUC'] = [results.get(d, {}).get('auc', np.nan) for d in comparison_df.index]
        comparison_df[f'{horizon_name}_CI_lower'] = [results.get(d, {}).get('ci_lower', np.nan) for d in comparison_df.index]
        comparison_df[f'{horizon_name}_CI_upper'] = [results.get(d, {}).get('ci_upper', np.nan) for d in comparison_df.index]
    
    comparison_file = output_dir / 'comparison_all_horizons.csv'
    comparison_df.to_csv(comparison_file)
    print(f"✓ Saved combined comparison to {comparison_file}")
    
    print(f"\n{'='*80}")
    print("COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    for horizon_name in all_results.keys():
        print(f"  - {horizon_name}_results.csv")

if __name__ == '__main__':
    main()

