#!/usr/bin/env python3
"""
Generate washout predictions for 10-year and 30-year horizons using pre-computed pi tensor.

This script processes all patients at once using the full pi tensor.
Only runs for pooled_retrospective approach.

Usage:
    python generate_washout_time_horizons.py --n_bootstraps 100
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
from fig5utils import evaluate_major_diseases_wsex_with_bootstrap_dynamic_withwashout_from_pi, evaluate_major_diseases_wsex_with_bootstrap_withwashout_from_pi

# Load essentials (disease names, etc.)
def load_essentials():
    """Load model essentials including disease names"""
    essentials_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/model_essentials.pt'
    essentials = torch.load(essentials_path, weights_only=False)
    return essentials

def main():
    parser = argparse.ArgumentParser(description='Generate washout predictions for 10yr and 30yr horizons from pre-computed pi')
    parser.add_argument('--n_bootstraps', type=int, default=100,
                       help='Number of bootstrap iterations')
    parser.add_argument('--output_dir', type=str,
                       default='/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks/results/washout_time_horizons/',
                       help='Output directory for results')
    parser.add_argument('--washout_years', type=int, default=1,
                       help='Washout period in years (default: 1)')
    
    args = parser.parse_args()
    
    # Only pooled_retrospective approach
    approach_name = 'pooled_retrospective'
    pi_path = '/Users/sarahurbut/Downloads/pi_full_400k.pt'
    
    # Create output directory
    output_dir = Path(args.output_dir) / approach_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f"GENERATING WASHOUT TIME HORIZON PREDICTIONS: {approach_name.upper()}")
    print("="*80)
    print(f"Pi tensor: {pi_path}")
    print(f"Washout period: {args.washout_years} years")
    print(f"Horizons: 10yr, 30yr")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    pi_full = torch.load(pi_path, weights_only=False)
    Y_full = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/Y_tensor.pt', weights_only=False)
    E_full = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/E_enrollment_full.pt', weights_only=False)
    
    # Load essentials
    essentials = load_essentials()
    disease_names = essentials['disease_names']
    
    # Load pce_df
    pce_df_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pce_prevent_full.csv'
    pce_df_full = pd.read_csv(pce_df_path)
    
    # Cap at 400K patients
    N = min(400000, pi_full.shape[0], Y_full.shape[0], E_full.shape[0], len(pce_df_full))
    print(f"\nCapping at {N:,} patients (from {pi_full.shape[0]:,} available)")
    
    pi_full = pi_full[:N]
    Y_full = Y_full[:N]
    E_full = E_full[:N]
    pce_df_full = pce_df_full.iloc[:N].reset_index(drop=True)
    
    # Verify sizes match
    if not (pi_full.shape[0] == Y_full.shape[0] == E_full.shape[0] == len(pce_df_full) == N):
        raise ValueError(f"Size mismatch after subsetting! pi: {pi_full.shape[0]}, Y: {Y_full.shape[0]}, E: {E_full.shape[0]}, pce_df: {len(pce_df_full)}")
    
    print(f"✓ Loaded {N:,} patients")
    print(f"  Pi shape: {pi_full.shape}")
    print(f"  Y shape: {Y_full.shape}")
    print(f"  E shape: {E_full.shape}")
    
    # Check if results already exist
    expected_files = [
        output_dir / f'washout_{args.washout_years}yr_10yr_dynamic_results.csv',
        output_dir / f'washout_{args.washout_years}yr_30yr_dynamic_results.csv',
        output_dir / f'washout_{args.washout_years}yr_10yr_static_results.csv'
    ]
    comparison_file = output_dir / f'washout_{args.washout_years}yr_comparison_all_horizons.csv'
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
    
    for horizon_years in [10, 30]:
        output_file = output_dir / f'washout_{args.washout_years}yr_{horizon_years}yr_dynamic_results.csv'
        
        # Skip if this specific result already exists
        if output_file.exists():
            print(f"\n{'='*80}")
            print(f"SKIPPING {horizon_years}-YEAR DYNAMIC (already exists)")
            print(f"{'='*80}")
            print(f"  File exists: {output_file}")
            continue
        
        print(f"\n{'='*80}")
        print(f"PROCESSING {horizon_years}-YEAR HORIZON WITH {args.washout_years}-YEAR WASHOUT")
        print(f"{'='*80}")
        
        print(f"Evaluating dynamic {horizon_years}-year predictions with {args.washout_years}-year washout...")
        results = evaluate_major_diseases_wsex_with_bootstrap_dynamic_withwashout_from_pi(
            pi=pi_full,
            Y_100k=Y_full,
            E_100k=E_full,
            disease_names=disease_names,
            pce_df=pce_df_full,
            washout_years=args.washout_years,
            n_bootstraps=args.n_bootstraps,
            follow_up_duration_years=horizon_years,
            patient_indices=None
        )
        
        all_results[f'{horizon_years}yr'] = results
        
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
        
        output_file = output_dir / f'washout_{args.washout_years}yr_{horizon_years}yr_dynamic_results.csv'
        results_df.to_csv(output_file)
        print(f"✓ Saved results to {output_file}")
    
    # Generate static 10-year predictions (1-year score for 10-year outcome)
    static_output_file = output_dir / f'washout_{args.washout_years}yr_10yr_static_results.csv'
    
    if static_output_file.exists():
        print(f"\n{'='*80}")
        print(f"SKIPPING STATIC 10-YEAR (already exists)")
        print(f"{'='*80}")
        print(f"  File exists: {static_output_file}")
    else:
        print(f"\n{'='*80}")
        print(f"PROCESSING STATIC 10-YEAR WITH {args.washout_years}-YEAR WASHOUT")
        print(f"{'='*80}")
        
        print(f"Evaluating static 10-year predictions (1-year score) with {args.washout_years}-year washout...")
        static_results = evaluate_major_diseases_wsex_with_bootstrap_withwashout_from_pi(
            pi=pi_full,
            Y_100k=Y_full,
            E_100k=E_full,
            disease_names=disease_names,
            pce_df=pce_df_full,
            washout_years=args.washout_years,
            n_bootstraps=args.n_bootstraps,
            follow_up_duration_years=10
        )
        
        all_results['10yr_static'] = static_results
        
        # Save static results
        static_results_df = pd.DataFrame({
            'Disease': list(static_results.keys()),
            'AUC': [r['auc'] for r in static_results.values()],
            'CI_lower': [r['ci_lower'] for r in static_results.values()],
            'CI_upper': [r['ci_upper'] for r in static_results.values()],
            'N_Events': [r['n_events'] for r in static_results.values()],
            'Event_Rate': [r['event_rate'] for r in static_results.values()]
        })
        static_results_df = static_results_df.set_index('Disease').sort_values('AUC', ascending=False)
        
        static_results_df.to_csv(static_output_file)
        print(f"✓ Saved results to {static_output_file}")
    
    # Create combined comparison file
    print(f"\n{'='*80}")
    print("CREATING COMBINED COMPARISON FILE")
    print(f"{'='*80}")
    
    comparison_data = []
    # Add dynamic horizons
    for horizon_years in [10, 30]:
        horizon_key = f'{horizon_years}yr'
        if horizon_key in all_results:
            for disease, metrics in all_results[horizon_key].items():
                comparison_data.append({
                    'Horizon': f'{horizon_years}yr_dynamic',
                    'Disease': disease,
                    'AUC': metrics['auc'],
                    'CI_lower': metrics['ci_lower'],
                    'CI_upper': metrics['ci_upper'],
                    'N_Events': metrics['n_events'],
                    'Event_Rate': metrics['event_rate']
                })
    # Add static 10yr
    if '10yr_static' in all_results:
        for disease, metrics in all_results['10yr_static'].items():
            comparison_data.append({
                'Horizon': '10yr_static',
                'Disease': disease,
                'AUC': metrics['auc'],
                'CI_lower': metrics['ci_lower'],
                'CI_upper': metrics['ci_upper'],
                'N_Events': metrics['n_events'],
                'Event_Rate': metrics['event_rate']
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.pivot_table(
        index='Disease',
        columns='Horizon',
        values=['AUC', 'CI_lower', 'CI_upper', 'N_Events', 'Event_Rate']
    )
    comparison_df.columns = [f'{col[1]}_{col[0]}' for col in comparison_df.columns]
    # Sort by 10yr_dynamic AUC if available, otherwise by first available AUC column
    sort_col = '10yr_dynamic_AUC' if '10yr_dynamic_AUC' in comparison_df.columns else comparison_df.columns[0]
    comparison_df = comparison_df.sort_values(sort_col, ascending=False, na_position='last')
    
    comparison_file = output_dir / f'washout_{args.washout_years}yr_comparison_all_horizons.csv'
    comparison_df.to_csv(comparison_file)
    print(f"✓ Saved combined comparison to {comparison_file}")
    
    print(f"\n{'='*80}")
    print("WASHOUT TIME HORIZON PREDICTIONS COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_dir}")
    for horizon_years in [10, 30]:
        print(f"  - washout_{args.washout_years}yr_{horizon_years}yr_dynamic_results.csv")
    print(f"  - washout_{args.washout_years}yr_10yr_static_results.csv")
    print(f"  - washout_{args.washout_years}yr_comparison_all_horizons.csv")

if __name__ == '__main__':
    main()

