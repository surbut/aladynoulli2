#!/usr/bin/env python3
"""
Evaluate AUCs for fixed gamma/kappa predictions on full 400k dataset.

This script:
1. Loads full 400k predictions (pi_enroll_fixedphi_sex_FULL.pt)
2. Runs static 10-year AUC evaluation
3. Runs 1-year dynamic AUC evaluation (washout 0, from enrollment)
4. Saves results to CSV files

Usage:
    python evaluate_fixedgk_full_400k_auc.py --n_bootstraps 100
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
    parser = argparse.ArgumentParser(description='Evaluate AUCs for fixed gamma/kappa predictions on full 400k')
    parser.add_argument('--pi_file', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_fixedgk_vectorized/pi_enroll_fixedphi_sex_FULL.pt',
                       help='Path to full 400k pi predictions')
    parser.add_argument('--n_bootstraps', type=int, default=100,
                       help='Number of bootstrap iterations for AUC CI')
    parser.add_argument('--max_patients', type=int, default=400000,
                       help='Maximum number of patients to use (default: 400000)')
    parser.add_argument('--output_dir', type=str,
                       default='/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("="*80)
    print("EVALUATING FIXED GAMMA/KAPPA PREDICTIONS - FULL 400K DATASET")
    print("="*80)
    print(f"Pi file: {args.pi_file}")
    print(f"Max patients: {args.max_patients}")
    print(f"Bootstrap iterations: {args.n_bootstraps}")
    print("="*80)
    
    # Load essentials
    essentials = load_essentials()
    disease_names = essentials['disease_names']
    
    # Load full data
    print("\nLoading full data files...")
    Y_full = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/Y_tensor.pt', 
                       map_location='cpu', weights_only=False)
    E_full = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/E_enrollment_full.pt', 
                       map_location='cpu', weights_only=False)
    pce_df_full = pd.read_csv('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pce_prevent_full.csv')
    
    # Load predictions
    print(f"\nLoading predictions from {args.pi_file}...")
    pi_full = torch.load(args.pi_file, map_location='cpu', weights_only=False)
    print(f"Loaded pi: {pi_full.shape}")
    
    # Subset to max_patients
    N_pi = min(pi_full.shape[0], args.max_patients)
    N_data = min(Y_full.shape[0], args.max_patients)
    
    if N_pi != N_data:
        print(f"⚠️  WARNING: pi has {pi_full.shape[0]} patients, data has {Y_full.shape[0]} patients")
        N = min(N_pi, N_data)
    else:
        N = N_pi
    
    print(f"Using {N:,} patients for evaluation")
    
    pi_eval = pi_full[:N]
    Y_eval = Y_full[:N]
    E_eval = E_full[:N]
    pce_df_eval = pce_df_full.iloc[:N].reset_index(drop=True)
    
    # Ensure 'Sex' and 'sex' columns exist
    if 'Sex' not in pce_df_eval.columns:
        if 'sex' in pce_df_eval.columns:
            if pce_df_eval['sex'].dtype in [int, float]:
                pce_df_eval['Sex'] = pce_df_eval['sex'].map({0: 'Female', 1: 'Male', 1: 'Female', 2: 'Male'}).fillna('Unknown')
            else:
                pce_df_eval['Sex'] = pce_df_eval['sex']
        else:
            raise ValueError("Neither 'Sex' nor 'sex' column found in pce_df")
    
    if 'sex' not in pce_df_eval.columns:
        if 'Sex' in pce_df_eval.columns:
            pce_df_eval['sex'] = pce_df_eval['Sex'].map({'Female': 0, 'Male': 1, 'F': 0, 'M': 1}).fillna(-1)
        else:
            raise ValueError("Neither 'Sex' nor 'sex' column found in pce_df")
    
    if 'age' not in pce_df_eval.columns:
        if 'Age' in pce_df_eval.columns:
            pce_df_eval['age'] = pce_df_eval['Age']
        else:
            raise ValueError("Neither 'age' nor 'Age' column found in pce_df")
    
    print(f"Data shapes: pi={pi_eval.shape}, Y={Y_eval.shape}, E={E_eval.shape}, pce_df={len(pce_df_eval)}")
    
    # ============================================================================
    # STATIC 10-YEAR AUC
    # ============================================================================
    print("\n" + "="*80)
    print("EVALUATING STATIC 10-YEAR AUC")
    print("="*80)
    
    results_static_10yr = evaluate_major_diseases_wsex_with_bootstrap_from_pi(
        pi=pi_eval,
        Y_100k=Y_eval,
        E_100k=E_eval,
        disease_names=disease_names,
        pce_df=pce_df_eval,
        n_bootstraps=args.n_bootstraps,
        follow_up_duration_years=10
    )
    
    # Save static 10-year results
    static_results = []
    for disease, metrics in results_static_10yr.items():
        static_results.append({
            'disease': disease,
            'auc': metrics.get('auc', np.nan),
            'ci_lower': metrics.get('ci_lower', np.nan),
            'ci_upper': metrics.get('ci_upper', np.nan),
            'n_events': metrics.get('n_events', np.nan),
            'event_rate': metrics.get('event_rate', np.nan),
            'horizon': 'static_10yr',
            'n_patients': N
        })
    
    static_df = pd.DataFrame(static_results)
    static_output = os.path.join(args.output_dir, 'fixedgk_static_10yr_auc_results.csv')
    os.makedirs(args.output_dir, exist_ok=True)
    static_df.to_csv(static_output, index=False)
    print(f"\n✓ Saved static 10-year results to: {static_output}")
    
    # ============================================================================
    # DYNAMIC 1-YEAR AUC (WASHOUT 0, FROM ENROLLMENT)
    # ============================================================================
    print("\n" + "="*80)
    print("EVALUATING DYNAMIC 1-YEAR AUC (WASHOUT 0, FROM ENROLLMENT)")
    print("="*80)
    
    results_1yr = evaluate_major_diseases_wsex_with_bootstrap_dynamic_1year_different_start_end_numeric_sex(
        pi=pi_eval,
        Y_100k=Y_eval,
        E_100k=E_eval,
        disease_names=disease_names,
        pce_df=pce_df_eval,
        n_bootstraps=args.n_bootstraps,
        follow_up_duration_years=1,
        start_offset=0  # From enrollment (washout 0)
    )
    
    # Save 1-year dynamic results
    dynamic_results = []
    for disease, metrics in results_1yr.items():
        dynamic_results.append({
            'disease': disease,
            'auc': metrics.get('auc', np.nan),
            'ci_lower': metrics.get('ci_lower', np.nan),
            'ci_upper': metrics.get('ci_upper', np.nan),
            'n_events': metrics.get('n_events', np.nan),
            'event_rate': metrics.get('event_rate', np.nan),
            'horizon': 'dynamic_1yr',
            'washout': 0,
            'n_patients': N
        })
    
    dynamic_df = pd.DataFrame(dynamic_results)
    dynamic_output = os.path.join(args.output_dir, 'fixedgk_dynamic_1yr_auc_results.csv')
    dynamic_df.to_csv(dynamic_output, index=False)
    print(f"\n✓ Saved dynamic 1-year results to: {dynamic_output}")
    
    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Static 10-year AUC: {len(static_df)} diseases evaluated")
    print(f"Dynamic 1-year AUC: {len(dynamic_df)} diseases evaluated")
    print(f"Results saved to:")
    print(f"  - {static_output}")
    print(f"  - {dynamic_output}")
    print("="*80)
    print("COMPLETED")
    print("="*80)


if __name__ == '__main__':
    main()
