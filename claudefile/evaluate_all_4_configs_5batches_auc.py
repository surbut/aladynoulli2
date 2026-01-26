#!/usr/bin/env python3
"""
Evaluate AUCs for 5 different configurations using first 5 batches (50K patients).

Configurations:
1. Fixed kappa, free gamma (fixedk_freeg)
2. Fixed gamma, free kappa (fixedg_freek)
3. Free kappa and gamma (original)
4. Fixed gamma and kappa (fixedgk) - with regularized pooled values
5. Fixed gamma and kappa (fixedgk_nolr) - with UNREGULARIZED pooled values

For each configuration:
- Pool first 5 batches (0-50000)
- Evaluate static 10-year AUC
- Evaluate dynamic 1-year AUC
- Save results to CSV

Usage:
    python evaluate_all_4_configs_5batches_auc.py --n_bootstraps 100
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

# Configuration directories
CONFIG_DIRS = {
    'fixedk_freeg': '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_fixedk_freeg_vectorized/',
    'fixedg_freek': '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_fixedg_freek_vectorized/',
    'original': '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_correctedE_vectorized/',
    'fixedgk': '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_fixedgk_vectorized/',
    'fixedgk_nolr': '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_fixedgk_nolr_vectorized/'
}

def load_essentials():
    """Load model essentials including disease names"""
    essentials_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/model_essentials.pt'
    essentials = torch.load(essentials_path, weights_only=False)
    return essentials

def pool_first_5_batches(config_dir, batch_size=10000, n_batches=5):
    """
    Pool first 5 batches of pi predictions from a configuration directory.
    
    Args:
        config_dir: Directory containing pi batch files
        batch_size: Size of each batch (default: 10000)
        n_batches: Number of batches to pool (default: 5)
    
    Returns:
        pi_pooled: Pooled pi tensor (N, T, D) where N = n_batches * batch_size
    """
    config_path = Path(config_dir)
    pi_batches = []
    
    print(f"  Pooling {n_batches} batches from {config_path.name}...")
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size
        
        pi_file = config_path / f'pi_enroll_fixedphi_sex_{start_idx}_{end_idx}.pt'
        
        if not pi_file.exists():
            print(f"    ⚠️  Batch {batch_idx} not found: {pi_file.name}")
            continue
        
        pi_batch = torch.load(pi_file, map_location='cpu', weights_only=False)
        pi_batches.append(pi_batch)
        print(f"    ✓ Batch {batch_idx}: {pi_batch.shape}")
    
    if not pi_batches:
        raise ValueError(f"No pi files found in {config_dir}")
    
    # Concatenate all batches
    pi_pooled = torch.cat(pi_batches, dim=0)
    print(f"  ✓ Pooled shape: {pi_pooled.shape}")
    
    return pi_pooled

def evaluate_config(config_name, config_dir, pi_pooled, Y_eval, E_eval, pce_df_eval, 
                   disease_names, n_bootstraps=100):
    """
    Evaluate AUCs for a single configuration.
    
    Returns:
        static_results: DataFrame with static 10-year AUC results
        dynamic_results: DataFrame with dynamic 1-year AUC results
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING: {config_name.upper()}")
    print(f"{'='*80}")
    
    # ============================================================================
    # STATIC 10-YEAR AUC
    # ============================================================================
    print("\nEvaluating static 10-year AUC...")
    
    results_static_10yr = evaluate_major_diseases_wsex_with_bootstrap_from_pi(
        pi=pi_pooled,
        Y_100k=Y_eval,
        E_100k=E_eval,
        disease_names=disease_names,
        pce_df=pce_df_eval,
        n_bootstraps=n_bootstraps,
        follow_up_duration_years=10
    )
    
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
            'config': config_name,
            'n_patients': pi_pooled.shape[0]
        })
    
    static_df = pd.DataFrame(static_results)
    
    # ============================================================================
    # DYNAMIC 1-YEAR AUC (WASHOUT 0, FROM ENROLLMENT)
    # ============================================================================
    print("Evaluating dynamic 1-year AUC...")
    
    results_1yr = evaluate_major_diseases_wsex_with_bootstrap_dynamic_1year_different_start_end_numeric_sex(
        pi=pi_pooled,
        Y_100k=Y_eval,
        E_100k=E_eval,
        disease_names=disease_names,
        pce_df=pce_df_eval,
        n_bootstraps=n_bootstraps,
        follow_up_duration_years=1,
        start_offset=0  # From enrollment (washout 0)
    )
    
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
            'config': config_name,
            'n_patients': pi_pooled.shape[0]
        })
    
    dynamic_df = pd.DataFrame(dynamic_results)
    
    print(f"  ✓ Static 10-year: {len(static_df)} diseases")
    print(f"  ✓ Dynamic 1-year: {len(dynamic_df)} diseases")
    
    return static_df, dynamic_df

def main():
    parser = argparse.ArgumentParser(description='Evaluate AUCs for 5 configurations using first 5 batches')
    parser.add_argument('--n_bootstraps', type=int, default=100,
                       help='Number of bootstrap iterations for AUC CI')
    parser.add_argument('--batch_size', type=int, default=10000,
                       help='Size of each batch (default: 10000)')
    parser.add_argument('--n_batches', type=int, default=5,
                       help='Number of batches to pool (default: 5)')
    parser.add_argument('--output_dir', type=str,
                       default='/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("="*80)
    print("EVALUATING ALL 5 CONFIGURATIONS - FIRST 5 BATCHES (50K PATIENTS)")
    print("="*80)
    print(f"Batch size: {args.batch_size}")
    print(f"Number of batches: {args.n_batches}")
    print(f"Total patients: {args.batch_size * args.n_batches:,}")
    print(f"Bootstrap iterations: {args.n_bootstraps}")
    print("="*80)
    
    # Load essentials
    essentials = load_essentials()
    disease_names = essentials['disease_names']
    
    # Load full data files
    print("\nLoading data files...")
    Y_full = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/Y_tensor.pt', 
                       map_location='cpu', weights_only=False)
    E_full = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/E_enrollment_full.pt', 
                       map_location='cpu', weights_only=False)
    pce_df_full = pd.read_csv('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pce_prevent_full.csv')
    
    # Subset to first 5 batches (50K patients)
    N_eval = args.batch_size * args.n_batches
    Y_eval = Y_full[:N_eval]
    E_eval = E_full[:N_eval]
    pce_df_eval = pce_df_full.iloc[:N_eval].reset_index(drop=True)
    
    print(f"✓ Loaded Y: {Y_eval.shape}, E: {E_eval.shape}, pce_df: {len(pce_df_eval)} rows")
    
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
    
    # Evaluate each configuration
    all_static_results = []
    all_dynamic_results = []
    
    for config_name, config_dir in CONFIG_DIRS.items():
        try:
            # Pool first 5 batches
            print(f"\n{'='*80}")
            print(f"CONFIGURATION: {config_name.upper()}")
            print(f"Directory: {config_dir}")
            print(f"{'='*80}")
            
            pi_pooled = pool_first_5_batches(config_dir, args.batch_size, args.n_batches)
            
            # Check shape compatibility
            if pi_pooled.shape[0] != N_eval:
                print(f"⚠️  WARNING: pi has {pi_pooled.shape[0]} patients, expected {N_eval}")
                # Use minimum
                N_actual = min(pi_pooled.shape[0], N_eval)
                pi_pooled = pi_pooled[:N_actual]
                Y_config = Y_eval[:N_actual]
                E_config = E_eval[:N_actual]
                pce_df_config = pce_df_eval.iloc[:N_actual].reset_index(drop=True)
            else:
                Y_config = Y_eval
                E_config = E_eval
                pce_df_config = pce_df_eval
            
            # Evaluate
            static_df, dynamic_df = evaluate_config(
                config_name, config_dir, pi_pooled, Y_config, E_config, pce_df_config,
                disease_names, args.n_bootstraps
            )
            
            all_static_results.append(static_df)
            all_dynamic_results.append(dynamic_df)
            
        except Exception as e:
            print(f"✗ ERROR evaluating {config_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Combine all results
    if all_static_results:
        combined_static = pd.concat(all_static_results, ignore_index=True)
        combined_dynamic = pd.concat(all_dynamic_results, ignore_index=True)
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        
        static_output = os.path.join(args.output_dir, 'all_5_configs_5batches_static_10yr_auc_results.csv')
        dynamic_output = os.path.join(args.output_dir, 'all_5_configs_5batches_dynamic_1yr_auc_results.csv')
        
        combined_static.to_csv(static_output, index=False)
        combined_dynamic.to_csv(dynamic_output, index=False)
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Evaluated {len(CONFIG_DIRS)} configurations")
        print(f"Static 10-year AUC: {len(combined_static)} disease-config combinations")
        print(f"Dynamic 1-year AUC: {len(combined_dynamic)} disease-config combinations")
        print(f"\nResults saved to:")
        print(f"  - {static_output}")
        print(f"  - {dynamic_output}")
        print("="*80)
        print("COMPLETED")
        print("="*80)
    else:
        print("\n✗ No results to save!")


if __name__ == '__main__':
    main()
