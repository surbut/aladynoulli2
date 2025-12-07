#!/usr/bin/env python3
"""
Compare pi tensors from different sources to check if they give the same results.

Tests:
1. pi_enroll_fixedphi_sex_0_10000.pt (from big run) vs 
   pi_enroll_fixedphi_age_offset_0_sex_0_10000_try2_withpcs_newrun_pooledall.pt (from offset run)
2. Both evaluated using the same evaluation function
"""

import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts')
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision')

from evaluatetdccode import (
    evaluate_major_diseases_wsex_with_bootstrap_dynamic_1year_different_start_end_numeric_sex,
    evaluate_major_diseases_rolling_1year_roc_curves
)
from clust_huge_amp import subset_data

def load_data_for_batch(start_idx=0, end_idx=10000):
    """Load Y, E, pce_df, essentials for a specific batch."""
    base_path = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
    
    print("Loading full data tensors...")
    Y = torch.load(base_path / 'Y_tensor.pt', weights_only=False)
    E = torch.load(base_path / 'E_matrix.pt', weights_only=False)
    G = torch.load(base_path / 'G_matrix.pt', weights_only=False)
    essentials = torch.load(base_path / 'model_essentials.pt', weights_only=False)
    
    print(f"Subsetting to batch {start_idx}-{end_idx}...")
    Y_batch, E_batch, G_batch, indices = subset_data(Y, E, G, start_index=start_idx, end_index=end_idx)
    
    # Load pce_df (use CSV to match washout script)
    pce_df_full = pd.read_csv('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pce_prevent_full.csv')
    
    # Convert Sex column to numeric if needed (to match washout script)
    # But keep 'Sex' column for rolling function which expects string format
    if 'Sex' in pce_df_full.columns and pce_df_full['Sex'].dtype == 'object':
        pce_df_full['sex'] = pce_df_full['Sex'].map({'Female': 0, 'Male': 1}).astype(int)
        # Keep 'Sex' column for rolling function
    elif 'sex' in pce_df_full.columns:
        # Convert numeric sex back to 'Sex' string for rolling function
        pce_df_full['Sex'] = pce_df_full['sex'].map({0: 'Female', 1: 'Male'}).astype(str)
    else:
        raise ValueError("Need 'Sex' or 'sex' column in pce_df")
    
    # Subset pce_df to batch indices
    pce_df_batch = pce_df_full.iloc[indices].reset_index(drop=True)
    
    return Y_batch, E_batch, pce_df_batch, essentials

def compare_pi_tensors():
    """Compare two pi tensors using the same evaluation function."""
    
    print("="*80)
    print("COMPARING PI TENSORS")
    print("="*80)
    
    # Load data
    print("\n1. Loading data for batch 0-10000...")
    Y_batch, E_batch, pce_df_batch, essentials = load_data_for_batch(0, 10000)
    disease_names = essentials['disease_names']
    
    # Also load E_enrollment_full for washout evaluation (to match generate_washout_predictions.py)
    E_enrollment_path = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/E_enrollment_full.pt')
    if E_enrollment_path.exists():
        print(f"  Also loading E_enrollment_full for comparison...")
        E_enrollment_full = torch.load(E_enrollment_path, weights_only=False)
        E_enrollment_batch = E_enrollment_full[:10000]
    else:
        print(f"  E_enrollment_full not found, using E_batch")
        E_enrollment_batch = E_batch
    
    # Load both pi tensors
    print("\n2. Loading pi tensors...")
    
    # Pi from big run (first 10K)
    pi_path_1 = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_correctedE_vectorized/pi_enroll_fixedphi_sex_FULL.pt')
    print(f"  Loading pi_1: {pi_path_1.name}")
    pi_full_1 = torch.load(pi_path_1, weights_only=False)
    pi_1 = pi_full_1[:10000]  # First 10K
    print(f"    Shape: {pi_1.shape}")
    
    # Pi from offset run (offset 0, batch 0-10K)
    pi_path_2 = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/age_offset_local_vectorized_E_corrected/pi_enroll_fixedphi_age_offset_0_sex_0_10000_try2_withpcs_newrun_pooledall.pt')
    print(f"  Loading pi_2: {pi_path_2.name}")
    pi_2 = torch.load(pi_path_2, weights_only=False)
    print(f"    Shape: {pi_2.shape}")
    
    # Check shapes match
    if pi_1.shape != pi_2.shape:
        print(f"\n⚠️  WARNING: Shape mismatch!")
        print(f"  pi_1: {pi_1.shape}")
        print(f"  pi_2: {pi_2.shape}")
        return
    
    # Check if they're identical
    print("\n3. Comparing pi tensors directly...")
    are_identical = torch.allclose(pi_1, pi_2, atol=1e-6)
    max_diff = (pi_1 - pi_2).abs().max().item()
    mean_diff = (pi_1 - pi_2).abs().mean().item()
    
    print(f"  Are identical (atol=1e-6): {are_identical}")
    print(f"  Max difference: {max_diff:.6e}")
    print(f"  Mean difference: {mean_diff:.6e}")
    
    # Evaluate both using washout evaluation function (0-year washout)
    print("\n4. Evaluating both pi tensors with 0-year washout...")
    print("   Using: evaluate_major_diseases_wsex_with_bootstrap_dynamic_1year_different_start_end_numeric_sex")
    
    print("\n   Evaluating pi_1 (from big run) with E_enrollment...")
    results_1 = evaluate_major_diseases_wsex_with_bootstrap_dynamic_1year_different_start_end_numeric_sex(
        pi=pi_1,
        Y_100k=Y_batch,
        E_100k=E_enrollment_batch,  # Use E_enrollment to match washout script
        disease_names=disease_names,
        pce_df=pce_df_batch,
        n_bootstraps=100,
        follow_up_duration_years=1,
        start_offset=0
    )
    
    print("\n   Evaluating pi_2 (from offset run) with E_enrollment...")
    results_2 = evaluate_major_diseases_wsex_with_bootstrap_dynamic_1year_different_start_end_numeric_sex(
        pi=pi_2,
        Y_100k=Y_batch,
        E_100k=E_enrollment_batch,  # Use E_enrollment to match washout script
        disease_names=disease_names,
        pce_df=pce_df_batch,
        n_bootstraps=100,
        follow_up_duration_years=1,
        start_offset=0
    )
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    comparison_data = []
    for disease in results_1.keys():
        if disease in results_2:
            auc_1 = results_1[disease]['auc']
            auc_2 = results_2[disease]['auc']
            diff = auc_1 - auc_2
            
            comparison_data.append({
                'Disease': disease,
                'AUC_pi1': auc_1,
                'AUC_pi2': auc_2,
                'Difference': diff,
                'CI_lower_1': results_1[disease]['ci_lower'],
                'CI_upper_1': results_1[disease]['ci_upper'],
                'CI_lower_2': results_2[disease]['ci_lower'],
                'CI_upper_2': results_2[disease]['ci_upper'],
                'N_Events_1': results_1[disease]['n_events'],
                'N_Events_2': results_2[disease]['n_events']
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Difference', key=abs, ascending=False)
    
    print("\nTop differences:")
    print(comparison_df.head(20).to_string(index=False))
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Mean absolute difference: {comparison_df['Difference'].abs().mean():.6f}")
    print(f"Max absolute difference: {comparison_df['Difference'].abs().max():.6f}")
    print(f"Min absolute difference: {comparison_df['Difference'].abs().min():.6f}")
    print(f"Std of differences: {comparison_df['Difference'].std():.6f}")
    
    # Check ASCVD specifically
    if 'ASCVD' in comparison_df['Disease'].values:
        ascvd_row = comparison_df[comparison_df['Disease'] == 'ASCVD'].iloc[0]
        print(f"\nASCVD comparison:")
        print(f"  pi_1 AUC: {ascvd_row['AUC_pi1']:.6f} ({ascvd_row['CI_lower_1']:.6f}-{ascvd_row['CI_upper_1']:.6f})")
        print(f"  pi_2 AUC: {ascvd_row['AUC_pi2']:.6f} ({ascvd_row['CI_lower_2']:.6f}-{ascvd_row['CI_upper_2']:.6f})")
        print(f"  Difference: {ascvd_row['Difference']:.6f}")
    
    # Save comparison
    output_file = Path('pi_tensor_comparison_results.csv')
    comparison_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved comparison to: {output_file}")
    
    # Now test both evaluation functions on the same pi tensor (10K)
    print("\n" + "="*80)
    print("TESTING BOTH EVALUATION FUNCTIONS ON SAME PI TENSOR (10K)")
    print("="*80)
    print("\nUsing pi_2 (10K from offset run) to test both functions...")
    
    # Test 1: Washout function (bootstrap)
    print("\n1. Testing with evaluate_major_diseases_wsex_with_bootstrap_dynamic_1year_different_start_end_numeric_sex...")
    results_washout = evaluate_major_diseases_wsex_with_bootstrap_dynamic_1year_different_start_end_numeric_sex(
        pi=pi_2,
        Y_100k=Y_batch,
        E_100k=E_enrollment_batch,
        disease_names=disease_names,
        pce_df=pce_df_batch,
        n_bootstraps=100,
        follow_up_duration_years=1,
        start_offset=0
    )
    
    # Test 2: Rolling function (no bootstrap, just AUC)
    print("\n2. Testing with evaluate_major_diseases_rolling_1year_roc_curves...")
    # This function expects pi_batches as a list (one per offset)
    pi_batches_list = [pi_2]  # Just offset 0
    results_rolling = evaluate_major_diseases_rolling_1year_roc_curves(
        pi_batches=pi_batches_list,
        Y_100k=Y_batch,
        E_100k=E_enrollment_batch,
        disease_names=disease_names,
        pce_df=pce_df_batch,
        patient_indices=None,
        plot_group=None  # Don't plot
    )
    
    # Compare the two functions
    print("\n" + "="*80)
    print("COMPARISON: WASHOUT FUNCTION vs ROLLING FUNCTION")
    print("="*80)
    
    function_comparison = []
    for disease in results_washout.keys():
        if disease in results_rolling and len(results_rolling[disease]) > 0:
            # Washout function returns dict with 'auc'
            auc_washout = results_washout[disease]['auc']
            
            # Rolling function returns list of tuples (fpr, tpr, thresholds, auc) for each offset
            # For offset 0, get the first element
            roc_data = results_rolling[disease][0]
            if roc_data is not None and len(roc_data) == 4:
                fpr, tpr, thresholds, auc_rolling = roc_data
                
                # Handle None AUC values
                if auc_rolling is None:
                    print(f"  Warning: {disease} has None AUC in rolling function, skipping")
                    continue
                
                diff = auc_washout - auc_rolling
                
                function_comparison.append({
                    'Disease': disease,
                    'AUC_washout_func': auc_washout,
                    'AUC_rolling_func': auc_rolling,
                    'Difference': diff,
                    'CI_lower_washout': results_washout[disease]['ci_lower'],
                    'CI_upper_washout': results_washout[disease]['ci_upper'],
                    'N_Events_washout': results_washout[disease]['n_events']
                })
    
    func_comparison_df = pd.DataFrame(function_comparison)
    func_comparison_df = func_comparison_df.sort_values('Difference', key=abs, ascending=False)
    
    print("\nTop differences between evaluation functions:")
    print(func_comparison_df.head(20).to_string(index=False))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: EVALUATION FUNCTION COMPARISON")
    print("="*80)
    print(f"Mean absolute difference: {func_comparison_df['Difference'].abs().mean():.6f}")
    print(f"Max absolute difference: {func_comparison_df['Difference'].abs().max():.6f}")
    print(f"Min absolute difference: {func_comparison_df['Difference'].abs().min():.6f}")
    print(f"Std of differences: {func_comparison_df['Difference'].std():.6f}")
    
    # Check ASCVD specifically
    if 'ASCVD' in func_comparison_df['Disease'].values:
        ascvd_row = func_comparison_df[func_comparison_df['Disease'] == 'ASCVD'].iloc[0]
        print(f"\nASCVD comparison:")
        print(f"  Washout function AUC: {ascvd_row['AUC_washout_func']:.6f} ({ascvd_row['CI_lower_washout']:.6f}-{ascvd_row['CI_upper_washout']:.6f})")
        print(f"  Rolling function AUC: {ascvd_row['AUC_rolling_func']:.6f}")
        print(f"  Difference: {ascvd_row['Difference']:.6f}")
    
    # Save function comparison
    func_output_file = Path('evaluation_function_comparison_results.csv')
    func_comparison_df.to_csv(func_output_file, index=False)
    print(f"\n✓ Saved function comparison to: {func_output_file}")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)

if __name__ == "__main__":
    compare_pi_tensors()

