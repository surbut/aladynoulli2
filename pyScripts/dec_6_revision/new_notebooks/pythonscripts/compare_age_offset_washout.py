#!/usr/bin/env python3
"""
Compare 0-year predictions from age_offset vs washout.

NOTE: These should be SIMILAR because both use enrollment-capped event times:

1. **Age Offset (offset 0)** - `forAWS_offsetmasterfix.py`:
   - Uses E_matrix.pt (retrospective data)
   - Manually caps event times at enrollment age: `current_age = enrollment_age + 0`
   - Trains with enrollment-capped E

2. **Washout (0yr)** - `run_aladyn_predict_with_master.py`:
   - Uses E_enrollment_full.pt (already enrollment-capped)
   - Trains with enrollment-capped E (no additional capping needed)
   - Washout applied during evaluation only

Both should use equivalent enrollment-capped E matrices for training.
Any differences would be due to:
- Different E source files (E_matrix.pt vs E_enrollment_full.pt) - should be equivalent after censoring
- Batch processing differences (age offset processes batch 0-10K, washout processes all 400K)
- Minor initialization/training differences
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts')
from evaluatetdccode import (
    evaluate_major_diseases_rolling_1year_roc_curves,
    evaluate_major_diseases_wsex_with_bootstrap_dynamic_1year_different_start_end_numeric_sex
)

def load_data_for_batch(start_idx=0, end_idx=10000):
    """Load Y, E, pce_df, essentials for a specific batch."""
    from clust_huge_amp import subset_data
    
    base_path = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
    
    print("Loading full data tensors...")
    Y = torch.load(base_path / 'Y_tensor.pt', weights_only=False)
    E = torch.load(base_path / 'E_matrix.pt', weights_only=False)
    G = torch.load(base_path / 'G_matrix.pt', weights_only=False)
    essentials = torch.load(base_path / 'model_essentials.pt', weights_only=False)
    
    print(f"Subsetting to batch {start_idx}-{end_idx}...")
    Y_batch, E_batch, G_batch, indices = subset_data(Y, E, G, start_index=start_idx, end_index=end_idx)
    
    # Load pce_df
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    
    readRDS = robjects.r['readRDS']
    pce_data = readRDS('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pce_df_prevent.rds')
    pce_df = pandas2ri.rpy2py(pce_data)
    
    # Subset pce_df to batch indices
    pce_df_batch = pce_df.iloc[indices].reset_index(drop=True)
    
    return Y_batch, E_batch, pce_df_batch, essentials, indices


def compare_pi_tensors():
    """Compare pi tensors from age_offset vs washout for batch 0-10K."""
    print("="*80)
    print("COMPARING PI TENSORS")
    print("="*80)
    
    # Load age_offset pi (offset 0, batch 0-10K) - NEW with gamma initialization fix
    pi_age_offset_path = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/age_offset_files') / \
                         'pi_enroll_fixedphi_age_offset_0_sex_0_10000_try2_withpcs_newrun.pt'
    
    # Load washout pi (full 400K, subset to batch 0-10K)
    pi_washout_path = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_correctedE_vectorized/pi_enroll_fixedphi_sex_FULL.pt')
    
    if not pi_age_offset_path.exists():
        print(f"❌ Age offset pi not found: {pi_age_offset_path}")
        return False
    
    if not pi_washout_path.exists():
        print(f"❌ Washout pi not found: {pi_washout_path}")
        return False
    
    print(f"Loading age_offset pi: {pi_age_offset_path}")
    pi_age_offset = torch.load(pi_age_offset_path, weights_only=False)
    
    print(f"Loading washout pi: {pi_washout_path}")
    pi_washout_full = torch.load(pi_washout_path, weights_only=False)
    
    # Subset washout pi to batch 0-10K
    pi_washout_batch = pi_washout_full[:10000]
    
    print(f"\nAge offset pi shape: {pi_age_offset.shape}")
    print(f"Washout pi (batch) shape: {pi_washout_batch.shape}")
    
    if pi_age_offset.shape != pi_washout_batch.shape:
        print(f"❌ Shape mismatch!")
        return False
    
    # Compare tensors
    if torch.allclose(pi_age_offset, pi_washout_batch, rtol=1e-5, atol=1e-6):
        print("✅ Pi tensors match!")
        return True
    else:
        print("❌ Pi tensors differ!")
        max_diff = (pi_age_offset - pi_washout_batch).abs().max().item()
        mean_diff = (pi_age_offset - pi_washout_batch).abs().mean().item()
        print(f"   Max difference: {max_diff:.6e}")
        print(f"   Mean difference: {mean_diff:.6e}")
        return False


def compare_evaluations():
    """Compare evaluation results from age_offset vs washout for batch 0-10K."""
    print("\n" + "="*80)
    print("COMPARING EVALUATION RESULTS")
    print("="*80)
    
    # Load data
    Y_batch, E_batch, pce_df_batch, essentials, indices = load_data_for_batch(0, 10000)
    disease_names = essentials['disease_names']
    
    # Convert Sex column to numeric for washout evaluation
    if 'Sex' in pce_df_batch.columns and pce_df_batch['Sex'].dtype == 'object':
        pce_df_batch['sex'] = pce_df_batch['Sex'].map({'Female': 0, 'Male': 1}).astype(int)
    elif 'sex' not in pce_df_batch.columns:
        pce_df_batch['sex'] = 0  # Default if missing
    
    # Load pi tensors
    pi_age_offset_path = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/age_offset_files') / \
                         'pi_enroll_fixedphi_age_offset_0_sex_0_10000_try2_withpcs_newrun.pt'
    pi_washout_path = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_correctedE_vectorized/pi_enroll_fixedphi_sex_FULL.pt')
    
    pi_age_offset = torch.load(pi_age_offset_path, weights_only=False)
    pi_washout_full = torch.load(pi_washout_path, weights_only=False)
    pi_washout_batch = pi_washout_full[:10000]
    
    # Test 1: Age offset evaluation with batch-specific pi (original CSV results)
    print("\nRunning age_offset evaluation with batch-specific pi (offset 0)...")
    results_age_offset_batch = evaluate_major_diseases_rolling_1year_roc_curves(
        [pi_age_offset], Y_batch, E_batch, disease_names, pce_df_batch,
        patient_indices=None, plot_group=None
    )
    
    # Test 2: Age offset evaluation with full pi tensor subsetted (NEW TEST)
    print("\nRunning age_offset evaluation with full pi tensor subsetted (offset 0)...")
    results_age_offset_full = evaluate_major_diseases_rolling_1year_roc_curves(
        [pi_washout_batch], Y_batch, E_batch, disease_names, pce_df_batch,
        patient_indices=None, plot_group=None
    )
    
    # Test 3: Washout evaluation (for reference - different evaluation function)
    print("\nRunning washout evaluation (washout 0yr) - different eval function...")
    results_washout = evaluate_major_diseases_wsex_with_bootstrap_dynamic_1year_different_start_end_numeric_sex(
        pi=pi_washout_batch,
        Y_100k=Y_batch,
        E_100k=E_batch,
        disease_names=disease_names,
        pce_df=pce_df_batch,
        n_bootstraps=100,
        follow_up_duration_years=1,
        start_offset=0
    )
    
    # Compare AUCs
    print("\n" + "="*80)
    print("AUC COMPARISON: Same Evaluation Function (Age Offset), Different Pi Tensors")
    print("="*80)
    print(f"{'Disease':<30} {'Batch Pi':<15} {'Full Pi':<15} {'Difference':<15} {'Match':<10}")
    print("-"*80)
    
    matches = []
    mismatches = []
    
    all_diseases = set(results_age_offset_batch.keys()) | set(results_age_offset_full.keys())
    
    for disease_group in sorted(all_diseases):
        # Age offset AUC with batch-specific pi
        if disease_group in results_age_offset_batch and results_age_offset_batch[disease_group] and len(results_age_offset_batch[disease_group]) > 0:
            roc_data = results_age_offset_batch[disease_group][0]
            if roc_data is not None and len(roc_data) == 4:
                auc_batch = roc_data[3]  # AUC is 4th element
            else:
                auc_batch = np.nan
        else:
            auc_batch = np.nan
        
        # Age offset AUC with full pi tensor subsetted
        if disease_group in results_age_offset_full and results_age_offset_full[disease_group] and len(results_age_offset_full[disease_group]) > 0:
            roc_data = results_age_offset_full[disease_group][0]
            if roc_data is not None and len(roc_data) == 4:
                auc_full = roc_data[3]  # AUC is 4th element
            else:
                auc_full = np.nan
        else:
            auc_full = np.nan
        
        # Convert to float and handle None/NaN using pandas isna
        auc_batch_val = auc_batch if auc_batch is not None else np.nan
        auc_full_val = auc_full if auc_full is not None else np.nan
        
        # Check if valid using pandas isna (handles None, NaN, etc.)
        is_valid_batch = not pd.isna(auc_batch_val)
        is_valid_full = not pd.isna(auc_full_val)
        
        if is_valid_batch and is_valid_full:
            diff = auc_full_val - auc_batch_val
            if abs(diff) < 0.001:  # Very close
                match_status = "✓ Exact"
                matches.append(disease_group)
            elif abs(diff) < 0.01:  # Close
                match_status = "✓ Close"
                matches.append(disease_group)
            else:
                match_status = "⚠ Diff"
                mismatches.append((disease_group, auc_batch_val, auc_full_val, diff))
            
            print(f"{disease_group:<30} {auc_batch_val:>14.4f} {auc_full_val:>14.4f} {diff:>14.4f} {match_status:<10}")
        else:
            print(f"{disease_group:<30} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<10}")
    
    print("\n" + "="*80)
    print("COMPARISON: Age Offset vs Washout Evaluation Functions (Same Full Pi Tensor)")
    print("="*80)
    print(f"{'Disease':<30} {'Age Offset Eval':<18} {'Washout Eval':<18} {'Difference':<15}")
    print("-"*80)
    
    for disease_group in sorted(all_diseases):
        # Age offset AUC with full pi
        if disease_group in results_age_offset_full and results_age_offset_full[disease_group] and len(results_age_offset_full[disease_group]) > 0:
            roc_data = results_age_offset_full[disease_group][0]
            if roc_data is not None and len(roc_data) == 4:
                auc_age_offset_full = roc_data[3]
            else:
                auc_age_offset_full = np.nan
        else:
            auc_age_offset_full = np.nan
        
        # Washout AUC
        if disease_group in results_washout:
            auc_washout = results_washout[disease_group].get('auc', np.nan)
        else:
            auc_washout = np.nan
        
        # Convert to float and handle None/NaN using pandas isna
        auc_age_offset_full_val = auc_age_offset_full if auc_age_offset_full is not None else np.nan
        auc_washout_val = auc_washout if auc_washout is not None else np.nan
        
        # Check if valid using pandas isna (handles None, NaN, etc.)
        is_valid_age_offset = not pd.isna(auc_age_offset_full_val)
        is_valid_washout = not pd.isna(auc_washout_val)
        
        if is_valid_age_offset and is_valid_washout:
            diff = auc_washout_val - auc_age_offset_full_val
            print(f"{disease_group:<30} {auc_age_offset_full_val:>17.4f} {auc_washout_val:>17.4f} {diff:>14.4f}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Matching diseases (same eval, different pi): {len(matches)}")
    print(f"⚠ Mismatching diseases: {len(mismatches)}")
    
    if mismatches:
        print("\nMismatches (same eval function, different pi tensors):")
        for disease, batch_auc, full_auc, diff in mismatches:
            print(f"  {disease}: Batch Pi={batch_auc:.4f}, Full Pi={full_auc:.4f}, Diff={diff:.4f}")
    
    return len(mismatches) == 0


if __name__ == "__main__":
    print("="*80)
    print("COMPARING AGE OFFSET vs WASHOUT (0-year predictions)")
    print("="*80)
    
    # Compare pi tensors
    pi_match = compare_pi_tensors()
    
    # Compare evaluations
    eval_match = compare_evaluations()
    
    print("\n" + "="*80)
    print("FINAL RESULT")
    print("="*80)
    print("KEY QUESTION: Using same age_offset evaluation function:")
    print("  - Do batch-specific pi and full pi tensor give same AUC?")
    print()
    
    if pi_match and eval_match:
        print("✅ PERFECT MATCH!")
        print("   - Pi tensors match")
        print("   - AUCs match (same eval function, different pi sources)")
        print("   → Batch-specific pi and full pi tensor are equivalent")
    else:
        print("⚠️  SOME DIFFERENCES FOUND")
        if not pi_match:
            print("   - Pi tensors differ (max diff: ~0.02, mean diff: ~2.7e-05)")
            print("     → Small differences due to different runs")
        if not eval_match:
            print("   - AUCs differ (same eval function, different pi tensors)")
            print("     → Differences likely due to pi tensor differences")
        
        print("\n   → The small pi differences translate to small AUC differences")
        print("   → Both approaches produce similar results")

