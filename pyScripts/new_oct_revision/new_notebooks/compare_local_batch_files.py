#!/usr/bin/env python3
"""
Compare two local batch files directly (no subsetting needed):
1. Age offset batch: pi_enroll_fixedphi_age_offset_0_sex_0_10000_try2_withpcs_newrun_pooledall.pt
2. Estimation from total: pi_enroll_fixedphi_sex_0_10000.pt

Both are batch 0-10K, so we can compare them directly.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts')
from evaluatetdccode import evaluate_major_diseases_rolling_1year_roc_curves


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
    
    return Y_batch, E_batch, pce_df_batch, essentials


def compare_pi_tensors():
    """Compare the two pi tensors directly."""
    print("="*80)
    print("COMPARING PI TENSORS (LOCAL BATCH FILES)")
    print("="*80)
    
    # Age offset batch
    pi_age_offset_path = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/pi_offset_using_pooled_retrospective_local/') / \
                         'pi_enroll_fixedphi_age_offset_0_sex_0_10000_try2_withpcs_newrun_pooledall.pt'
    
    # Estimation from total
    pi_total_path = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal') / \
                   'enrollment_predictions_fixedphi_RETROSPECTIVE_pooled' / \
                   'pi_enroll_fixedphi_sex_0_10000.pt'
    
    if not pi_age_offset_path.exists():
        print(f"❌ Age offset pi not found: {pi_age_offset_path}")
        return False
    
    if not pi_total_path.exists():
        print(f"❌ Total pi not found: {pi_total_path}")
        return False
    
    print(f"\nLoading age_offset pi: {pi_age_offset_path}")
    pi_age_offset = torch.load(pi_age_offset_path, weights_only=False)
    
    print(f"Loading total pi: {pi_total_path}")
    pi_total = torch.load(pi_total_path, weights_only=False)
    
    print(f"\nAge offset pi shape: {pi_age_offset.shape}")
    print(f"Total pi shape: {pi_total.shape}")
    
    if pi_age_offset.shape != pi_total.shape:
        print(f"❌ Shape mismatch!")
        return False
    
    # Compare tensors
    if torch.allclose(pi_age_offset, pi_total, rtol=1e-4, atol=1e-5):
        print("✅ Pi tensors match!")
        return True
    else:
        print("❌ Pi tensors differ!")
        max_diff = (pi_age_offset - pi_total).abs().max().item()
        mean_diff = (pi_age_offset - pi_total).abs().mean().item()
        print(f"   Max difference: {max_diff:.6e}")
        print(f"   Mean difference: {mean_diff:.6e}")
        
        # Show some sample differences
        print("\n   Sample differences (first 5 patients, first disease, first 5 timepoints):")
        for i in range(min(5, pi_age_offset.shape[0])):
            for d in range(min(1, pi_age_offset.shape[1])):
                for t in range(min(5, pi_age_offset.shape[2])):
                    diff = abs(pi_age_offset[i, d, t].item() - pi_total[i, d, t].item())
                    if diff > 1e-5:
                        print(f"     Patient {i}, Disease {d}, Time {t}: "
                              f"AgeOffset={pi_age_offset[i, d, t].item():.6f}, "
                              f"Total={pi_total[i, d, t].item():.6f}, "
                              f"Diff={diff:.6e}")
        
        return False


def compare_evaluations():
    """Compare evaluation results using the same evaluation function."""
    print("\n" + "="*80)
    print("COMPARING EVALUATION RESULTS (SAME EVAL FUNCTION)")
    print("="*80)
    
    # Load data
    Y_batch, E_batch, pce_df_batch, essentials = load_data_for_batch(0, 10000)
    disease_names = essentials['disease_names']
    
    # Load pi tensors
    pi_age_offset_path = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/pi_offset_using_pooled_retrospective_local/') / \
                         'pi_enroll_fixedphi_age_offset_0_sex_0_10000_try2_withpcs_newrun_pooledall.pt'
    pi_total_path = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal') / \
                   'enrollment_predictions_fixedphi_RETROSPECTIVE_pooled' / \
                   'pi_enroll_fixedphi_sex_0_10000.pt'
    
    pi_age_offset = torch.load(pi_age_offset_path, weights_only=False)
    pi_total = torch.load(pi_total_path, weights_only=False)
    
    # Run same evaluation function on both
    print("\nRunning age_offset evaluation on age_offset pi...")
    results_age_offset = evaluate_major_diseases_rolling_1year_roc_curves(
        [pi_age_offset], Y_batch, E_batch, disease_names, pce_df_batch,
        patient_indices=None, plot_group=None
    )
    
    print("\nRunning age_offset evaluation on total pi...")
    results_total = evaluate_major_diseases_rolling_1year_roc_curves(
        [pi_total], Y_batch, E_batch, disease_names, pce_df_batch,
        patient_indices=None, plot_group=None
    )
    
    # Compare AUCs
    print("\n" + "="*80)
    print("AUC COMPARISON: Same Evaluation Function, Different Pi Tensors")
    print("="*80)
    print(f"{'Disease':<30} {'Age Offset Pi':<18} {'Total Pi':<18} {'Difference':<15} {'Match':<10}")
    print("-"*80)
    
    matches = []
    close_matches = []
    mismatches = []
    
    all_diseases = set(results_age_offset.keys()) | set(results_total.keys())
    
    for disease_group in sorted(all_diseases):
        # Age offset AUC
        if disease_group in results_age_offset and results_age_offset[disease_group] and len(results_age_offset[disease_group]) > 0:
            roc_data = results_age_offset[disease_group][0]
            if roc_data is not None and len(roc_data) == 4:
                auc_age_offset = roc_data[3]  # AUC is 4th element
            else:
                auc_age_offset = None
        else:
            auc_age_offset = None
        
        # Total AUC
        if disease_group in results_total and results_total[disease_group] and len(results_total[disease_group]) > 0:
            roc_data = results_total[disease_group][0]
            if roc_data is not None and len(roc_data) == 4:
                auc_total = roc_data[3]  # AUC is 4th element
            else:
                auc_total = None
        else:
            auc_total = None
        
        # Check if valid
        is_valid_age_offset = not pd.isna(auc_age_offset) if auc_age_offset is not None else False
        is_valid_total = not pd.isna(auc_total) if auc_total is not None else False
        
        if is_valid_age_offset and is_valid_total:
            diff = abs(auc_total - auc_age_offset)
            if diff < 0.001:  # Very close
                match_status = "✓ Exact"
                matches.append(disease_group)
            elif diff < 0.01:  # Close
                match_status = "✓ Close"
                close_matches.append(disease_group)
            else:
                match_status = "⚠ Diff"
                mismatches.append((disease_group, auc_age_offset, auc_total, diff))
            
            print(f"{disease_group:<30} {auc_age_offset:>17.4f} {auc_total:>17.4f} {diff:>14.4f} {match_status:<10}")
        else:
            print(f"{disease_group:<30} {'N/A':<18} {'N/A':<18} {'N/A':<15} {'N/A':<10}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Exact matches (<0.001): {len(matches)}")
    print(f"✓ Close matches (<0.01): {len(close_matches)}")
    print(f"⚠ Differences (>=0.01): {len(mismatches)}")
    
    if matches:
        print(f"\nExact matches: {', '.join(matches[:10])}{'...' if len(matches) > 10 else ''}")
    
    if close_matches:
        print(f"\nClose matches: {', '.join(close_matches[:10])}{'...' if len(close_matches) > 10 else ''}")
    
    if mismatches:
        print(f"\nDifferences:")
        for disease, ao_auc, total_auc, diff in mismatches[:10]:
            print(f"  {disease}: AgeOffset={ao_auc:.4f}, Total={total_auc:.4f}, Diff={diff:.4f}")
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches) - 10} more")
    
    return len(mismatches) == 0


if __name__ == "__main__":
    print("="*80)
    print("COMPARING LOCAL BATCH FILES (NO SUBSETTING NEEDED)")
    print("="*80)
    print("\nComparing:")
    print("  1. Age offset batch: pi_enroll_fixedphi_age_offset_0_sex_0_10000_try2_withpcs_newrun_pooledall.pt")
    print("  2. Total batch: pi_enroll_fixedphi_sex_0_10000.pt")
    print("\nBoth are batch 0-10K, so comparing directly (no AWS issues)")
    print("="*80)
    
    # Compare pi tensors
    pi_match = compare_pi_tensors()
    
    # Compare evaluations
    eval_match = compare_evaluations()
    
    print("\n" + "="*80)
    print("FINAL RESULT")
    print("="*80)
    if pi_match and eval_match:
        print("✅ PERFECT MATCH!")
        print("   - Pi tensors match")
        print("   - AUCs match (same eval function)")
        print("   → Both approaches produce identical results")
    else:
        print("⚠️  SOME DIFFERENCES FOUND")
        if not pi_match:
            print("   - Pi tensors differ")
        if not eval_match:
            print("   - AUCs differ (same eval function, different pi tensors)")
        
        print("\n   → Differences may be due to:")
        print("     * Different training runs (even with same seed)")
        print("     * Numerical precision")
        print("     * Different initialization")
        print("     * Training dynamics")

