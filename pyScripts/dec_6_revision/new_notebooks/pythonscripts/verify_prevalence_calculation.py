"""
Verify that computing unweighted prevalence on all 400K patients with the same code
gives the same result as prevalence_t_corrected.pt
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

print("="*80)
print("VERIFYING PREVALENCE CALCULATION: All 400K Patients")
print("="*80)

data_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')

# Load saved corrected prevalence (old, all 400K)
print("\n1. Loading saved prevalence_t_corrected.pt (all 400K)...")
prevalence_t_saved = torch.load(str(data_dir / 'prevalence_t_corrected.pt'), weights_only=False)
if torch.is_tensor(prevalence_t_saved):
    prevalence_t_saved = prevalence_t_saved.numpy()
print(f"   ✓ Loaded saved prevalence: {prevalence_t_saved.shape}")

# Load Y and E_corrected (all 400K)
print("\n2. Loading Y and E_corrected (all 400K)...")
Y = torch.load(str(data_dir / 'Y_tensor.pt'), weights_only=False)
E_corrected = torch.load(str(data_dir / 'E_matrix_corrected.pt'), weights_only=False)
print(f"   ✓ Loaded Y: {Y.shape}")
print(f"   ✓ Loaded E_corrected: {E_corrected.shape}")

# Recompute using the same function as the verified notebook
def compute_smoothed_prevalence_at_risk(Y, E_corrected, window_size=5, smooth_on_logit=True):
    """
    Compute smoothed prevalence with proper at-risk filtering.
    This matches the verified code in R3_Verify_Corrected_Data.ipynb
    """
    if torch.is_tensor(Y):
        Y = Y.numpy()
    if torch.is_tensor(E_corrected):
        E_corrected = E_corrected.numpy()
    
    N, D, T = Y.shape
    prevalence_t = np.zeros((D, T))
    
    # Convert timepoints to ages (assuming timepoint 0 = age 30)
    timepoint_ages = np.arange(T) + 30
    
    print(f"\n  Computing prevalence for {D} diseases, {T} timepoints...")
    print(f"  Using at-risk filtering with corrected E")
    
    for d in range(D):
        if d % 50 == 0:
            print(f"    Processing disease {d}/{D}...")
        
        for t in range(T):
            age_t = timepoint_ages[t]
            
            # Only include people who are still at risk at timepoint t
            # This matches the verified code: E_corrected[:, d] >= t
            at_risk_mask = (E_corrected[:, d] >= t)
            
            if at_risk_mask.sum() > 0:
                prevalence_t[d, t] = Y[at_risk_mask, d, t].mean()
            else:
                prevalence_t[d, t] = np.nan
        
        # Smooth as before
        if smooth_on_logit:
            epsilon = 1e-8
            # Handle NaN values
            valid_mask = ~np.isnan(prevalence_t[d, :])
            if valid_mask.sum() > 0:
                logit_prev = np.full(T, np.nan)
                logit_prev[valid_mask] = np.log(
                    (prevalence_t[d, valid_mask] + epsilon) / 
                    (1 - prevalence_t[d, valid_mask] + epsilon)
                )
                # Smooth only valid values
                smoothed_logit = gaussian_filter1d(
                    np.nan_to_num(logit_prev, nan=0), 
                    sigma=window_size
                )
                # Restore NaN where original was NaN
                smoothed_logit[~valid_mask] = np.nan
                prevalence_t[d, :] = 1 / (1 + np.exp(-smoothed_logit))
        else:
            prevalence_t[d, :] = gaussian_filter1d(
                np.nan_to_num(prevalence_t[d, :], nan=0), 
                sigma=window_size
            )
    
    return prevalence_t

# Recompute
print("\n3. Recomputing prevalence with same code (all 400K)...")
prevalence_t_recomputed = compute_smoothed_prevalence_at_risk(
    Y=Y,
    E_corrected=E_corrected,
    window_size=5,
    smooth_on_logit=True
)

print(f"\n✓ Recomputed prevalence: {prevalence_t_recomputed.shape}")

# Compare
print("\n" + "="*80)
print("COMPARISON: Saved vs Recomputed")
print("="*80)

if prevalence_t_saved.shape == prevalence_t_recomputed.shape:
    # Calculate differences
    diff = prevalence_t_saved - prevalence_t_recomputed
    mean_diff = np.nanmean(np.abs(diff))
    max_diff = np.nanmax(np.abs(diff))
    
    # Correlation (excluding NaN)
    valid_mask = ~(np.isnan(prevalence_t_saved) | np.isnan(prevalence_t_recomputed))
    if valid_mask.sum() > 0:
        saved_flat = prevalence_t_saved[valid_mask]
        recomputed_flat = prevalence_t_recomputed[valid_mask]
        correlation = np.corrcoef(saved_flat, recomputed_flat)[0, 1]
    else:
        correlation = np.nan
    
    print(f"\nComparison Statistics:")
    print(f"  Mean absolute difference: {mean_diff:.10f}")
    print(f"  Max absolute difference: {max_diff:.10f}")
    print(f"  Correlation: {correlation:.10f}")
    
    # Check if they match (within numerical precision)
    if mean_diff < 1e-6 and max_diff < 1e-5:
        print(f"\n✅ PERFECT MATCH! Recomputed prevalence matches saved prevalence_t_corrected.pt")
        print(f"   (Differences are within numerical precision)")
    else:
        print(f"\n⚠️  Differences detected. Checking specific diseases...")
        
        # Check breast cancer (disease 16)
        breast_cancer_idx = 16
        if breast_cancer_idx < prevalence_t_saved.shape[0]:
            bc_saved = prevalence_t_saved[breast_cancer_idx, :]
            bc_recomputed = prevalence_t_recomputed[breast_cancer_idx, :]
            bc_diff = np.abs(bc_saved - bc_recomputed)
            print(f"\n  Breast Cancer (disease {breast_cancer_idx}):")
            print(f"    Mean diff: {np.nanmean(bc_diff):.10f}")
            print(f"    Max diff: {np.nanmax(bc_diff):.10f}")
            
            # Check a few other diseases
            for disease_idx in [112, 66, 127, 47]:
                if disease_idx < prevalence_t_saved.shape[0]:
                    d_saved = prevalence_t_saved[disease_idx, :]
                    d_recomputed = prevalence_t_recomputed[disease_idx, :]
                    d_diff = np.abs(d_saved - d_recomputed)
                    print(f"  Disease {disease_idx}:")
                    print(f"    Mean diff: {np.nanmean(d_diff):.10f}")
                    print(f"    Max diff: {np.nanmax(d_diff):.10f}")
else:
    print(f"\n⚠️  Shape mismatch:")
    print(f"   Saved: {prevalence_t_saved.shape}")
    print(f"   Recomputed: {prevalence_t_recomputed.shape}")

print(f"\n{'='*80}")
print("VERIFICATION COMPLETE")
print(f"{'='*80}")

