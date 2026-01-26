#!/usr/bin/env python
"""
Check gamma variability across prediction batches.
Shows how much gamma varies when learned per batch (no regularization).
"""

import torch
import numpy as np
import glob
import os
from pathlib import Path

def load_gamma_from_checkpoint(checkpoint_path):
    """Load gamma from a checkpoint file"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        if 'gamma' in checkpoint['model_state_dict']:
            gamma = checkpoint['model_state_dict']['gamma']
            if torch.is_tensor(gamma):
                return gamma.detach().cpu().numpy()
            return np.array(gamma)
    
    if 'gamma' in checkpoint:
        gamma = checkpoint['gamma']
        if torch.is_tensor(gamma):
            return gamma.detach().cpu().numpy()
        return np.array(gamma)
    
    return None

def main():
    # Path to prediction output directory
    prediction_dir = "/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_correctedE_vectorized/"
    
    # Find all model checkpoint files
    pattern = os.path.join(prediction_dir, "model_enroll_fixedphi_sex_*.pt")
    checkpoint_files = sorted(glob.glob(pattern))
    
    if not checkpoint_files:
        print(f"No checkpoint files found matching pattern: {pattern}")
        return
    
    print("="*80)
    print("GAMMA VARIABILITY ACROSS PREDICTION BATCHES")
    print("="*80)
    print(f"Found {len(checkpoint_files)} checkpoint files")
    print()
    
    all_gammas = []
    
    for i, checkpoint_path in enumerate(checkpoint_files):
        gamma = load_gamma_from_checkpoint(checkpoint_path)
        if gamma is None:
            print(f"Warning: Could not load gamma from {os.path.basename(checkpoint_path)}")
            continue
        
        all_gammas.append(gamma)
        if i < 3:  # Print first 3
            print(f"  Loaded batch {i+1}: {os.path.basename(checkpoint_path)} (shape: {gamma.shape})")
    
    if not all_gammas:
        print("No gamma values found!")
        return
    
    # Stack and compute statistics
    gamma_stack = np.stack(all_gammas, axis=0)  # (n_batches, P, K)
    gamma_mean = np.mean(gamma_stack, axis=0)  # (P, K)
    gamma_std = np.std(gamma_stack, axis=0)    # (P, K) - std across batches for each element
    
    print(f"\n{'='*80}")
    print("GAMMA VARIABILITY STATISTICS")
    print(f"{'='*80}")
    print(f"  Number of batches: {len(all_gammas)}")
    print(f"  Gamma shape: {gamma_mean.shape}")
    
    # Statistics of the mean gamma (pooled)
    print(f"\n  Pooled gamma (mean across batches):")
    print(f"    Mean |γ|: {np.abs(gamma_mean).mean():.6f}")
    print(f"    Max |γ|: {np.abs(gamma_mean).max():.6f}")
    print(f"    Min: {gamma_mean.min():.6f}, Max: {gamma_mean.max():.6f}")
    
    # Statistics of the std (variability across batches)
    print(f"\n  Variability across batches (std per element):")
    print(f"    Mean std: {gamma_std.mean():.6f}")
    print(f"    Median std: {np.median(gamma_std):.6f}")
    print(f"    Max std: {gamma_std.max():.6f}")
    print(f"    P95 std: {np.percentile(gamma_std, 95):.6f}")
    print(f"    P99 std: {np.percentile(gamma_std, 99):.6f}")
    
    # Coefficient of variation (std / mean) for non-zero elements
    gamma_abs_mean = np.abs(gamma_mean)
    non_zero_mask = gamma_abs_mean > 1e-6
    if non_zero_mask.sum() > 0:
        cv = gamma_std[non_zero_mask] / (gamma_abs_mean[non_zero_mask] + 1e-10)
        print(f"\n  Coefficient of Variation (std/mean) for non-zero elements:")
        print(f"    Mean CV: {cv.mean():.4f}")
        print(f"    Median CV: {np.median(cv):.4f}")
        print(f"    Max CV: {cv.max():.4f}")
    
    # Compare per-batch statistics
    print(f"\n  Per-batch statistics (showing first 5 batches):")
    for i in range(min(5, len(all_gammas))):
        gamma_batch = all_gammas[i]
        print(f"    Batch {i+1}: Mean |γ| = {np.abs(gamma_batch).mean():.6f}, Max |γ| = {np.abs(gamma_batch).max():.6f}")
    
    # Overall assessment
    mean_std = gamma_std.mean()
    if mean_std < 0.0001:
        print(f"\n  ✓ Gamma is very consistent across batches (mean std = {mean_std:.6f})")
    elif mean_std < 0.001:
        print(f"\n  ⚠️  Gamma shows moderate variability (mean std = {mean_std:.6f})")
    else:
        print(f"\n  ⚠️  WARNING: Gamma varies significantly across batches (mean std = {mean_std:.6f})")
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()
