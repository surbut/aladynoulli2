#!/usr/bin/env python
"""
Compare pooled gamma from batches with lambda_reg vs without lambda_reg

This script:
- Loads and pools gamma from old batches (with lambda_reg)
- Loads and pools gamma from new _nolr batches (no lambda_reg)
- Compares magnitudes to check for shrinkage
"""

import torch
import numpy as np
import glob
import argparse
from pathlib import Path

def pool_gamma_from_batches(pattern, max_batches=None):
    """Load and pool gamma from all batch files matching the pattern."""
    all_gammas = []
    
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} files matching pattern: {pattern}")
    
    if max_batches is not None:
        files = files[:max_batches]
    
    for file_path in files:
        try:
            checkpoint = torch.load(file_path, weights_only=False)
            
            # Extract gamma
            if 'model_state_dict' in checkpoint and 'gamma' in checkpoint['model_state_dict']:
                gamma = checkpoint['model_state_dict']['gamma']
            elif 'gamma' in checkpoint:
                gamma = checkpoint['gamma']
            else:
                print(f"Warning: No gamma found in {file_path}")
                continue
            
            # Convert to numpy if tensor
            if torch.is_tensor(gamma):
                gamma = gamma.detach().cpu().numpy()
            elif not isinstance(gamma, np.ndarray):
                gamma = np.array(gamma)
            
            # Check if gamma is all zeros
            if np.allclose(gamma, 0):
                print(f"  Warning: {Path(file_path).name} has gamma=0 (possibly untrained)")
            else:
                all_gammas.append(gamma)
                print(f"  Loaded gamma from {Path(file_path).name}, shape: {gamma.shape}")
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    if len(all_gammas) == 0:
        raise ValueError(f"No gamma arrays loaded from pattern: {pattern}")
    
    # Stack and compute mean
    gamma_stack = np.stack(all_gammas, axis=0)  # (n_batches, P, K_total)
    gamma_pooled = np.mean(gamma_stack, axis=0)  # (P, K_total)
    
    print(f"\nPooled gamma from {len(all_gammas)} batches")
    print(f"Pooled gamma shape: {gamma_pooled.shape}")
    print(f"Pooled gamma stats: min={gamma_pooled.min():.6f}, max={gamma_pooled.max():.6f}, mean={gamma_pooled.mean():.6f}")
    print(f"Non-zero gamma values: {np.count_nonzero(gamma_pooled)}/{gamma_pooled.size}")
    
    return gamma_pooled

def compare_gammas(gamma_with_lr, gamma_no_lr):
    """Compare two gamma arrays and compute shrinkage metrics."""
    print("\n" + "="*80)
    print("GAMMA COMPARISON: With lambda_reg vs No lambda_reg")
    print("="*80)
    
    # Statistics for each
    print("\nWith lambda_reg (shrunken):")
    print(f"  Mean |γ|: {np.abs(gamma_with_lr).mean():.6f}")
    print(f"  Max |γ|:  {np.abs(gamma_with_lr).max():.6f}")
    print(f"  Std |γ|:  {np.abs(gamma_with_lr).std():.6f}")
    print(f"  P50 |γ|:  {np.percentile(np.abs(gamma_with_lr), 50):.6f}")
    print(f"  P95 |γ|:  {np.percentile(np.abs(gamma_with_lr), 95):.6f}")
    print(f"  P99 |γ|:  {np.percentile(np.abs(gamma_with_lr), 99):.6f}")
    
    print("\nNo lambda_reg (unshrunken):")
    print(f"  Mean |γ|: {np.abs(gamma_no_lr).mean():.6f}")
    print(f"  Max |γ|:  {np.abs(gamma_no_lr).max():.6f}")
    print(f"  Std |γ|:  {np.abs(gamma_no_lr).std():.6f}")
    print(f"  P50 |γ|:  {np.percentile(np.abs(gamma_no_lr), 50):.6f}")
    print(f"  P95 |γ|:  {np.percentile(np.abs(gamma_no_lr), 95):.6f}")
    print(f"  P99 |γ|:  {np.percentile(np.abs(gamma_no_lr), 99):.6f}")
    
    # Compute shrinkage ratios
    print("\n" + "="*80)
    print("SHRINKAGE RATIOS (With LR / No LR)")
    print("="*80)
    
    mean_ratio = np.abs(gamma_with_lr).mean() / np.abs(gamma_no_lr).mean()
    max_ratio = np.abs(gamma_with_lr).max() / np.abs(gamma_no_lr).max()
    std_ratio = np.abs(gamma_with_lr).std() / np.abs(gamma_no_lr).std()
    p50_ratio = np.percentile(np.abs(gamma_with_lr), 50) / np.percentile(np.abs(gamma_no_lr), 50)
    p95_ratio = np.percentile(np.abs(gamma_with_lr), 95) / np.percentile(np.abs(gamma_no_lr), 95)
    p99_ratio = np.percentile(np.abs(gamma_with_lr), 99) / np.percentile(np.abs(gamma_no_lr), 99)
    
    print(f"Mean |γ| ratio: {mean_ratio:.4f}x (shrinkage: {(1-mean_ratio)*100:.2f}%)")
    print(f"Max |γ| ratio:  {max_ratio:.4f}x (shrinkage: {(1-max_ratio)*100:.2f}%)")
    print(f"Std |γ| ratio:  {std_ratio:.4f}x (shrinkage: {(1-std_ratio)*100:.2f}%)")
    print(f"P50 |γ| ratio:  {p50_ratio:.4f}x (shrinkage: {(1-p50_ratio)*100:.2f}%)")
    print(f"P95 |γ| ratio:  {p95_ratio:.4f}x (shrinkage: {(1-p95_ratio)*100:.2f}%)")
    print(f"P99 |γ| ratio:  {p99_ratio:.4f}x (shrinkage: {(1-p99_ratio)*100:.2f}%)")
    
    # Element-wise comparison
    print("\n" + "="*80)
    print("ELEMENT-WISE COMPARISON")
    print("="*80)
    
    abs_diff = np.abs(gamma_with_lr - gamma_no_lr)
    rel_diff = abs_diff / (np.abs(gamma_no_lr) + 1e-10)  # Avoid division by zero
    
    print(f"Mean absolute difference: {abs_diff.mean():.6f}")
    print(f"Max absolute difference:  {abs_diff.max():.6f}")
    print(f"Mean relative difference: {rel_diff.mean():.4f} ({rel_diff.mean()*100:.2f}%)")
    print(f"Max relative difference:  {rel_diff.max():.4f} ({rel_diff.max()*100:.2f}%)")
    
    # Per-signature comparison
    print("\n" + "="*80)
    print("PER-SIGNATURE COMPARISON")
    print("="*80)
    
    K = gamma_with_lr.shape[1]
    print(f"\n{'Signature':<12} {'With LR (mean |γ|)':<20} {'No LR (mean |γ|)':<20} {'Ratio':<10} {'Shrinkage':<12}")
    print("-" * 80)
    
    for k in range(K):
        with_lr_mean = np.abs(gamma_with_lr[:, k]).mean()
        no_lr_mean = np.abs(gamma_no_lr[:, k]).mean()
        ratio = with_lr_mean / (no_lr_mean + 1e-10)
        shrinkage = (1 - ratio) * 100
        
        print(f"Signature {k:<3} {with_lr_mean:<20.6f} {no_lr_mean:<20.6f} {ratio:<10.4f} {shrinkage:<12.2f}%")
    
    # Overall assessment
    print("\n" + "="*80)
    print("ASSESSMENT")
    print("="*80)
    
    if mean_ratio < 0.5:
        print("🔴 **Strong shrinkage detected**: Mean gamma is <50% of unshrunken version")
        print("   This indicates lambda_reg penalty is having a substantial effect.")
    elif mean_ratio < 0.8:
        print("🟡 **Moderate shrinkage detected**: Mean gamma is 50-80% of unshrunken version")
        print("   This indicates lambda_reg penalty is having a noticeable effect.")
    elif mean_ratio < 0.95:
        print("🟠 **Mild shrinkage detected**: Mean gamma is 80-95% of unshrunken version")
        print("   This indicates lambda_reg penalty is having a small effect.")
    else:
        print("🟢 **Minimal shrinkage**: Mean gamma is >95% of unshrunken version")
        print("   This suggests lambda_reg penalty has little effect on gamma magnitude.")
    
    return {
        'mean_ratio': mean_ratio,
        'max_ratio': max_ratio,
        'shrinkage_pct': (1 - mean_ratio) * 100,
        'abs_diff_mean': abs_diff.mean(),
        'abs_diff_max': abs_diff.max(),
        'rel_diff_mean': rel_diff.mean(),
    }

def main():
    parser = argparse.ArgumentParser(description='Compare pooled gamma from batches with vs without lambda_reg')
    parser.add_argument('--with_lr_pattern', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized/enrollment_model_W0.0001_batch_*_*.pt',
                       help='Pattern for batches WITH lambda_reg')
    parser.add_argument('--no_lr_pattern', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_nolr/enrollment_model_VECTORIZED_W0.0001_nolr_batch_*_*.pt',
                       help='Pattern for batches WITHOUT lambda_reg (_nolr)')
    parser.add_argument('--max_batches', type=int, default=None,
                       help='Maximum number of batches to pool (None = all)')
    args = parser.parse_args()
    
    print("="*80)
    print("GAMMA SHRINKAGE COMPARISON")
    print("="*80)
    print("\nComparing pooled gamma from:")
    print(f"  With lambda_reg:    {args.with_lr_pattern}")
    print(f"  Without lambda_reg: {args.no_lr_pattern}")
    print("="*80)
    
    # Pool gamma from batches with lambda_reg
    print("\n1. Pooling gamma from batches WITH lambda_reg...")
    try:
        gamma_with_lr = pool_gamma_from_batches(args.with_lr_pattern, args.max_batches)
    except Exception as e:
        print(f"✗ Error pooling gamma with lambda_reg: {e}")
        return
    
    # Pool gamma from batches without lambda_reg
    print("\n2. Pooling gamma from batches WITHOUT lambda_reg (_nolr)...")
    try:
        gamma_no_lr = pool_gamma_from_batches(args.no_lr_pattern, args.max_batches)
    except Exception as e:
        print(f"✗ Error pooling gamma without lambda_reg: {e}")
        return
    
    # Compare
    print("\n3. Comparing gammas...")
    if gamma_with_lr.shape != gamma_no_lr.shape:
        print(f"✗ Error: Shape mismatch! With LR: {gamma_with_lr.shape}, No LR: {gamma_no_lr.shape}")
        return
    
    results = compare_gammas(gamma_with_lr, gamma_no_lr)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print(f"\nSummary:")
    print(f"  Mean shrinkage: {results['shrinkage_pct']:.2f}%")
    print(f"  Mean ratio: {results['mean_ratio']:.4f}x")
    print(f"  Max ratio: {results['max_ratio']:.4f}x")

if __name__ == '__main__':
    main()