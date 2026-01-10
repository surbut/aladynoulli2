#!/usr/bin/env python
"""
Compare pooled gamma from three versions:
1. _nolr_nopcs_nosex (PRS only, P=36, no lambda_reg, corrected E)
2. _nolr (PRS + sex + PCs, P=47, no lambda_reg, corrected E)
3. March version (old, from resultshighamp, likely PRS only, no lambda_reg, original E)
"""

import torch
import numpy as np
import glob
import argparse
from pathlib import Path

def pool_gamma_from_batches(pattern, max_batches=None, description=""):
    """Load and pool gamma from all batch files matching the pattern."""
    all_gammas = []
    
    files = sorted(glob.glob(pattern))
    print(f"\n{description}")
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
                print(f"  Warning: No gamma found in {Path(file_path).name}")
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
                if len(all_gammas) <= 3:  # Print first 3
                    print(f"  Loaded gamma from {Path(file_path).name}, shape: {gamma.shape}")
            
        except Exception as e:
            print(f"  Error loading {file_path}: {e}")
            continue
    
    if len(all_gammas) == 0:
        raise ValueError(f"No gamma arrays loaded from pattern: {pattern}")
    
    # Stack and compute mean
    gamma_stack = np.stack(all_gammas, axis=0)  # (n_batches, P, K_total)
    gamma_pooled = np.mean(gamma_stack, axis=0)  # (P, K_total)
    
    print(f"  ✓ Pooled gamma from {len(all_gammas)} batches")
    print(f"  Pooled gamma shape: {gamma_pooled.shape}")
    print(f"  Stats: min={gamma_pooled.min():.6f}, max={gamma_pooled.max():.6f}, mean={gamma_pooled.mean():.6f}")
    print(f"  Mean |γ|: {np.abs(gamma_pooled).mean():.6f}, Max |γ|: {np.abs(gamma_pooled).max():.6f}")
    
    return gamma_pooled, len(all_gammas)

def compare_all_versions(gamma_nopcs_nosex, gamma_with_pcs, gamma_march=None):
    """Compare all three gamma versions."""
    print("\n" + "="*80)
    print("COMPREHENSIVE GAMMA COMPARISON")
    print("="*80)
    
    # Statistics for each
    print("\n1. _nolr_nopcs_nosex (PRS only, P={}, corrected E):".format(gamma_nopcs_nosex.shape[0]))
    print(f"   Mean |γ|: {np.abs(gamma_nopcs_nosex).mean():.6f}")
    print(f"   Max |γ|:  {np.abs(gamma_nopcs_nosex).max():.6f}")
    print(f"   Std |γ|:  {np.abs(gamma_nopcs_nosex).std():.6f}")
    print(f"   P95 |γ|:  {np.percentile(np.abs(gamma_nopcs_nosex), 95):.6f}")
    print(f"   P99 |γ|:  {np.percentile(np.abs(gamma_nopcs_nosex), 99):.6f}")
    
    print("\n2. _nolr (PRS + sex + PCs, P={}, corrected E):".format(gamma_with_pcs.shape[0]))
    print(f"   Mean |γ|: {np.abs(gamma_with_pcs).mean():.6f}")
    print(f"   Max |γ|:  {np.abs(gamma_with_pcs).max():.6f}")
    print(f"   Std |γ|:  {np.abs(gamma_with_pcs).std():.6f}")
    print(f"   P95 |γ|:  {np.percentile(np.abs(gamma_with_pcs), 95):.6f}")
    print(f"   P99 |γ|:  {np.percentile(np.abs(gamma_with_pcs), 99):.6f}")
    
    if gamma_march is not None:
        print("\n3. March version (old, P={}, original E):".format(gamma_march.shape[0]))
        print(f"   Mean |γ|: {np.abs(gamma_march).mean():.6f}")
        print(f"   Max |γ|:  {np.abs(gamma_march).max():.6f}")
        print(f"   Std |γ|:  {np.abs(gamma_march).std():.6f}")
        print(f"   P95 |γ|:  {np.percentile(np.abs(gamma_march), 95):.6f}")
        print(f"   P99 |γ|:  {np.percentile(np.abs(gamma_march), 99):.6f}")
    
    # Compare PRS-only versions (nopcs_nosex vs March)
    print("\n" + "="*80)
    print("COMPARISON: _nolr_nopcs_nosex vs _nolr (with PCs/sex)")
    print("="*80)
    
    # Extract PRS portion from _nolr (first 36 components)
    if gamma_with_pcs.shape[0] >= 36:
        gamma_with_pcs_prs_only = gamma_with_pcs[:36, :]
        print(f"\nExtracting PRS portion from _nolr (first 36 of {gamma_with_pcs.shape[0]} components)")
        
        print("\nPRS-only comparison (first 36 components):")
        ratio_mean = np.abs(gamma_nopcs_nosex).mean() / np.abs(gamma_with_pcs_prs_only).mean()
        ratio_max = np.abs(gamma_nopcs_nosex).max() / np.abs(gamma_with_pcs_prs_only).max()
        
        print(f"  Mean |γ| ratio (nopcs_nosex / with_pcs): {ratio_mean:.4f}x")
        print(f"  Max |γ| ratio (nopcs_nosex / with_pcs): {ratio_max:.4f}x")
        
        if ratio_mean > 0.9 and ratio_mean < 1.1:
            print("  ✓ Results are very similar (within 10%)")
        elif ratio_mean > 0.8 and ratio_mean < 1.2:
            print("  ⚠ Results are moderately similar (within 20%)")
        else:
            print("  ⚠ Results differ significantly (>20%)")
    
    # Compare with March if available
    if gamma_march is not None:
        print("\n" + "="*80)
        print("COMPARISON: _nolr_nopcs_nosex vs March version")
        print("="*80)
        
        # Check if dimensions match
        if gamma_nopcs_nosex.shape == gamma_march.shape:
            ratio_mean = np.abs(gamma_nopcs_nosex).mean() / np.abs(gamma_march).mean()
            ratio_max = np.abs(gamma_nopcs_nosex).max() / np.abs(gamma_march).max()
            
            print(f"\nMean |γ| ratio (nopcs_nosex / March): {ratio_mean:.4f}x")
            print(f"Max |γ| ratio (nopcs_nosex / March): {ratio_max:.4f}x")
            
            # Element-wise correlation
            correlation = np.corrcoef(gamma_nopcs_nosex.flatten(), gamma_march.flatten())[0, 1]
            print(f"Correlation: {correlation:.4f}")
            
            if ratio_mean > 0.9 and ratio_mean < 1.1:
                print("  ✓ Results are very similar to March (within 10%)")
            elif ratio_mean > 0.8 and ratio_mean < 1.2:
                print("  ⚠ Results are moderately similar to March (within 20%)")
            else:
                print("  ⚠ Results differ from March (>20%) - likely due to corrected E matrix")
        else:
            print(f"\n⚠ Shape mismatch: nopcs_nosex {gamma_nopcs_nosex.shape} vs March {gamma_march.shape}")
            print("  Cannot directly compare - may have different P dimensions")
    
    # Per-signature comparison
    print("\n" + "="*80)
    print("PER-SIGNATURE COMPARISON (_nolr_nopcs_nosex vs _nolr)")
    print("="*80)
    K = min(gamma_nopcs_nosex.shape[1], gamma_with_pcs.shape[1])
    print(f"\n{'Signature':<12} {'nopcs_nosex (mean |γ|)':<25} {'with_pcs (mean |γ|)':<25} {'Ratio':<10}")
    print("-" * 80)
    
    for k in range(K):
        nopcs_mean = np.abs(gamma_nopcs_nosex[:, k]).mean()
        if gamma_with_pcs.shape[0] >= 36:
            with_pcs_mean = np.abs(gamma_with_pcs[:36, k]).mean()
            ratio = nopcs_mean / (with_pcs_mean + 1e-10)
        else:
            with_pcs_mean = np.abs(gamma_with_pcs[:, k]).mean()
            ratio = nopcs_mean / (with_pcs_mean + 1e-10)
        
        print(f"Signature {k:<2} {nopcs_mean:<25.6f} {with_pcs_mean:<25.6f} {ratio:<10.4f}")

def main():
    parser = argparse.ArgumentParser(description='Compare gamma from all versions')
    parser.add_argument('--nolr_nopcs_nosex_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_nolr_nopcs_nosex',
                       help='Directory with _nolr_nopcs_nosex batches')
    parser.add_argument('--nolr_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_nolr',
                       help='Directory with _nolr batches (with PCs/sex)')
    parser.add_argument('--march_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/resultshighamp',
                       help='Directory with March version batches')
    args = parser.parse_args()
    
    # Load _nolr_nopcs_nosex
    nopcs_nosex_pattern = f"{args.nolr_nopcs_nosex_dir}/enrollment_model_VECTORIZED_W*_nolr_nopcs_nosex_batch_*_*.pt"
    gamma_nopcs_nosex, n1 = pool_gamma_from_batches(
        nopcs_nosex_pattern, 
        description="Loading _nolr_nopcs_nosex batches (PRS only, P=36):"
    )
    
    # Load _nolr (with PCs/sex)
    nolr_pattern = f"{args.nolr_dir}/enrollment_model_VECTORIZED_W*_nolr_batch_*_*.pt"
    gamma_with_pcs, n2 = pool_gamma_from_batches(
        nolr_pattern,
        description="Loading _nolr batches (PRS + sex + PCs, P=47):"
    )
    
    # Try to load March version
    gamma_march = None
    # March version is in results/output_*_*/model.pt
    march_pattern1 = f"{args.march_dir}/results/output_*_*/model.pt"
    march_pattern2 = f"{args.march_dir}/output_*_*/model.pt"
    try:
        # Try results/ subdirectory first
        try:
            gamma_march, n3 = pool_gamma_from_batches(
                march_pattern1,
                description="Loading March version batches:"
            )
        except:
            # Fallback to direct pattern
            gamma_march, n3 = pool_gamma_from_batches(
                march_pattern2,
                description="Loading March version batches:"
            )
    except Exception as e:
        print(f"\n⚠ Could not load March version: {e}")
        print("  Continuing without March comparison...")
    
    # Compare all versions
    compare_all_versions(gamma_nopcs_nosex, gamma_with_pcs, gamma_march)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Compared {n1} _nolr_nopcs_nosex batches vs {n2} _nolr batches")
    if gamma_march is not None:
        print(f"✓ Compared with {n3} March version batches")
    print("="*80)

if __name__ == '__main__':
    main()

