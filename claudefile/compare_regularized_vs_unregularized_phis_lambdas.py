#!/usr/bin/env python3
"""
Compare pooled phis and sample lambdas between regularized and unregularized training.

This script:
1. Loads pooled phi from master checkpoint (regularized)
2. Loads pooled phi from nolr batches (unregularized)
3. Compares phis (correlation, mean difference, etc.)
4. Loads sample lambdas from both sets and compares them
"""

import torch
import numpy as np
import glob
from pathlib import Path
from scipy.stats import pearsonr
import pandas as pd

# Paths - adjust these to match your actual file locations
MASTER_CHECKPOINT_PATH = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/master_for_fitting_pooled_correctedE.pt'

# Patterns for batch files - adjust these paths as needed
# Regularized batches (used to create master checkpoint) - note: no "VECTORIZED" in name
REGULARIZED_BATCH_PATTERN = '/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized/enrollment_model_W0.0001_batch_*_*.pt'
# Unregularized batches (nolr = no learning rate regularization)
UNREGULARIZED_BATCH_PATTERN = '/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_nolr/enrollment_model_VECTORIZED_W0.0001_nolr_batch_*_*.pt'

def pool_phi_from_batches(pattern, max_batches=None):
    """Load and pool phi from batch files."""
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} files matching pattern")
    
    if max_batches is not None:
        files = files[:max_batches]
    
    all_phis = []
    for file_path in files:
        try:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
            
            if 'model_state_dict' in checkpoint and 'phi' in checkpoint['model_state_dict']:
                phi = checkpoint['model_state_dict']['phi']
            elif 'phi' in checkpoint:
                phi = checkpoint['phi']
            else:
                continue
            
            if torch.is_tensor(phi):
                phi = phi.detach().cpu().numpy()
            
            all_phis.append(phi)
        except Exception as e:
            print(f"Error loading {Path(file_path).name}: {e}")
            continue
    
    if len(all_phis) == 0:
        raise ValueError(f"No phi found in files matching {pattern}")
    
    # Stack and compute mean
    phi_stack = np.stack(all_phis, axis=0)  # (n_batches, K, D, T)
    phi_pooled = np.mean(phi_stack, axis=0)  # (K, D, T)
    
    print(f"Pooled phi from {len(all_phis)} batches, shape: {phi_pooled.shape}")
    return phi_pooled, all_phis

def load_sample_lambdas(pattern, n_samples=5):
    """Load lambda from a sample of batch files."""
    files = sorted(glob.glob(pattern))
    if len(files) == 0:
        print(f"⚠️  No files found matching pattern: {pattern}")
        return None
    
    # Sample a few batches
    sample_files = files[:min(n_samples, len(files))]
    all_lambdas = []
    
    for file_path in sample_files:
        try:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
            
            # Try different possible locations for lambda
            lambda_ = None
            if 'model_state_dict' in checkpoint:
                if 'lambda_' in checkpoint['model_state_dict']:
                    lambda_ = checkpoint['model_state_dict']['lambda_']
                elif 'lambda' in checkpoint['model_state_dict']:
                    lambda_ = checkpoint['model_state_dict']['lambda']
            
            if lambda_ is None:
                if 'lambda_' in checkpoint:
                    lambda_ = checkpoint['lambda_']
                elif 'lambda' in checkpoint:
                    lambda_ = checkpoint['lambda']
            
            if lambda_ is None:
                print(f"⚠️  No lambda found in {Path(file_path).name}")
                print(f"    Available keys: {list(checkpoint.keys())[:10]}")
                if 'model_state_dict' in checkpoint:
                    print(f"    Model state dict keys: {list(checkpoint['model_state_dict'].keys())[:10]}")
                continue
            
            if torch.is_tensor(lambda_):
                lambda_ = lambda_.detach().cpu().numpy()
            
            all_lambdas.append(lambda_)
            print(f"✓ Loaded lambda from {Path(file_path).name}, shape: {lambda_.shape}")
        except Exception as e:
            print(f"✗ Error loading lambda from {Path(file_path).name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(all_lambdas) == 0:
        print(f"⚠️  Could not load any lambdas from files matching {pattern}")
        return None
    
    return all_lambdas

def compare_phis(phi_reg, phi_nolr):
    """Compare two phi arrays."""
    print("\n" + "="*80)
    print("PHI COMPARISON")
    print("="*80)
    
    # Flatten for correlation
    phi_reg_flat = phi_reg.flatten()
    phi_nolr_flat = phi_nolr.flatten()
    
    # Correlation
    corr, pval = pearsonr(phi_reg_flat, phi_nolr_flat)
    print(f"Pearson correlation: {corr:.6f} (p={pval:.2e})")
    
    # Mean absolute difference
    mad = np.mean(np.abs(phi_reg - phi_nolr))
    print(f"Mean absolute difference: {mad:.6f}")
    
    # Max absolute difference
    max_diff = np.max(np.abs(phi_reg - phi_nolr))
    print(f"Max absolute difference: {max_diff:.6f}")
    
    # Relative difference (as %)
    rel_diff = 100 * mad / (np.abs(phi_reg).mean() + 1e-10)
    print(f"Mean relative difference: {rel_diff:.4f}%")
    
    # Per-signature comparison
    print("\nPer-signature comparison (mean absolute difference):")
    for k in range(min(phi_reg.shape[0], 21)):
        sig_diff = np.mean(np.abs(phi_reg[k] - phi_nolr[k]))
        print(f"  Signature {k}: {sig_diff:.6f}")
    
    # Per-disease comparison (top 10)
    print("\nPer-disease comparison (top 10 by difference):")
    disease_diffs = []
    for d in range(phi_reg.shape[1]):
        disease_diff = np.mean(np.abs(phi_reg[:, d, :] - phi_nolr[:, d, :]))
        disease_diffs.append((d, disease_diff))
    
    disease_diffs.sort(key=lambda x: x[1], reverse=True)
    for d, diff in disease_diffs[:10]:
        print(f"  Disease {d}: {diff:.6f}")
    
    return {
        'correlation': corr,
        'pvalue': pval,
        'mean_abs_diff': mad,
        'max_abs_diff': max_diff,
        'rel_diff_pct': rel_diff
    }

def compare_lambdas(lambdas_reg, lambdas_nolr):
    """Compare lambda arrays from sample batches."""
    print("\n" + "="*80)
    print("LAMBDA COMPARISON")
    print("="*80)
    
    if lambdas_reg is None or lambdas_nolr is None:
        print("⚠️  Could not load lambdas from one or both sets")
        return None
    
    # Take first batch from each for comparison
    lambda_reg = lambdas_reg[0]
    lambda_nolr = lambdas_nolr[0]
    
    print(f"Regularized lambda shape: {lambda_reg.shape}")
    print(f"Unregularized lambda shape: {lambda_nolr.shape}")
    
    if lambda_reg.shape != lambda_nolr.shape:
        print("⚠️  Lambda shapes don't match - comparing first N samples")
        min_samples = min(lambda_reg.shape[0], lambda_nolr.shape[0])
        lambda_reg = lambda_reg[:min_samples]
        lambda_nolr = lambda_nolr[:min_samples]
    
    # Flatten for correlation
    lambda_reg_flat = lambda_reg.flatten()
    lambda_nolr_flat = lambda_nolr.flatten()
    
    # Correlation
    corr, pval = pearsonr(lambda_reg_flat, lambda_nolr_flat)
    print(f"Pearson correlation: {corr:.6f} (p={pval:.2e})")
    
    # Mean absolute difference
    mad = np.mean(np.abs(lambda_reg - lambda_nolr))
    print(f"Mean absolute difference: {mad:.6f}")
    
    # Per-signature comparison
    print("\nPer-signature comparison (mean absolute difference):")
    for k in range(min(lambda_reg.shape[1], 21)):
        sig_diff = np.mean(np.abs(lambda_reg[:, k, :] - lambda_nolr[:, k, :]))
        print(f"  Signature {k}: {sig_diff:.6f}")
    
    # Per-individual correlation (sample of 10)
    print("\nPer-individual correlation (first 10 individuals):")
    n_individuals = min(10, lambda_reg.shape[0])
    individual_corrs = []
    for i in range(n_individuals):
        ind_corr, _ = pearsonr(lambda_reg[i].flatten(), lambda_nolr[i].flatten())
        individual_corrs.append(ind_corr)
        print(f"  Individual {i}: {ind_corr:.6f}")
    
    print(f"\nMean individual correlation: {np.mean(individual_corrs):.6f}")
    
    return {
        'correlation': corr,
        'pvalue': pval,
        'mean_abs_diff': mad,
        'mean_individual_corr': np.mean(individual_corrs)
    }

def main():
    print("="*80)
    print("COMPARING REGULARIZED vs UNREGULARIZED PHIS AND LAMBDAS")
    print("="*80)
    
    # Load pooled phi from master checkpoint (regularized)
    print("\n" + "="*80)
    print("LOADING REGULARIZED PHI FROM MASTER CHECKPOINT")
    print("="*80)
    master_ckpt = torch.load(MASTER_CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in master_ckpt and 'phi' in master_ckpt['model_state_dict']:
        phi_reg = master_ckpt['model_state_dict']['phi']
    elif 'phi' in master_ckpt:
        phi_reg = master_ckpt['phi']
    else:
        raise ValueError("No phi found in master checkpoint")
    
    if torch.is_tensor(phi_reg):
        phi_reg = phi_reg.detach().cpu().numpy()
    
    print(f"Regularized phi shape: {phi_reg.shape}")
    print(f"Regularized phi range: [{phi_reg.min():.4f}, {phi_reg.max():.4f}]")
    print(f"Regularized phi mean: {phi_reg.mean():.4f}")
    
    # Load pooled phi from unregularized batches
    print("\n" + "="*80)
    print("LOADING UNREGULARIZED PHI FROM BATCHES")
    print("="*80)
    phi_nolr, _ = pool_phi_from_batches(UNREGULARIZED_BATCH_PATTERN)
    print(f"Unregularized phi range: [{phi_nolr.min():.4f}, {phi_nolr.max():.4f}]")
    print(f"Unregularized phi mean: {phi_nolr.mean():.4f}")
    
    # Compare phis
    phi_stats = compare_phis(phi_reg, phi_nolr)
    
    # Load sample lambdas
    print("\n" + "="*80)
    print("LOADING SAMPLE LAMBDAS")
    print("="*80)
    lambdas_reg = load_sample_lambdas(REGULARIZED_BATCH_PATTERN, n_samples=5)
    lambdas_nolr = load_sample_lambdas(UNREGULARIZED_BATCH_PATTERN, n_samples=5)
    
    # Compare lambdas
    lambda_stats = compare_lambdas(lambdas_reg, lambdas_nolr)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Phi correlation: {phi_stats['correlation']:.6f}")
    print(f"Phi mean absolute difference: {phi_stats['mean_abs_diff']:.6f}")
    print(f"Phi relative difference: {phi_stats['rel_diff_pct']:.4f}%")
    
    if lambda_stats:
        print(f"Lambda correlation: {lambda_stats['correlation']:.6f}")
        print(f"Lambda mean absolute difference: {lambda_stats['mean_abs_diff']:.6f}")
        print(f"Mean individual lambda correlation: {lambda_stats['mean_individual_corr']:.6f}")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
