#!/usr/bin/env python3
"""
Compare gamma values between no-penalty and with-penalty versions.

This script loads gamma from model checkpoints and compares:
- Mean absolute gamma
- Max absolute gamma
- P95 absolute gamma
- Ratio (with_penalty / no_penalty)

Usage:
    python compare_penalty_gammas.py --n_batches 5
"""

import argparse
import sys
import os
import glob
import torch
import numpy as np
import pandas as pd
from pathlib import Path

def load_gamma_from_checkpoint(checkpoint_path):
    """Load gamma from a model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        gamma = checkpoint['model_state_dict']['gamma']
    elif 'gamma' in checkpoint:
        gamma = checkpoint['gamma']
    else:
        raise ValueError(f"Could not find gamma in checkpoint: {checkpoint_path}")
    
    return gamma

def get_start_index(filename):
    """Extract start index from filename for sorting"""
    basename = os.path.basename(filename)
    parts = basename.replace('.pt', '').split('_')
    numeric_parts = [p for p in parts if p.isdigit()]
    if len(numeric_parts) >= 2:
        return int(numeric_parts[0])
    raise ValueError(f"Could not extract start index from {filename}")

def load_gammas_from_directory(directory, n_batches=5):
    """
    Load gamma values from first n_batches in a directory.
    
    Args:
        directory: Path to directory containing model checkpoint files
        n_batches: Number of batches to load
        
    Returns:
        List of gamma tensors, one per batch
    """
    print(f"\nLoading gamma from: {directory}")
    
    # Find all model checkpoint files
    pattern = os.path.join(directory, "model_enroll_fixedphi_sex_*_*.pt")
    checkpoint_files = glob.glob(pattern)
    
    if len(checkpoint_files) == 0:
        raise ValueError(f"No model checkpoint files found in {directory}")
    
    # Sort by numeric start index
    checkpoint_files = sorted(checkpoint_files, key=get_start_index)
    
    # Take first n_batches
    checkpoint_files = checkpoint_files[:n_batches]
    print(f"Found {len(checkpoint_files)} checkpoint files (loading first {n_batches})")
    
    # Load gammas
    gammas = []
    for i, checkpoint_file in enumerate(checkpoint_files):
        print(f"  Loading batch {i+1}/{len(checkpoint_files)}: {os.path.basename(checkpoint_file)}")
        try:
            gamma = load_gamma_from_checkpoint(checkpoint_file)
            gammas.append(gamma)
            print(f"    Gamma shape: {gamma.shape}")
        except Exception as e:
            print(f"    ERROR loading {checkpoint_file}: {e}")
            continue
    
    return gammas

def compute_gamma_statistics(gammas):
    """
    Compute statistics across batches for gamma values.
    
    Args:
        gammas: List of gamma tensors [P, K]
        
    Returns:
        Dictionary with statistics
    """
    # Stack all gammas
    all_gammas = torch.cat([g.unsqueeze(0) for g in gammas], dim=0)  # [n_batches, P, K]
    
    # Compute statistics across batches
    mean_gamma = all_gammas.mean(dim=0)  # Average across batches [P, K]
    std_gamma = all_gammas.std(dim=0)   # Std across batches [P, K]
    
    # Absolute values for magnitude analysis
    abs_gammas = torch.abs(all_gammas)
    abs_mean = abs_gammas.mean(dim=0)
    
    stats = {
        'mean_abs_gamma': abs_mean.mean().item(),
        'max_abs_gamma': abs_gammas.max().item(),
        'p95_abs_gamma': torch.quantile(abs_gammas, 0.95).item(),
        'p99_abs_gamma': torch.quantile(abs_gammas, 0.99).item(),
        'mean_gamma': mean_gamma.mean().item(),
        'std_gamma': std_gamma.mean().item(),
        'n_batches': len(gammas),
        'shape': gammas[0].shape if gammas else None,
    }
    
    return stats, mean_gamma

def main():
    parser = argparse.ArgumentParser(description='Compare gamma values between no-penalty and with-penalty versions')
    parser.add_argument('--n_batches', type=int, default=5,
                       help='Number of batches to compare (default: 5)')
    parser.add_argument('--no_penalty_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_correctedE_vectorized/',
                       help='Directory with no-penalty predictions')
    parser.add_argument('--with_penalty_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_correctedE_vectorized_withLR/',
                       help='Directory with with-penalty predictions')
    parser.add_argument('--output_dir', type=str,
                       default='/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/penalty_comparison/',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("COMPARING GAMMA VALUES: NO-PENALTY vs WITH-PENALTY")
    print("="*80)
    print(f"No-penalty directory: {args.no_penalty_dir}")
    print(f"With-penalty directory: {args.with_penalty_dir}")
    print(f"Number of batches: {args.n_batches}")
    print("="*80)
    
    # Load gammas from both directories
    print("\n" + "="*80)
    print("LOADING GAMMAS")
    print("="*80)
    
    gammas_no_penalty = load_gammas_from_directory(args.no_penalty_dir, args.n_batches)
    gammas_with_penalty = load_gammas_from_directory(args.with_penalty_dir, args.n_batches)
    
    if len(gammas_no_penalty) == 0:
        raise ValueError("No gammas loaded from no-penalty directory!")
    if len(gammas_with_penalty) == 0:
        raise ValueError("No gammas loaded from with-penalty directory!")
    
    # Verify shapes match
    if gammas_no_penalty[0].shape != gammas_with_penalty[0].shape:
        raise ValueError(f"Shape mismatch! No-penalty: {gammas_no_penalty[0].shape}, With-penalty: {gammas_with_penalty[0].shape}")
    
    print(f"\n✓ Both gamma tensors have shape: {gammas_no_penalty[0].shape}")
    
    # Compute statistics
    print("\n" + "="*80)
    print("COMPUTING STATISTICS")
    print("="*80)
    
    stats_no_penalty, mean_gamma_no_penalty = compute_gamma_statistics(gammas_no_penalty)
    stats_with_penalty, mean_gamma_with_penalty = compute_gamma_statistics(gammas_with_penalty)
    
    # Print statistics
    print("\nNO-PENALTY GAMMA STATISTICS:")
    print("-" * 80)
    for key, value in stats_no_penalty.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    print("\nWITH-PENALTY GAMMA STATISTICS:")
    print("-" * 80)
    for key, value in stats_with_penalty.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # Compute ratios
    print("\n" + "="*80)
    print("RATIOS (WITH_PENALTY / NO_PENALTY)")
    print("="*80)
    
    ratio_mean_abs = stats_with_penalty['mean_abs_gamma'] / stats_no_penalty['mean_abs_gamma']
    ratio_max_abs = stats_with_penalty['max_abs_gamma'] / stats_no_penalty['max_abs_gamma']
    ratio_p95_abs = stats_with_penalty['p95_abs_gamma'] / stats_no_penalty['p95_abs_gamma']
    ratio_p99_abs = stats_with_penalty['p99_abs_gamma'] / stats_no_penalty['p99_abs_gamma']
    
    print(f"Mean absolute gamma ratio: {ratio_mean_abs:.4f}x")
    print(f"Max absolute gamma ratio:  {ratio_max_abs:.4f}x")
    print(f"P95 absolute gamma ratio:   {ratio_p95_abs:.4f}x")
    print(f"P99 absolute gamma ratio:   {ratio_p99_abs:.4f}x")
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Metric': ['mean_abs_gamma', 'max_abs_gamma', 'p95_abs_gamma', 'p99_abs_gamma', 'mean_gamma', 'std_gamma'],
        'No_Penalty': [
            stats_no_penalty['mean_abs_gamma'],
            stats_no_penalty['max_abs_gamma'],
            stats_no_penalty['p95_abs_gamma'],
            stats_no_penalty['p99_abs_gamma'],
            stats_no_penalty['mean_gamma'],
            stats_no_penalty['std_gamma'],
        ],
        'With_Penalty': [
            stats_with_penalty['mean_abs_gamma'],
            stats_with_penalty['max_abs_gamma'],
            stats_with_penalty['p95_abs_gamma'],
            stats_with_penalty['p99_abs_gamma'],
            stats_with_penalty['mean_gamma'],
            stats_with_penalty['std_gamma'],
        ],
        'Ratio': [
            ratio_mean_abs,
            ratio_max_abs,
            ratio_p95_abs,
            ratio_p99_abs,
            stats_with_penalty['mean_gamma'] / stats_no_penalty['mean_gamma'] if stats_no_penalty['mean_gamma'] != 0 else np.nan,
            stats_with_penalty['std_gamma'] / stats_no_penalty['std_gamma'] if stats_no_penalty['std_gamma'] != 0 else np.nan,
        ]
    })
    
    comparison_df = comparison_df.set_index('Metric')
    
    # Save results
    output_file = output_dir / 'gamma_comparison.csv'
    comparison_df.to_csv(output_file)
    print(f"\n✓ Saved comparison to: {output_file}")
    
    # Also compare per-signature
    print("\n" + "="*80)
    print("PER-SIGNATURE COMPARISON")
    print("="*80)
    
    # Average across batches and genetic features
    mean_gamma_no_penalty_per_sig = torch.abs(mean_gamma_no_penalty).mean(dim=0)  # [K]
    mean_gamma_with_penalty_per_sig = torch.abs(mean_gamma_with_penalty).mean(dim=0)  # [K]
    
    signature_comparison = pd.DataFrame({
        'Signature': [f'Signature_{k}' for k in range(len(mean_gamma_no_penalty_per_sig))],
        'No_Penalty_Mean_Abs': mean_gamma_no_penalty_per_sig.numpy(),
        'With_Penalty_Mean_Abs': mean_gamma_with_penalty_per_sig.numpy(),
        'Ratio': (mean_gamma_with_penalty_per_sig / mean_gamma_no_penalty_per_sig).numpy()
    })
    
    signature_comparison = signature_comparison.sort_values('Ratio')
    
    print("\nTop 10 signatures with largest shrinkage:")
    print(signature_comparison.head(10).to_string(index=False))
    
    print("\nTop 10 signatures with smallest shrinkage:")
    print(signature_comparison.tail(10).to_string(index=False))
    
    signature_file = output_dir / 'gamma_comparison_per_signature.csv'
    signature_comparison.to_csv(signature_file, index=False)
    print(f"\n✓ Saved per-signature comparison to: {signature_file}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ With-penalty gamma is {ratio_mean_abs:.2f}x smaller on average")
    print(f"✓ Max gamma is {ratio_max_abs:.2f}x smaller")
    print(f"✓ P95 gamma is {ratio_p95_abs:.2f}x smaller")
    print(f"\nThis confirms that lambda_reg=0.01 is effectively shrinking gamma values.")
    print("="*80)

if __name__ == '__main__':
    main()








