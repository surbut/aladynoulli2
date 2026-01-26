#!/usr/bin/env python
"""
Quick script to compare predictions from fixed gamma/kappa vs original predictions
Compares first 5 batches to check if ordering/correlations are preserved
"""

import torch
import numpy as np
from scipy.stats import pearsonr
import argparse

def compare_batch(old_path, new_path, batch_num, start, stop):
    """Compare predictions for a single batch"""
    try:
        pi_old = torch.load(old_path, weights_only=False)
        pi_new = torch.load(new_path, weights_only=False)
        
        print(f"\n{'='*60}")
        print(f"BATCH {batch_num}: samples {start} to {stop}")
        print(f"{'='*60}")
        print(f"Old pi shape: {pi_old.shape}")
        print(f"New pi shape: {pi_new.shape}")
        
        if pi_old.shape != pi_new.shape:
            print(f"⚠️  WARNING: Shape mismatch!")
            return None
        
        # Sample elements for correlation (to avoid memory issues)
        n_samples = min(10000, pi_old.numel())
        total_elements = pi_old.numel()
        
        if total_elements > n_samples:
            np.random.seed(42)
            sample_indices_flat = np.random.choice(total_elements, size=n_samples, replace=False)
            indices = np.unravel_index(sample_indices_flat, pi_old.shape)
            pi_old_sample = pi_old[indices].cpu().numpy().flatten()
            pi_new_sample = pi_new[indices].cpu().numpy().flatten()
        else:
            pi_old_sample = pi_old.flatten().cpu().numpy()
            pi_new_sample = pi_new.flatten().cpu().numpy()
        
        # Compute correlation
        corr, pval = pearsonr(pi_old_sample, pi_new_sample)
        
        # Also compute mean absolute difference
        mean_diff = np.mean(np.abs(pi_old_sample - pi_new_sample))
        max_diff = np.max(np.abs(pi_old_sample - pi_new_sample))
        
        print(f"Correlation: r = {corr:.6f}, p = {pval:.2e}")
        print(f"Mean absolute difference: {mean_diff:.6f}")
        print(f"Max absolute difference: {max_diff:.6f}")
        print(f"Mean old pi: {pi_old_sample.mean():.6f}")
        print(f"Mean new pi: {pi_new_sample.mean():.6f}")
        
        return {
            'correlation': corr,
            'pval': pval,
            'mean_diff': mean_diff,
            'max_diff': max_diff,
            'mean_old': pi_old_sample.mean(),
            'mean_new': pi_new_sample.mean()
        }
        
    except FileNotFoundError as e:
        print(f"✗ File not found: {e}")
        return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Compare fixed gamma/kappa predictions with original')
    parser.add_argument('--old_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_correctedE_vectorized/',
                       help='Directory with original predictions')
    parser.add_argument('--new_dir', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_fixedgk_vectorized/',
                       help='Directory with new fixed gamma/kappa predictions')
    parser.add_argument('--n_batches', type=int, default=5,
                       help='Number of batches to compare')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Comparing Predictions: Fixed Gamma/Kappa vs Original")
    print(f"{'='*60}")
    print(f"Old directory: {args.old_dir}")
    print(f"New directory: {args.new_dir}")
    print(f"Comparing first {args.n_batches} batches")
    print(f"{'='*60}")
    
    results = []
    batch_size = 10000
    
    for batch_num in range(1, args.n_batches + 1):
        start = (batch_num - 1) * batch_size
        stop = batch_num * batch_size
        
        old_path = f"{args.old_dir}/pi_enroll_fixedphi_sex_{start}_{stop}.pt"
        new_path = f"{args.new_dir}/pi_enroll_fixedphi_sex_{start}_{stop}.pt"
        
        result = compare_batch(old_path, new_path, batch_num, start, stop)
        if result:
            result['batch'] = batch_num
            results.append(result)
    
    # Summary
    if results:
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        correlations = [r['correlation'] for r in results]
        mean_diffs = [r['mean_diff'] for r in results]
        
        print(f"Mean correlation across batches: {np.mean(correlations):.6f} ± {np.std(correlations):.6f}")
        print(f"Min correlation: {np.min(correlations):.6f}")
        print(f"Max correlation: {np.max(correlations):.6f}")
        print(f"Mean absolute difference: {np.mean(mean_diffs):.6f} ± {np.std(mean_diffs):.6f}")
        print(f"{'='*60}\n")
    else:
        print("\n✗ No successful comparisons!")


if __name__ == '__main__':
    main()
