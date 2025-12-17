#!/usr/bin/env python3
"""
Compare gamma values from first few batches between no-penalty and with-penalty versions.
This confirms that lambda_reg=0.01 is actually shrinking gamma values.
"""

import torch
import glob
import os
import numpy as np

def get_start_index(filename):
    """Extract start index from filename for sorting"""
    basename = os.path.basename(filename)
    parts = basename.replace('.pt', '').split('_')
    numeric_parts = [p for p in parts if p.isdigit()]
    if len(numeric_parts) >= 2:
        return int(numeric_parts[0])
    raise ValueError(f"Could not extract start index from {filename}")

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

# Directories
no_penalty_dir = '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_correctedE_vectorized/'
with_penalty_dir = '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_correctedE_vectorized_withLR/'

# Number of batches to compare
n_batches = 5

print("="*80)
print(f"COMPARING GAMMAS: First {n_batches} Batches")
print("="*80)
print(f"No-penalty directory: {no_penalty_dir}")
print(f"With-penalty directory: {with_penalty_dir}")
print("="*80)

# Find checkpoint files
no_penalty_pattern = os.path.join(no_penalty_dir, "model_enroll_fixedphi_sex_*_*.pt")
with_penalty_pattern = os.path.join(with_penalty_dir, "model_enroll_fixedphi_sex_*_*.pt")

no_penalty_files = sorted(glob.glob(no_penalty_pattern), key=get_start_index)
with_penalty_files = sorted(glob.glob(with_penalty_pattern), key=get_start_index)

if len(no_penalty_files) == 0:
    raise ValueError(f"No checkpoint files found in {no_penalty_dir}")
if len(with_penalty_files) == 0:
    raise ValueError(f"No checkpoint files found in {with_penalty_dir}")

# Take first n_batches
no_penalty_files = no_penalty_files[:n_batches]
with_penalty_files = with_penalty_files[:n_batches]

print(f"\nFound {len(no_penalty_files)} no-penalty batches and {len(with_penalty_files)} with-penalty batches")
print(f"Comparing first {n_batches} batches\n")

# Load and compare gammas from each batch
batch_comparisons = []

for i, (no_penalty_file, with_penalty_file) in enumerate(zip(no_penalty_files, with_penalty_files)):
    batch_name = os.path.basename(no_penalty_file)
    print(f"\n{'='*80}")
    print(f"BATCH {i+1}/{n_batches}: {batch_name}")
    print(f"{'='*80}")
    
    # Load gammas
    gamma_no_penalty = load_gamma_from_checkpoint(no_penalty_file)
    gamma_with_penalty = load_gamma_from_checkpoint(with_penalty_file)
    
    # Compute statistics
    stats_no = {
        'mean_abs': gamma_no_penalty.abs().mean().item(),
        'max_abs': gamma_no_penalty.abs().max().item(),
        'std': gamma_no_penalty.std().item(),
    }
    
    stats_with = {
        'mean_abs': gamma_with_penalty.abs().mean().item(),
        'max_abs': gamma_with_penalty.abs().max().item(),
        'std': gamma_with_penalty.std().item(),
    }
    
    # Ratios
    ratio_mean = stats_with['mean_abs'] / stats_no['mean_abs']
    ratio_max = stats_with['max_abs'] / stats_no['max_abs']
    max_diff = (gamma_no_penalty - gamma_with_penalty).abs().max().item()
    are_identical = torch.allclose(gamma_no_penalty, gamma_with_penalty, atol=1e-6)
    
    print(f"\nNo-penalty gamma:")
    print(f"  Mean abs: {stats_no['mean_abs']:.6f}")
    print(f"  Max abs:  {stats_no['max_abs']:.6f}")
    print(f"  Std:      {stats_no['std']:.6f}")
    
    print(f"\nWith-penalty gamma:")
    print(f"  Mean abs: {stats_with['mean_abs']:.6f}")
    print(f"  Max abs:  {stats_with['max_abs']:.6f}")
    print(f"  Std:      {stats_with['std']:.6f}")
    
    print(f"\nComparison:")
    print(f"  Mean abs ratio: {ratio_mean:.4f}x")
    print(f"  Max abs ratio:  {ratio_max:.4f}x")
    print(f"  Max difference: {max_diff:.10f}")
    print(f"  Are identical:  {are_identical}")
    
    batch_comparisons.append({
        'batch': batch_name,
        'ratio_mean': ratio_mean,
        'ratio_max': ratio_max,
        'max_diff': max_diff,
        'are_identical': are_identical,
        'no_penalty_mean_abs': stats_no['mean_abs'],
        'with_penalty_mean_abs': stats_with['mean_abs'],
    })

# Summary across all batches
print("\n" + "="*80)
print("SUMMARY ACROSS ALL BATCHES")
print("="*80)

mean_ratios = [b['ratio_mean'] for b in batch_comparisons]
max_ratios = [b['ratio_max'] for b in batch_comparisons]
max_diffs = [b['max_diff'] for b in batch_comparisons]

print(f"\nMean abs ratios (with-penalty / no-penalty):")
for i, comp in enumerate(batch_comparisons):
    print(f"  Batch {i+1}: {comp['ratio_mean']:.4f}x")

print(f"\nOverall statistics:")
print(f"  Mean ratio (across batches): {np.mean(mean_ratios):.4f}x")
print(f"  Median ratio: {np.median(mean_ratios):.4f}x")
print(f"  Min ratio: {np.min(mean_ratios):.4f}x")
print(f"  Max ratio: {np.max(mean_ratios):.4f}x")
print(f"  Max difference across all batches: {max(max_diffs):.10f}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if all(not b['are_identical'] for b in batch_comparisons) and np.mean(mean_ratios) < 0.5:
    print("✓ CONFIRMED: lambda_reg=0.01 is shrinking gamma values")
    print(f"  Average shrinkage: {1/np.mean(mean_ratios):.2f}x")
    print("  Gammas are significantly different between versions")
elif np.mean(mean_ratios) > 0.9:
    print("✗ WARNING: Gammas are very similar - penalty may not be working")
    print(f"  Average ratio: {np.mean(mean_ratios):.4f}x (should be < 0.5)")
else:
    print(f"⚠ Moderate difference: Average ratio = {np.mean(mean_ratios):.4f}x")
    print(f"  Shrinkage: {1/np.mean(mean_ratios):.2f}x")

print("="*80)

