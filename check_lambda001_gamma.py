#!/usr/bin/env python3
"""
Compare gamma values between lambda_reg=0.01 (withLR directory) and no-penalty version.
This verifies that lambda_reg=0.01 is too weak to have any effect.
"""

import torch
import glob
import os

def get_start_index(filename):
    """Extract start index from filename for sorting"""
    basename = os.path.basename(filename)
    parts = basename.replace('.pt', '').split('_')
    numeric_parts = [p for p in parts if p.isdigit()]
    if len(numeric_parts) >= 2:
        return int(numeric_parts[0])
    raise ValueError(f"Could not extract start index from {filename}")

# Load first batch from both directories
no_penalty_dir = '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_correctedE_vectorized/'
with_penalty_dir = '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_correctedE_vectorized_withLR/'

# Find first batch file (0_10000)
no_penalty_pattern = os.path.join(no_penalty_dir, "model_enroll_fixedphi_sex_0_10000.pt")
with_penalty_pattern = os.path.join(with_penalty_dir, "model_enroll_fixedphi_sex_0_10000.pt")

print("="*80)
print("COMPARING: lambda_reg=0.01 vs No Penalty")
print("="*80)

# Load checkpoints
print(f"\nLoading no-penalty checkpoint: {no_penalty_pattern}")
no_penalty_model = torch.load(no_penalty_pattern, map_location='cpu', weights_only=False)

print(f"Loading with-penalty (lambda_reg=0.01) checkpoint: {with_penalty_pattern}")
with_penalty_model = torch.load(with_penalty_pattern, map_location='cpu', weights_only=False)

# Extract gamma
gamma_no_penalty = no_penalty_model['model_state_dict']['gamma']
gamma_with_penalty = with_penalty_model['model_state_dict']['gamma']

print("\n" + "="*80)
print("GAMMA STATISTICS")
print("="*80)

print("\nNo-penalty gamma stats:")
print(f"  Mean abs: {gamma_no_penalty.abs().mean():.6f}")
print(f"  Max abs: {gamma_no_penalty.abs().max():.6f}")
print(f"  Std: {gamma_no_penalty.std():.6f}")
print(f"  Shape: {gamma_no_penalty.shape}")

print("\nWith-penalty (lambda_reg=0.01) gamma stats:")
print(f"  Mean abs: {gamma_with_penalty.abs().mean():.6f}")
print(f"  Max abs: {gamma_with_penalty.abs().max():.6f}")
print(f"  Std: {gamma_with_penalty.std():.6f}")
print(f"  Shape: {gamma_with_penalty.shape}")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)

# Check if identical
are_identical = torch.allclose(gamma_no_penalty, gamma_with_penalty, atol=1e-6)
max_diff = (gamma_no_penalty - gamma_with_penalty).abs().max()
mean_diff = (gamma_no_penalty - gamma_with_penalty).abs().mean()

print(f"\nAre they identical (within 1e-6)? {are_identical}")
print(f"Max difference: {max_diff:.10f}")
print(f"Mean difference: {mean_diff:.10f}")

# Calculate ratios
ratio_mean = gamma_with_penalty.abs().mean() / gamma_no_penalty.abs().mean()
ratio_max = gamma_with_penalty.abs().max() / gamma_no_penalty.abs().max()

print(f"\nRatios (With-Penalty / No-Penalty):")
print(f"  Mean abs ratio: {ratio_mean:.6f}x")
print(f"  Max abs ratio:  {ratio_max:.6f}x")

# Per-signature comparison
print("\n" + "="*80)
print("PER-SIGNATURE COMPARISON (Mean Absolute Gamma)")
print("="*80)
print(f"{'Signature':<12} {'No-Penalty':<15} {'With-Penalty':<15} {'Ratio':<10} {'Max Diff':<12}")
print("-" * 80)

max_diffs_per_sig = []
for k in range(gamma_no_penalty.shape[1]):
    no_penalty_mean = gamma_no_penalty[:, k].abs().mean()
    with_penalty_mean = gamma_with_penalty[:, k].abs().mean()
    ratio = with_penalty_mean / no_penalty_mean if no_penalty_mean > 0 else float('inf')
    max_diff_sig = (gamma_no_penalty[:, k] - gamma_with_penalty[:, k]).abs().max()
    max_diffs_per_sig.append(max_diff_sig.item())
    
    print(f"Signature {k:<3} {no_penalty_mean:<15.6f} {with_penalty_mean:<15.6f} {ratio:<10.6f} {max_diff_sig:<12.10f}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

if are_identical or max_diff < 1e-5:
    print("✓ CONFIRMED: lambda_reg=0.01 has NO EFFECT")
    print("  Gamma values are essentially identical to no-penalty version")
    print("  This confirms that lambda_reg=0.01 is too weak with lr=0.1")
elif ratio_mean > 0.95:
    print("⚠ lambda_reg=0.01 has MINIMAL EFFECT")
    print(f"  Gamma shrunk by only {1/ratio_mean:.2f}x (essentially unchanged)")
    print("  This confirms that lambda_reg=0.01 is too weak with lr=0.1")
else:
    print(f"✗ lambda_reg=0.01 DOES have an effect")
    print(f"  Gamma shrunk by {1/ratio_mean:.2f}x")
    print(f"  Max difference: {max_diff:.10f}")

print(f"\nMax difference across all signatures: {max(max_diffs_per_sig):.10f}")
print(f"Mean difference across all signatures: {sum(max_diffs_per_sig)/len(max_diffs_per_sig):.10f}")

